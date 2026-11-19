import argparse
import gc
import math
import os
import random

import torch
import bitsandbytes as bnb
from tqdm import tqdm

from peft import get_peft_model

from .config import get_config
from .model import (
    load_model,
    load_unet,
    get_lora_config,
    DEVICE,
    MODEL_ID,
    NUM_INFERENCE_STEPS,
    SEGMENT_TIMESTEP_RANGES,
    SINGLE_ADAPTER,
    SINGLE_ADAPTER_RANGE,
    VAE_ID,
)
from .dataset import BUCKETS, DATASET_ID, get_samples
from .encoder import (
    compute_perceptual_discrepancy,
    SubjectMaskBuilder,
    CrossAttentionCapture,
)
from .encoding import (
    CHUNK_TOKENS,
    HIDDEN_SPLIT,
    MAX_CHUNKS,
    encode_prompt,
    rms_scaled_noise,
)
from .scheduler import (
    SpatiallyVaryingDDPMScheduler,
    compute_spatial_noise_scale,
    lowfreq_noise,
)
from .loss import compute_diffusion_loss
from .cache import (
    build_cache,
    blend_alpha,
    blend_masks,
    cache_key,
    load_cache,
    pack_text,
    save_cache,
    stack_added_cond,
    stack_text,
    BucketSampler,
)
from .io import save_lora

SEGMENTS = ["early", "mid", "late"]
TIMESTEP_RANGES = {**SEGMENT_TIMESTEP_RANGES, SINGLE_ADAPTER: SINGLE_ADAPTER_RANGE}

DISCREPANCY_MAX_T = 600


def segment_plan(segments: list[str], refresh: bool) -> list[tuple[int, str, int]]:
    if segments == [SINGLE_ADAPTER]:
        return [(0, SINGLE_ADAPTER, 0)]
    unknown = [s for s in segments if s not in SEGMENTS]
    if unknown or not segments:
        raise ValueError(
            f"--segments must be {SINGLE_ADAPTER} alone or a non-empty subset of "
            f"{SEGMENTS}, got {segments}"
        )
    return [
        (index, name, index if refresh else 0)
        for index, name in enumerate(SEGMENTS)
        if name in segments
    ]


def cache_path(cfg: argparse.Namespace, seed: int) -> str:
    key = cache_key(
        dataset=DATASET_ID,
        model=MODEL_ID,
        vae=VAE_ID,
        buckets=BUCKETS,
        chunk_tokens=CHUNK_TOKENS,
        max_chunks=MAX_CHUNKS,
        seed=seed,
        cache_size=cfg.cache_size,
        shuffle_buffer=cfg.shuffle_buffer,
        caption_subset_prob=cfg.caption_subset_prob,
        caption_subset_min=cfg.caption_subset_min,
    )
    name = f"latents-seed{seed}-n{cfg.cache_size}-{key}.pt"
    return os.path.join(cfg.cache_dir, name)


def _make_inputs_require_grad(module, input, output):  # noqa: ARG001
    output.requires_grad_(True)


def apply_conditioning_augmentation(
    text_emb: torch.Tensor,
    pooled: torch.Tensor,
    token_content: torch.Tensor,
    empty_emb: torch.Tensor,
    empty_pooled: torch.Tensor,
    cfg: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = text_emb.shape[0]
    if cfg.cond_dropout_prob <= 0.0 and cfg.cond_partial_prob <= 0.0:
        return text_emb, pooled, token_content

    roll = torch.rand(batch, device=text_emb.device)
    drop = roll < cfg.cond_dropout_prob
    partial = (roll >= cfg.cond_dropout_prob) & (
        roll < cfg.cond_dropout_prob + cfg.cond_partial_prob
    )

    if not bool((drop | partial).any()):
        return text_emb, pooled, token_content

    scale = torch.ones(batch, device=text_emb.device)
    scale = torch.where(drop, torch.zeros_like(scale), scale)
    scale = torch.where(
        partial,
        torch.rand(batch, device=text_emb.device) * cfg.cond_partial_max,
        scale,
    )

    empty = empty_emb.expand(batch, -1, -1).to(text_emb.dtype)
    blended = torch.lerp(empty.float(), text_emb.float(), scale.view(-1, 1, 1))

    empty_p = empty_pooled.expand(batch, -1).to(pooled.dtype)
    blended_pooled = torch.lerp(empty_p.float(), pooled.float(), scale.view(-1, 1))

    return (
        blended.to(text_emb.dtype),
        blended_pooled.to(pooled.dtype),
        token_content * scale.view(-1, 1),
    )


def finite_stages(**tensors: torch.Tensor) -> dict:
    return {name: torch.isfinite(x).all() for name, x in tensors.items()}


def first_nonfinite(finite: dict) -> str:
    for name, ok in finite.items():
        if not bool(ok):
            return name
    return "the LoRA forward"


def train_segment(
    segment_name,
    scheduler,
    mask_builder,
    sampler,
    empty_emb,
    empty_pooled,
    cfg: argparse.Namespace,
):
    print(f"\nTraining segment: {segment_name}")
    t_low, t_high = TIMESTEP_RANGES[segment_name]

    # pyrefly: ignore [bad-argument-type]
    unet = get_peft_model(load_unet(), get_lora_config(cfg.lora_rank, cfg.lora_alpha))
    unet.to(DEVICE)
    torch.cuda.empty_cache()

    unet.base_model.model.conv_in.register_forward_hook(_make_inputs_require_grad)
    # pyrefly: ignore [not-callable]
    unet.enable_gradient_checkpointing()
    unet.train()

    trainable = [p for p in unet.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda")

    capture = CrossAttentionCapture(unet, gain=cfg.mask_gain)

    for step in (pbar := tqdm(range(cfg.train_steps))):
        optimizer.zero_grad(set_to_none=True)

        accum_loss = 0.0
        alpha = blend_alpha(step, cfg.train_steps)

        for _ in range(cfg.grad_accum_steps):
            items = sampler.draw(cfg.mini_batch_size)

            latents = torch.cat([x["latents"] for x in items]).float().to(DEVICE)
            text_emb, token_content, token_valid = stack_text(items, DEVICE)
            pooled, time_ids = stack_added_cond(items, DEVICE)

            batch = latents.shape[0]
            spatial_size = (latents.shape[2], latents.shape[3])

            text_emb, pooled, token_content = apply_conditioning_augmentation(
                text_emb, pooled, token_content, empty_emb, empty_pooled, cfg
            )
            text_emb, encoder_mask, token_content = pack_text(
                text_emb, token_valid, token_content
            )

            if cfg.embed_jitter_max > 0.0:
                jitter = random.uniform(0.0, cfg.embed_jitter_max)
                text_emb = text_emb + rms_scaled_noise(
                    text_emb, jitter, split=HIDDEN_SPLIT
                )
                pooled = pooled + rms_scaled_noise(pooled, jitter)

            added_cond_kwargs = {"text_embeds": pooled, "time_ids": time_ids}

            noise = lowfreq_noise(latents, cfg.noise_lf_strength)
            t = torch.randint(
                t_low, t_high + 1, (batch,), device=DEVICE, dtype=torch.long
            )

            with torch.no_grad():
                uniform_noisy = scheduler.add_noise(latents, noise, t)

                capture.set_context(token_content, spatial_size)
                with capture, unet.disable_adapter():
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        base_pred = unet(
                            uniform_noisy.to(dtype=torch.float16),
                            t,
                            encoder_hidden_states=text_emb,
                            added_cond_kwargs=added_cond_kwargs,
                            encoder_attention_mask=encoder_mask,
                        ).sample.float()

                    attn_mask = capture.build_mask(spatial_size)

                denoised_latents = scheduler.predict_x0(
                    uniform_noisy.float(), base_pred, t
                )
                raw_diff = compute_perceptual_discrepancy([denoised_latents], [latents])
                del denoised_latents

                gate = (t < DISCREPANCY_MAX_T).float().view(-1, 1, 1, 1)
                mask = blend_masks(
                    attn_mask, raw_diff, spatial_size, alpha * gate, gain=cfg.mask_gain
                )
                finite = finite_stages(
                    base_pred=base_pred,
                    attn_mask=attn_mask,
                    discrepancy=torch.where(gate > 0, raw_diff, 0.0),
                    blended_mask=mask,
                )
                del attn_mask, raw_diff

            blur_sigma = mask_builder.blur_sigma_for_step(step, cfg.train_steps)
            mask = mask_builder.build_mask(mask, blur_sigma)

            noise_scale = compute_spatial_noise_scale(
                mask,
                t.float() / 1000.0,
                bg_boost=cfg.noise_bg_boost,
                t_ramp=cfg.noise_t_ramp,
            )
            finite.update(finite_stages(noise_scale=noise_scale))
            noisy_latents = scheduler.add_noise(
                latents, noise, t, noise_scale=noise_scale
            )

            loss = compute_diffusion_loss(
                unet,
                noisy_latents,
                t,
                text_emb,
                noise,
                added_cond_kwargs=added_cond_kwargs,
                noise_scale=noise_scale,
                mask=mask,
                bg_weight=cfg.loss_bg_weight,
                alphas_cumprod=scheduler.alphas_cumprod,
                snr_gamma=cfg.snr_gamma,
                encoder_attention_mask=encoder_mask,
            )

            # pyrefly: ignore [missing-attribute]
            scaler.scale(loss / cfg.grad_accum_steps).backward()
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print(
                    f"\nNon-finite loss at step {step}: "
                    f"{first_nonfinite(finite)} went non-finite first; "
                    f"t={t.tolist()}, latent {tuple(latents.shape[2:])}"
                )
            accum_loss += loss_value

            del noisy_latents, noise_scale, mask, base_pred, uniform_noisy

        if cfg.grad_clip_norm > 0.0:
            # pyrefly: ignore [missing-attribute]
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        pbar.set_postfix(loss=f"{accum_loss / cfg.grad_accum_steps:.4f}")

        done = step + 1
        if cfg.save_every and done % cfg.save_every == 0 and done < cfg.train_steps:
            save_lora(unet, os.path.join(cfg.output_dir, f"step{done}"), segment_name)

        if step % 20 == 0:
            torch.cuda.empty_cache()

    del optimizer, scaler
    torch.cuda.empty_cache()

    return unet


def build_sampler(
    vae, text_encoders, tokenizers, min_size, seed, cfg, zero_uncond=True
):
    for encoder in text_encoders:
        encoder.to(DEVICE)

    path = cfg.cache_file or (cache_path(cfg, seed) if cfg.cache_dir else "")
    if cfg.cache_file and not os.path.exists(path):
        raise FileNotFoundError(f"--cache_file {path} does not exist")
    if path and os.path.exists(path):
        print(f"Loading latent cache (seed={seed}) from {path}...")
        cached = load_cache(path)
    else:
        print(f"Building latent cache (seed={seed})...")
        cached = build_cache(
            get_samples(cfg.cache_size, seed=seed, shuffle_buffer=cfg.shuffle_buffer),
            vae,
            text_encoders,
            tokenizers,
            DEVICE,
            total=cfg.cache_size,
            subset_prob=cfg.caption_subset_prob,
            subset_min=cfg.caption_subset_min,
            seed=seed,
            workers=cfg.cache_workers,
        )
        if path:
            save_cache(cached, path)
            print(f"Saved latent cache to {path}")

    empty_emb, empty_pooled, _, _ = encode_prompt("", text_encoders, tokenizers, DEVICE)
    empty_emb = empty_emb.detach().to(dtype=torch.float16)
    empty_pooled = empty_pooled.detach().to(dtype=torch.float16)

    if zero_uncond:
        empty_emb = torch.zeros_like(empty_emb)
        empty_pooled = torch.zeros_like(empty_pooled)

    for encoder in text_encoders:
        encoder.to("cpu")
    torch.cuda.empty_cache()

    sampler = BucketSampler(cached, min_size)
    if not sampler:
        raise RuntimeError(
            f"No aspect bucket holds at least {min_size} samples. "
            f"Raise --cache_size, or lower --mini_batch_size."
        )

    print(
        f"Cached {len(cached)} samples; {sampler.usable} usable "
        f"across {len(sampler.buckets)} buckets:"
    )
    for bucket, idxs in sorted(sampler.groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  {bucket[0]:>4}x{bucket[1]:<4}  {len(idxs)}")

    return sampler, empty_emb, empty_pooled


def train(cfg: argparse.Namespace):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    plan = segment_plan(cfg.segments, cfg.refresh_cache_per_segment)
    if cfg.cache_only and not cfg.cache_dir:
        raise ValueError("--cache_only 1 needs --cache_dir to save the cache into")

    print(
        f"Config: segments={cfg.segments}, cache_dir={cfg.cache_dir or None}, "
        f"cache_file={cfg.cache_file or None}, "
        f"steps={cfg.steps}, mini_batch={cfg.mini_batch_size}, "
        f"grad_accum={cfg.grad_accum_steps}, train_steps={cfg.train_steps}, "
        f"cache_size={cfg.cache_size}, shuffle_buffer={cfg.shuffle_buffer}, "
        f"refresh_per_segment="
        f"{cfg.refresh_cache_per_segment}, lr={cfg.lr}, lora_rank={cfg.lora_rank}, "
        f"lora_alpha={cfg.lora_alpha}, bg_boost={cfg.noise_bg_boost}, "
        f"lf_strength={cfg.noise_lf_strength}, "
        f"snr_gamma={cfg.snr_gamma}, cond_partial={cfg.cond_partial_prob}, "
        f"mask_gain={cfg.mask_gain}, "
        f"caption_subset={cfg.caption_subset_prob}/{cfg.caption_subset_min}, "
        f"save_every={cfg.save_every}, cache_only={cfg.cache_only}, "
        f"output_dir={cfg.output_dir}"
    )

    models = load_model()
    vae = models["vae"]
    text_encoders = models["text_encoders"]
    tokenizers = models["tokenizers"]

    scheduler = SpatiallyVaryingDDPMScheduler.from_config(
        # pyrefly: ignore [missing-attribute]
        models["pipe"].scheduler.config
    )
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)

    # pyrefly: ignore [missing-attribute]
    models["pipe"].unet = None
    torch.cuda.empty_cache()

    mask_builder = SubjectMaskBuilder(
        blur_sigma_start=cfg.mask_blur_sigma_start,
        blur_sigma_end=cfg.mask_blur_sigma_end,
        min_mask_value=cfg.mask_min_value,
        gain=cfg.mask_gain,
    )

    min_size = cfg.mini_batch_size

    if cfg.cache_only:
        for cache_seed in dict.fromkeys(seed for _, _, seed in plan):
            build_sampler(
                vae,
                text_encoders,
                tokenizers,
                min_size,
                cache_seed,
                cfg,
                models["zero_uncond"],
            )
            gc.collect()
        return

    built = None
    for _, segment_name, cache_seed in plan:
        if built is None or built[0] != cache_seed:
            built = sampler = None
            gc.collect()
            torch.cuda.empty_cache()
            built = (
                cache_seed,
                *build_sampler(
                    vae,
                    text_encoders,
                    tokenizers,
                    min_size,
                    cache_seed,
                    cfg,
                    models["zero_uncond"],
                ),
            )
        _, sampler, empty_emb, empty_pooled = built

        unet = train_segment(
            segment_name=segment_name,
            scheduler=scheduler,
            mask_builder=mask_builder,
            sampler=sampler,
            empty_emb=empty_emb,
            empty_pooled=empty_pooled,
            cfg=cfg,
        )

        save_lora(unet, cfg.output_dir, segment_name)

        del unet
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train(get_config())
