import argparse
import gc
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
    NUM_INFERENCE_STEPS,
    EARLY_SEG,
    MID_SEG,
    SEGMENT_TIMESTEP_RANGES,
)
from .dataset import get_samples
from .encoder import (
    compute_perceptual_discrepancy,
    SubjectMaskBuilder,
    CrossAttentionCapture,
)
from .encoding import encode_prompt, rms_scaled_noise, HIDDEN_SPLIT
from .scheduler import (
    SpatiallyVaryingDDPMScheduler,
    compute_spatial_noise_scale,
    pyramid_noise,
)
from .loss import compute_diffusion_loss
from .cache import (
    build_cache,
    blend_alpha,
    blend_masks,
    stack_added_cond,
    BucketSampler,
)
from .rl import rl_segment
from .io import save_lora

LATE_START = EARLY_SEG + MID_SEG

SEGMENTS = ["early", "mid", "late"]

SEGMENT_INDICES = {
    "early": range(0, EARLY_SEG),
    "mid": range(EARLY_SEG, LATE_START),
    "late": range(LATE_START, NUM_INFERENCE_STEPS),
}

DISCREPANCY_MAX_T = 600


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
    t_low, t_high = SEGMENT_TIMESTEP_RANGES[segment_name]

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
            text_emb = torch.cat([x["text_emb"] for x in items]).to(
                DEVICE, dtype=torch.float16
            )
            token_content = torch.cat([x["token_content_mask"] for x in items]).to(
                DEVICE
            )
            pooled, time_ids = stack_added_cond(items, DEVICE)

            batch = latents.shape[0]
            spatial_size = (latents.shape[2], latents.shape[3])

            text_emb, pooled, token_content = apply_conditioning_augmentation(
                text_emb, pooled, token_content, empty_emb, empty_pooled, cfg
            )

            if cfg.embed_jitter_max > 0.0:
                jitter = random.uniform(0.0, cfg.embed_jitter_max)
                text_emb = text_emb + rms_scaled_noise(
                    text_emb, jitter, split=HIDDEN_SPLIT
                )
                pooled = pooled + rms_scaled_noise(pooled, jitter)

            added_cond_kwargs = {"text_embeds": pooled, "time_ids": time_ids}

            noise = pyramid_noise(
                latents, levels=cfg.noise_lf_levels, decay=cfg.noise_lf_decay
            )
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
                del attn_mask, raw_diff

            blur_sigma = mask_builder.blur_sigma_for_step(step, cfg.train_steps)
            mask = mask_builder.build_mask(mask, blur_sigma)

            noise_scale = compute_spatial_noise_scale(
                mask,
                t.float() / 1000.0,
                bg_boost=cfg.noise_bg_boost,
                t_ramp=cfg.noise_t_ramp,
            )
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
            )

            # pyrefly: ignore [missing-attribute]
            scaler.scale(loss / cfg.grad_accum_steps).backward()
            accum_loss += loss.item()

            del noisy_latents, noise_scale, mask, base_pred, uniform_noisy

        if cfg.grad_clip_norm > 0.0:
            # pyrefly: ignore [missing-attribute]
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        pbar.set_postfix(loss=f"{accum_loss / cfg.grad_accum_steps:.4f}")

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
    )

    empty_emb, empty_pooled, _ = encode_prompt("", text_encoders, tokenizers, DEVICE)
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
            f"Raise --cache_size, or lower --mini_batch_size / --rl_refs."
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

    print(
        f"Config: steps={cfg.steps}, mini_batch={cfg.mini_batch_size}, "
        f"grad_accum={cfg.grad_accum_steps}, train_steps={cfg.train_steps}, "
        f"cache_size={cfg.cache_size}, shuffle_buffer={cfg.shuffle_buffer}, "
        f"refresh_per_segment="
        f"{cfg.refresh_cache_per_segment}, lr={cfg.lr}, lora_rank={cfg.lora_rank}, "
        f"lora_alpha={cfg.lora_alpha}, bg_boost={cfg.noise_bg_boost}, "
        f"lf_levels={cfg.noise_lf_levels}, lf_decay={cfg.noise_lf_decay}, "
        f"snr_gamma={cfg.snr_gamma}, cond_partial={cfg.cond_partial_prob}, "
        f"mask_gain={cfg.mask_gain}, "
        f"caption_subset={cfg.caption_subset_prob}/{cfg.caption_subset_min}, "
        f"rl_steps={cfg.rl_steps}, rl_lr={cfg.rl_lr}, rl_refs={cfg.rl_refs}, "
        f"rl_group={cfg.rl_group}, output_dir={cfg.output_dir}"
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
    timesteps = scheduler.timesteps.to(DEVICE)

    needs_base = cfg.rl_steps > 0 and cfg.rl_grounding_weight > 0.0
    base_unet = None
    if needs_base:
        # pyrefly: ignore [missing-attribute]
        base_unet = models["pipe"].unet.eval()
        for param in base_unet.parameters():
            param.requires_grad_(False)
        base_unet.to("cpu")
    else:
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
    if cfg.rl_steps > 0:
        min_size = max(min_size, cfg.rl_refs)

    sampler, empty_emb, empty_pooled = build_sampler(
        vae, text_encoders, tokenizers, min_size, 0, cfg, models["zero_uncond"]
    )

    for index, segment_name in enumerate(SEGMENTS):
        if index > 0 and cfg.refresh_cache_per_segment:
            sampler = None
            gc.collect()
            torch.cuda.empty_cache()
            sampler, empty_emb, empty_pooled = build_sampler(
                vae,
                text_encoders,
                tokenizers,
                min_size,
                index,
                cfg,
                models["zero_uncond"],
            )

        unet = train_segment(
            segment_name=segment_name,
            scheduler=scheduler,
            mask_builder=mask_builder,
            sampler=sampler,
            empty_emb=empty_emb,
            empty_pooled=empty_pooled,
            cfg=cfg,
        )

        unet = rl_segment(
            segment_name=segment_name,
            unet=unet,
            base_unet=base_unet,
            t_indices=SEGMENT_INDICES[segment_name],
            timesteps=timesteps,
            scheduler=scheduler,
            vae=vae,
            sampler=sampler,
            cfg=cfg,
        )

        save_lora(unet, cfg.output_dir, segment_name)

        del unet
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train(get_config())
