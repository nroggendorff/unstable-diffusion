import argparse
import copy
import random

import torch
import bitsandbytes as bnb
from tqdm import tqdm

from peft import get_peft_model

from .config import get_config
from .model import (
    load_model,
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
from .encoding import encode_prompt, rms_scaled_noise
from .scheduler import SpatiallyVaryingDDPMScheduler, compute_spatial_noise_scale
from .loss import compute_diffusion_loss
from .cache import (
    build_cache,
    blend_alpha,
    blend_masks,
    group_by_bucket,
    sample_minibatch,
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


def train_segment(
    segment_name,
    base_unet,
    scheduler,
    mask_builder,
    cached,
    groups,
    empty_emb,
    cfg: argparse.Namespace,
):
    print(f"\nTraining segment: {segment_name}")
    t_low, t_high = SEGMENT_TIMESTEP_RANGES[segment_name]

    base_unet.to(DEVICE)
    unet = get_peft_model(
        copy.deepcopy(base_unet), get_lora_config(cfg.lora_rank, cfg.lora_alpha)
    )
    base_unet.to("cpu")
    torch.cuda.empty_cache()

    unet.base_model.model.conv_in.register_forward_hook(_make_inputs_require_grad)
    # pyrefly: ignore [not-callable]
    unet.enable_gradient_checkpointing()
    unet.train()

    optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda")

    capture = CrossAttentionCapture(unet)

    for step in (pbar := tqdm(range(cfg.train_steps))):
        optimizer.zero_grad(set_to_none=True)

        accum_loss = 0.0
        alpha = blend_alpha(step, cfg.train_steps)

        for _ in range(cfg.grad_accum_steps):
            items = sample_minibatch(cached, groups, cfg.mini_batch_size)

            latents = torch.cat([x["latents"] for x in items]).float().to(DEVICE)
            text_emb = torch.cat([x["text_emb"] for x in items]).to(
                DEVICE, dtype=torch.float16
            )
            token_content = torch.cat([x["token_content_mask"] for x in items]).to(
                DEVICE
            )

            batch = latents.shape[0]
            spatial_size = (latents.shape[2], latents.shape[3])

            if cfg.cond_dropout_prob > 0.0:
                drop = torch.rand(batch, device=DEVICE) < cfg.cond_dropout_prob
                if drop.any():
                    empty = empty_emb.expand(batch, -1, -1).to(text_emb.dtype)
                    text_emb = torch.where(drop.view(-1, 1, 1), empty, text_emb)
                    token_content = token_content * (~drop).float().view(-1, 1)

            if cfg.embed_jitter_max > 0.0:
                text_emb = text_emb + rms_scaled_noise(
                    text_emb, random.uniform(0.0, cfg.embed_jitter_max)
                )

            noise = torch.randn_like(latents)
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
                        ).sample.float()

                    attn_mask = capture.build_mask(spatial_size)

                denoised_latents = scheduler.predict_x0(
                    uniform_noisy.float(), base_pred, t
                )
                raw_diff = compute_perceptual_discrepancy([denoised_latents], [latents])
                del denoised_latents

                gate = (t < DISCREPANCY_MAX_T).float().view(-1, 1, 1, 1)
                mask = blend_masks(attn_mask, raw_diff, spatial_size, alpha * gate)
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
                noise_scale=noise_scale,
                mask=mask,
                bg_weight=cfg.loss_bg_weight,
            )

            # pyrefly: ignore [missing-attribute]
            scaler.scale(loss / cfg.grad_accum_steps).backward()
            accum_loss += loss.item()

            del noisy_latents, noise_scale, mask, base_pred, uniform_noisy

        scaler.step(optimizer)
        scaler.update()

        pbar.set_postfix(loss=f"{accum_loss / cfg.grad_accum_steps:.4f}")

        if step % 20 == 0:
            torch.cuda.empty_cache()

    del optimizer, scaler
    torch.cuda.empty_cache()

    return unet


def train(cfg: argparse.Namespace):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(
        f"Config: steps={cfg.steps}, mini_batch={cfg.mini_batch_size}, "
        f"grad_accum={cfg.grad_accum_steps}, train_steps={cfg.train_steps}, "
        f"cache_size={cfg.cache_size}, lr={cfg.lr}, lora_rank={cfg.lora_rank}, "
        f"lora_alpha={cfg.lora_alpha}, bg_boost={cfg.noise_bg_boost}, "
        f"rl_steps={cfg.rl_steps}, rl_lr={cfg.rl_lr}, output_dir={cfg.output_dir}"
    )

    models = load_model()
    vae = models["vae"]
    text_encoder = models["text_encoder"]
    tokenizer = models["tokenizer"]

    # pyrefly: ignore [missing-attribute]
    base_unet = models["pipe"].unet.eval()
    for param in base_unet.parameters():
        param.requires_grad_(False)
    base_unet.to("cpu")
    torch.cuda.empty_cache()

    scheduler = SpatiallyVaryingDDPMScheduler.from_config(
        # pyrefly: ignore [missing-attribute]
        models["pipe"].scheduler.config
    )
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    timesteps = scheduler.timesteps.to(DEVICE)

    mask_builder = SubjectMaskBuilder(
        blur_sigma_start=cfg.mask_blur_sigma_start,
        blur_sigma_end=cfg.mask_blur_sigma_end,
        min_mask_value=cfg.mask_min_value,
    )

    samples = get_samples(cfg.cache_size)

    print("Building latent cache...")
    cached = build_cache(
        samples, vae, text_encoder, tokenizer, DEVICE, total=cfg.cache_size
    )

    empty_emb, _ = encode_prompt("", text_encoder, tokenizer, DEVICE)
    empty_emb = empty_emb.detach().to(dtype=torch.float16)

    del text_encoder
    torch.cuda.empty_cache()
    print("Text encoder released.")

    groups = group_by_bucket(cached, cfg.mini_batch_size)
    if not groups:
        raise RuntimeError(
            f"No aspect bucket holds at least mini_batch_size={cfg.mini_batch_size} "
            f"samples. Raise --cache_size or lower --mini_batch_size."
        )

    usable = sum(len(v) for v in groups.values())
    print(
        f"Cached {len(cached)} samples; {usable} usable across {len(groups)} buckets:"
    )
    for bucket, idxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  {bucket[0]:>3}x{bucket[1]:<3}  {len(idxs)}")

    for segment_name in SEGMENTS:
        unet = train_segment(
            segment_name=segment_name,
            base_unet=base_unet,
            scheduler=scheduler,
            mask_builder=mask_builder,
            cached=cached,
            groups=groups,
            empty_emb=empty_emb,
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
            cached=cached,
            groups=groups,
            cfg=cfg,
        )

        save_lora(unet, cfg.output_dir, segment_name)

        del unet
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train(get_config())
