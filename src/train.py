import argparse
import os
import random
import copy

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
)
from .dataset import get_samples, get_transform
from .encoder import (
    compute_perceptual_discrepancy,
    SubjectMaskBuilder,
    CrossAttentionCapture,
)
from .scheduler import SpatiallyVaryingDDPMScheduler, compute_spatial_noise_scale
from .loss import compute_loss
from .cache import build_cache, make_time_ids, blend_alpha, blend_masks
from .io import save_lora

LATE_START = EARLY_SEG + MID_SEG

SEGMENTS = [
    ("early", range(0, EARLY_SEG)),
    ("mid", range(EARLY_SEG, LATE_START)),
    ("late", range(LATE_START, NUM_INFERENCE_STEPS)),
]


def _make_inputs_require_grad(module, input, output):  # noqa: ARG001
    output.requires_grad_(True)


def train_segment(
    segment_name,
    t_indices,
    base_unet,
    timesteps,
    scheduler,
    mask_builder,
    cached,
    cfg: argparse.Namespace,
):
    print(f"\nTraining segment: {segment_name}")

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
    t_indices_list = list(t_indices)
    alphas = scheduler.alphas_cumprod.to(DEVICE)

    capture = CrossAttentionCapture(unet)

    for step in (pbar := tqdm(range(cfg.train_steps))):
        optimizer.zero_grad(set_to_none=True)

        accum_loss = accum_ground = accum_subj = accum_div = 0.0
        alpha = blend_alpha(step, cfg.train_steps)
        t_idx = random.choice(t_indices_list)
        t_base = timesteps[t_idx]

        for _ in range(cfg.grad_accum_steps):
            indices = list(range(len(cached)))
            random.shuffle(indices)
            items = [cached[i] for i in indices[: cfg.mini_batch_size]]

            latents = torch.cat([x["latents"] for x in items]).float().to(DEVICE)
            text_emb = torch.cat([x["text_emb"] for x in items]).to(
                DEVICE, dtype=torch.float16
            )
            pooled = torch.cat([x["pooled_text_emb"] for x in items]).to(
                DEVICE, dtype=torch.float16
            )
            token_mask = torch.cat([x["token_attention_mask"] for x in items]).to(
                DEVICE
            )

            # pyrefly: ignore [bad-argument-type]
            time_ids = make_time_ids(cfg.mini_batch_size, DEVICE)
            added_cond_kwargs = {"text_embeds": pooled, "time_ids": time_ids}

            noise = torch.randn_like(latents)
            t = t_base.unsqueeze(0).expand(cfg.mini_batch_size).clone()

            with torch.no_grad():
                uniform_noisy = scheduler.add_noise(latents, noise, t)

                with capture, unet.disable_adapter():
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        base_pred = unet(
                            uniform_noisy.to(dtype=torch.float16),
                            t,
                            encoder_hidden_states=text_emb,
                            added_cond_kwargs=added_cond_kwargs,
                        ).sample.float()

                    # pyrefly: ignore [bad-argument-type]
                    attn_mask = capture.build_mask(token_mask, latents.shape[2:])

                a = alphas[t.long()].view(-1, 1, 1, 1) ** 0.5
                b = (1 - alphas[t.long()]).view(-1, 1, 1, 1) ** 0.5
                denoised_latents = (uniform_noisy.float() - b * base_pred) / a

                raw_diff = compute_perceptual_discrepancy([denoised_latents], [latents])
                del denoised_latents

                mask = blend_masks(attn_mask, raw_diff, latents.shape[2:], alpha=alpha)
                del attn_mask, raw_diff

            blur_sigma = mask_builder.blur_sigma_for_step(step, cfg.train_steps)
            mask = mask_builder.build_mask(mask, blur_sigma)

            t_norm = t.float() / 1000.0
            noise_scale = compute_spatial_noise_scale(
                mask,
                t_norm,
                subject_power=cfg.scheduler_subject_power,
                bg_scale=cfg.scheduler_bg_scale,
                min_scale=cfg.scheduler_min_scale,
            )
            noisy_latents = scheduler.add_noise(
                latents, noise, t, noise_scale=noise_scale
            )

            loss, ground, subj_drift, diversity = compute_loss(
                unet,
                noisy_latents,
                t,
                text_emb,
                added_cond_kwargs,
                mask,
                noise_scale,
                alphas_cumprod=alphas,
                t_normalized=t_norm,
                clean_latents=latents,
                base_pred=base_pred,
                uniform_noisy=uniform_noisy,
                grounding_weight=cfg.grounding_weight,
                subject_drift_weight=(
                    0.0 if segment_name == "final" else cfg.subject_drift_weight
                ),
                diversity_weight=(
                    0.0 if segment_name == "final" else cfg.diversity_weight
                ),
            )

            # pyrefly: ignore [missing-attribute]
            scaler.scale(loss / cfg.grad_accum_steps).backward()

            accum_loss += loss.item()
            accum_ground += ground.item()
            accum_subj += subj_drift.item()
            accum_div += diversity.item()

            del noisy_latents, noise_scale, mask, base_pred, uniform_noisy

        scaler.step(optimizer)
        scaler.update()

        pbar.set_postfix(
            loss=f"{accum_loss / cfg.grad_accum_steps:.4f}",
            ground=f"{accum_ground / cfg.grad_accum_steps:.4f}",
            s_drift=f"{accum_subj / cfg.grad_accum_steps:.4f}",
            div=f"{accum_div / cfg.grad_accum_steps:.4f}",
            blend=f"{alpha:.2f}",
            t=t_base.item(),
        )

        if step % 20 == 0:
            torch.cuda.empty_cache()

    save_lora(unet, os.path.join(cfg.output_dir, segment_name))

    del unet, optimizer, scaler
    torch.cuda.empty_cache()


def train(cfg: argparse.Namespace):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(
        f"Config: steps={cfg.steps}, mini_batch={cfg.mini_batch_size}, "
        f"grad_accum={cfg.grad_accum_steps}, train_steps={cfg.train_steps}, "
        f"lr={cfg.lr}, lora_rank={cfg.lora_rank}, lora_alpha={cfg.lora_alpha}, "
        f"output_dir={cfg.output_dir}"
    )

    models = load_model()
    vae = models["vae"]
    text_encoder = models["text_encoder"]
    text_encoder_2 = models["text_encoder_2"]
    tokenizer = models["tokenizer"]
    tokenizer_2 = models["tokenizer_2"]

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

    samples = get_samples(cfg.train_steps)
    transform = get_transform()

    print("Building latent cache...")
    cached = build_cache(
        samples,
        transform,
        vae,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        DEVICE,
    )

    del text_encoder, text_encoder_2
    torch.cuda.empty_cache()
    print("Text encoders released.")

    for segment_name, t_indices in SEGMENTS:
        train_segment(
            segment_name=segment_name,
            t_indices=t_indices,
            base_unet=base_unet,
            timesteps=timesteps,
            scheduler=scheduler,
            mask_builder=mask_builder,
            cached=cached,
            cfg=cfg,
        )


if __name__ == "__main__":
    train(get_config())
