import random
import copy

import torch
from tqdm import tqdm

from peft import get_peft_model

from .model import (
    load_model,
    get_lora_config,
    DEVICE,
    NUM_INFERENCE_STEPS,
    EARLY_SEG,
    MID_SEG,
    LR,
)
from .dataset import get_samples, get_transform
from .encoder import (
    CLIPVisionEncoder,
    compute_perceptual_discrepancy,
    SubjectMaskBuilder,
    CrossAttentionCapture,
)
from .scheduler import SpatiallyVaryingDDPMScheduler, compute_spatial_noise_scale
from .encoding import decode_for_clip
from .loss import (
    compute_loss,
    GROUNDING_WEIGHT,
    SUBJECT_DRIFT_WEIGHT,
    DIVERSITY_WEIGHT,
)
from .cache import build_cache, make_time_ids, blend_alpha, blend_masks
from .io import save_lora

STEPS = 2000
BATCH_SIZE = 8

TRAIN_STEPS = STEPS // BATCH_SIZE

MASK_BLUR_SIGMA_START = 7.0
MASK_BLUR_SIGMA_END = 1.0
MASK_MIN_VALUE = 0.0
SCHEDULER_SUBJECT_POWER = 0.6
SCHEDULER_BG_SCALE = 1.0
SCHEDULER_MIN_SCALE = 0.0

VISION_ENCODER_MODEL = "openai/clip-vit-base-patch32"
FEATURE_LAYERS = [2, 4, 6, 8]

LATE_START = EARLY_SEG + MID_SEG

SEGMENTS = [
    ("early", range(0, EARLY_SEG)),
    ("mid", range(EARLY_SEG, LATE_START)),
    ("late", range(LATE_START, NUM_INFERENCE_STEPS)),
    ("final", range(LATE_START, NUM_INFERENCE_STEPS)),
]


def _make_inputs_require_grad(module, input, output):  # noqa: ARG001
    output.requires_grad_(True)


def train_segment(
    segment_name,
    t_indices,
    base_unet,
    timesteps,
    vae,
    scheduler,
    vision_encoder,
    mask_builder,
    clip_mean,
    clip_std,
    cached,
):
    print(f"\nTraining segment: {segment_name}")

    unet = get_peft_model(copy.deepcopy(base_unet), get_lora_config())
    unet.base_model.model.conv_in.register_forward_hook(_make_inputs_require_grad)
    # pyrefly: ignore [not-callable]
    unet.enable_gradient_checkpointing()
    unet.train()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=LR)
    t_indices_list = list(t_indices)
    alphas = scheduler.alphas_cumprod.to(DEVICE)

    capture = CrossAttentionCapture(base_unet)

    for step in (pbar := tqdm(range(TRAIN_STEPS))):
        indices = list(range(len(cached)))
        random.shuffle(indices)
        items = [cached[i] for i in indices[:BATCH_SIZE]]

        latents = torch.cat([x["latents"] for x in items]).float().to(DEVICE)
        text_emb = torch.cat([x["text_emb"] for x in items]).to(
            DEVICE, dtype=torch.float16
        )
        pooled = torch.cat([x["pooled_text_emb"] for x in items]).to(
            DEVICE, dtype=torch.float16
        )
        token_mask = torch.cat([x["token_attention_mask"] for x in items]).to(DEVICE)
        target_features = [
            torch.cat([x["target_features"][i] for x in items]).float().to(DEVICE)
            for i in range(len(items[0]["target_features"]))
        ]

        # pyrefly: ignore [bad-argument-type]
        time_ids = make_time_ids(BATCH_SIZE, DEVICE)
        added_cond_kwargs = {"text_embeds": pooled, "time_ids": time_ids}

        noise = torch.randn_like(latents)
        t_idx = random.choice(t_indices_list)
        t = timesteps[t_idx].unsqueeze(0).expand(BATCH_SIZE).clone()

        with torch.no_grad():
            uniform_noisy = scheduler.add_noise(latents, noise, t)

            with capture:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    base_pred = base_unet(
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

            pred_features = vision_encoder.extract_features(
                decode_for_clip(vae, denoised_latents, clip_mean, clip_std)
            )
            del denoised_latents

            raw_diff = compute_perceptual_discrepancy(pred_features, target_features)
            del pred_features

            alpha = blend_alpha(step, TRAIN_STEPS)
            mask = blend_masks(attn_mask, raw_diff, latents.shape[2:], alpha=alpha)
            del attn_mask, raw_diff

        blur_sigma = mask_builder.blur_sigma_for_step(step, TRAIN_STEPS)
        mask = mask_builder.build_mask(mask, blur_sigma)

        t_norm = t.float() / 1000.0
        noise_scale = compute_spatial_noise_scale(
            mask,
            t_norm,
            subject_power=SCHEDULER_SUBJECT_POWER,
            bg_scale=SCHEDULER_BG_SCALE,
            min_scale=SCHEDULER_MIN_SCALE,
        )
        noisy_latents = scheduler.add_noise(latents, noise, t, noise_scale=noise_scale)

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
            grounding_weight=GROUNDING_WEIGHT,
            subject_drift_weight=(
                0.0 if segment_name == "final" else SUBJECT_DRIFT_WEIGHT
            ),
            diversity_weight=0.0 if segment_name == "final" else DIVERSITY_WEIGHT,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del noisy_latents, noise_scale, mask, base_pred, uniform_noisy

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            ground=f"{ground.item():.4f}",
            s_drift=f"{subj_drift.item():.4f}",
            div=f"{diversity.item():.4f}",
            blend=f"{alpha:.2f}",
            t=t[0].item(),
        )

        if step % 20 == 0:
            torch.cuda.empty_cache()

    save_lora(unet, f"creative-lora/{segment_name}")

    del unet, optimizer
    torch.cuda.empty_cache()


def train():
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

    scheduler = SpatiallyVaryingDDPMScheduler.from_config(
        # pyrefly: ignore [missing-attribute]
        models["pipe"].scheduler.config
    )
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    timesteps = scheduler.timesteps.to(DEVICE)

    vision_encoder = CLIPVisionEncoder(
        model_name=VISION_ENCODER_MODEL,
        feature_layers=FEATURE_LAYERS,
    ).to(DEVICE)

    mask_builder = SubjectMaskBuilder(
        blur_sigma_start=MASK_BLUR_SIGMA_START,
        blur_sigma_end=MASK_BLUR_SIGMA_END,
        min_mask_value=MASK_MIN_VALUE,
    )

    clip_mean = torch.tensor([0.481, 0.457, 0.408]).view(1, 3, 1, 1).to(DEVICE)
    clip_std = torch.tensor([0.269, 0.261, 0.276]).view(1, 3, 1, 1).to(DEVICE)

    samples = get_samples(TRAIN_STEPS)
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
        vision_encoder,
        clip_mean,
        clip_std,
        DEVICE,
    )

    for segment_name, t_indices in SEGMENTS:
        train_segment(
            segment_name=segment_name,
            t_indices=t_indices,
            base_unet=base_unet,
            timesteps=timesteps,
            vae=vae,
            scheduler=scheduler,
            vision_encoder=vision_encoder,
            mask_builder=mask_builder,
            clip_mean=clip_mean,
            clip_std=clip_std,
            cached=cached,
        )


if __name__ == "__main__":
    train()
