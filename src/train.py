import random
import copy
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from peft import get_peft_model
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file

from .model import (
    load_model,
    get_lora_config,
    DEVICE,
    NUM_INFERENCE_STEPS,
    EARLY_SEG,
    MID_SEG,
    LR,
)
from .dataset import get_samples, get_transform, prepare_sample
from .encoder import CLIPVisionEncoder, SubjectMaskBuilder
from .encoder.feature_diff import compute_perceptual_discrepancy
from .scheduler import SpatiallyVaryingDDPMScheduler, compute_spatial_noise_scale

STEPS = 200
NOISE_LOSS_WEIGHT = 1.0
UNSPECIFIED_WEIGHT = 0.4
VISION_ENCODER_MODEL = "openai/clip-vit-base-patch32"
FEATURE_LAYERS = [2, 4, 6, 8]
MASK_BLUR_SIGMA_START = 5.0
MASK_BLUR_SIGMA_END = 1.0
MASK_MIN_VALUE = 0.1
SCHEDULER_GAMMA = 2.25
SCHEDULER_K = 5.0

SEGMENTS = [
    ("early", range(0, EARLY_SEG)),
    ("mid", range(EARLY_SEG, EARLY_SEG + MID_SEG)),
    ("late", range(EARLY_SEG + MID_SEG, NUM_INFERENCE_STEPS)),
]


def compute_loss(
    unet_creative,
    noisy_latents,
    noise,
    t,
    text_emb,
    subject_mask,
):
    noisy_latents = noisy_latents.to(dtype=torch.float16)
    t = t.to(dtype=torch.float16)

    pred_noise = unet_creative(
        noisy_latents,
        t,
        encoder_hidden_states=text_emb,
    ).sample

    pred_noise_f = pred_noise.float()
    noise_f = noise.float()
    mask = subject_mask.to(pred_noise_f.device)

    per_pixel_loss = (pred_noise_f - noise_f).pow(2)

    specified_loss = (per_pixel_loss * mask).mean()
    unspecified_loss = (per_pixel_loss * (1.0 - mask)).mean()

    loss = NOISE_LOSS_WEIGHT * specified_loss - UNSPECIFIED_WEIGHT * unspecified_loss

    return loss, specified_loss.detach(), unspecified_loss.detach()


def save_lora(model, path):
    state_dict = get_peft_model_state_dict(model)

    converted = {}
    for k, v in state_dict.items():
        k = k.replace("base_model.model.", "unet.")
        k = k.replace(".early", str())
        k = k.replace(".mid", str())
        k = k.replace(".late", str())
        converted[k] = v

    os.makedirs(path, exist_ok=True)
    save_file(converted, os.path.join(path, "pytorch_lora_weights.safetensors"))


def train_segment(
    segment_name,
    t_indices,
    base_unet,
    timesteps,
    vae,
    text_encoder,
    tokenizer,
    scheduler,
    vision_encoder,
    mask_builder,
    clip_mean,
    clip_std,
    samples,
    transform,
):
    print(f"\nTraining segment: {segment_name}")

    unet = get_peft_model(copy.deepcopy(base_unet), get_lora_config()).train()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LR)
    t_indices_list = list(t_indices)

    for step in (pbar := tqdm(range(STEPS))):
        sample = random.choice(samples)
        image, prompt = prepare_sample(sample, transform, DEVICE)
        if image is None:
            continue

        with torch.no_grad():
            latents = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor

        noise = torch.randn_like(latents)
        t_idx = random.choice(t_indices_list)
        t = timesteps[t_idx].unsqueeze(0)

        with torch.no_grad():
            target_image_raw = vae.decode(
                (latents / vae.config.scaling_factor).to(dtype=torch.float16)
            ).sample
            target_image_for_encoder = F.interpolate(
                target_image_raw.to(torch.float32),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
            target_normalized = (target_image_for_encoder - clip_mean) / clip_std
            target_features = vision_encoder.extract_features(target_normalized)

        with torch.no_grad():
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).to(DEVICE)
            text_emb = text_encoder(**inputs).last_hidden_state

            uniform_noisy = scheduler.add_noise(latents, noise, t)
            init_pred = unet(
                uniform_noisy.to(dtype=torch.float16),
                t.to(dtype=torch.float16),
                encoder_hidden_states=text_emb,
            ).sample.float()

            alphas_cumprod = scheduler.alphas_cumprod.to(DEVICE)
            sqrt_alpha_prod = alphas_cumprod[t.item()] ** 0.5
            sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[t.item()]) ** 0.5
            denoised_latents = (
                uniform_noisy.float() - sqrt_one_minus_alpha_prod * init_pred
            ) / sqrt_alpha_prod
            denoised_image_raw = vae.decode(
                (denoised_latents / vae.config.scaling_factor).to(dtype=torch.float16)
            ).sample
            denoised_image_for_encoder = F.interpolate(
                denoised_image_raw.to(torch.float32),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
            denoised_normalized = (denoised_image_for_encoder - clip_mean) / clip_std
            pred_features = vision_encoder.extract_features(denoised_normalized)

            raw_discrepancy = compute_perceptual_discrepancy(
                pred_features, target_features
            )

        blur_sigma = mask_builder.blur_sigma_for_step(step, STEPS)
        subject_mask = mask_builder.build_mask(raw_discrepancy, blur_sigma)
        subject_mask = F.interpolate(
            subject_mask, size=latents.shape[2:], mode="bilinear", align_corners=False
        )

        t_norm = t.float() / 1000.0
        noise_scale = compute_spatial_noise_scale(
            subject_mask, t_norm, gamma=SCHEDULER_GAMMA, k=SCHEDULER_K
        )
        noisy_latents = scheduler.add_noise(latents, noise, t, noise_scale=noise_scale)

        loss, spec, unspec = compute_loss(
            unet,
            noisy_latents,
            noise,
            t.float(),
            text_emb,
            subject_mask,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            spec=f"{spec.item():.4f}",
            unspec=f"{unspec.item():.4f}",
            t=t.item(),
        )

        if step % 50 == 0:
            torch.cuda.empty_cache()

    save_lora(unet, f"creative-lora/{segment_name}")

    del unet, optimizer
    torch.cuda.empty_cache()


def train():
    models = load_model()
    vae = models["vae"]
    text_encoder = models["text_encoder"]
    tokenizer = models["tokenizer"]

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

    samples = get_samples(100)
    transform = get_transform()

    # pyrefly: ignore [missing-attribute]
    base_unet = models["pipe"].unet

    for segment_name, t_indices in SEGMENTS:
        train_segment(
            segment_name=segment_name,
            t_indices=t_indices,
            base_unet=base_unet,
            timesteps=timesteps,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            vision_encoder=vision_encoder,
            mask_builder=mask_builder,
            clip_mean=clip_mean,
            clip_std=clip_std,
            samples=samples,
            transform=transform,
        )


if __name__ == "__main__":
    train()
