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
from .encoder import (
    CLIPVisionEncoder,
    compute_perceptual_discrepancy,
    SubjectMaskBuilder,
    CrossAttentionCapture,
)
from .scheduler import SpatiallyVaryingDDPMScheduler, compute_spatial_noise_scale

STEPS = 2000
BATCH_SIZE = 8
NOISE_LOSS_WEIGHT = 1.0
DIVERSITY_WEIGHT = 0.4
RECON_LOSS_WEIGHT = 0.2

STEPS //= BATCH_SIZE

MASK_BLUR_SIGMA_START = 5.0
MASK_BLUR_SIGMA_END = 1.0
MASK_MIN_VALUE = 0.05
SCHEDULER_SUBJECT_POWER = 1.5
SCHEDULER_BG_SCALE = 0.4

VISION_ENCODER_MODEL = "openai/clip-vit-base-patch32"
FEATURE_LAYERS = [2, 4, 6, 8]
MASK_BLEND_ALPHA_MAX = 0.2

SEGMENTS = [
    ("early", range(0, EARLY_SEG)),
    ("mid", range(EARLY_SEG, EARLY_SEG + MID_SEG)),
    ("late", range(EARLY_SEG + MID_SEG, NUM_INFERENCE_STEPS)),
]


def decode_for_clip(
    vae, latents: torch.Tensor, clip_mean: torch.Tensor, clip_std: torch.Tensor
) -> torch.Tensor:
    latents = latents.to(dtype=vae.dtype)

    decoded = vae.decode(latents / vae.config.scaling_factor).sample

    decoded = (decoded.float().clamp(-1, 1) + 1) / 2
    decoded = F.interpolate(
        decoded, size=(224, 224), mode="bilinear", align_corners=False
    )
    return (decoded - clip_mean) / clip_std


def _blend_alpha(step: int, total_steps: int) -> float:
    return MASK_BLEND_ALPHA_MAX * (step / max(total_steps - 1, 1))


def _blend_masks(
    attn_mask: torch.Tensor,
    clip_raw: torch.Tensor,
    spatial_size: tuple,
    alpha: float,
) -> torch.Tensor:
    clip_mask = F.interpolate(
        clip_raw, size=spatial_size, mode="bilinear", align_corners=False
    )
    mn = clip_mask.flatten(1).min(1).values.view(-1, 1, 1, 1)
    mx = clip_mask.flatten(1).max(1).values.view(-1, 1, 1, 1)
    clip_mask = (clip_mask - mn) / (mx - mn + 1e-8)
    return (1.0 - alpha) * attn_mask + alpha * clip_mask


def compute_loss(unet, noisy_latents, noise, t, text_emb, mask, scheduler, noise_scale):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        pred = unet(
            noisy_latents,
            t,
            encoder_hidden_states=text_emb,
        ).sample

    pred_f = pred.float()
    noise_f = noise.float()
    mask_f = mask.float().to(pred_f.device)

    per_pixel = (pred_f - noise_f).pow(2)
    specified_loss = (per_pixel * mask_f).mean()

    bg = pred_f * (1.0 - mask_f)
    diversity_loss = -bg.var(dim=0).mean()

    x0_pred = scheduler.predict_x0(noisy_latents, pred_f, t, noise_scale)
    x0_target = scheduler.get_x0_target(noisy_latents, noise_f, t, noise_scale)

    recon_weight = mask_f
    recon_loss = ((x0_pred - x0_target).pow(2) * recon_weight).mean()

    loss = (
        NOISE_LOSS_WEIGHT * specified_loss
        + DIVERSITY_WEIGHT * diversity_loss
        + RECON_LOSS_WEIGHT * recon_loss
    )
    return loss, specified_loss.detach(), diversity_loss.detach(), recon_loss.detach()


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


def build_cache(
    samples,
    transform,
    vae,
    text_encoder,
    tokenizer,
    vision_encoder,
    clip_mean,
    clip_std,
):
    cached = []
    for sample in tqdm(samples, desc="Caching"):
        image, prompt = prepare_sample(sample, transform, DEVICE)
        if image is None:
            continue

        with torch.no_grad():
            latents = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).to(DEVICE)
            text_emb = text_encoder(**inputs).last_hidden_state

            target_features = [
                f.cpu()
                for f in vision_encoder.extract_features(
                    decode_for_clip(vae, latents, clip_mean, clip_std)
                )
            ]

        cached.append(
            {
                "latents": latents.cpu().float(),
                "text_emb": text_emb.cpu(),
                "token_attention_mask": inputs.attention_mask.cpu(),
                "target_features": target_features,
            }
        )

    return cached


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

    unet = get_peft_model(copy.deepcopy(base_unet), get_lora_config()).train()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LR)
    t_indices_list = list(t_indices)
    alphas = scheduler.alphas_cumprod.to(DEVICE)

    capture = CrossAttentionCapture(unet)

    for step in (pbar := tqdm(range(STEPS))):
        items = random.choices(cached, k=BATCH_SIZE)
        latents = torch.cat([x["latents"] for x in items]).to(DEVICE)
        text_emb = torch.cat([x["text_emb"] for x in items]).to(DEVICE)
        token_mask = torch.cat([x["token_attention_mask"] for x in items]).to(DEVICE)
        target_features = [
            torch.cat([x["target_features"][i] for x in items]).to(DEVICE)
            for i in range(len(items[0]["target_features"]))
        ]

        noise = torch.randn_like(latents)
        t_idx = random.choice(t_indices_list)
        t = timesteps[t_idx].unsqueeze(0).expand(BATCH_SIZE).clone()

        with torch.no_grad():
            uniform_noisy = scheduler.add_noise(latents, noise, t)

            with capture:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    init_pred = unet(
                        uniform_noisy.to(dtype=torch.float16),
                        t,
                        encoder_hidden_states=text_emb.to(dtype=torch.float16),
                    ).sample.float()

            attn_mask = capture.build_mask(token_mask, latents.shape[2:])

            a = alphas[t.long()].view(-1, 1, 1, 1) ** 0.5
            b = (1 - alphas[t.long()]).view(-1, 1, 1, 1) ** 0.5
            denoised_latents = (uniform_noisy.float() - b * init_pred) / a

            pred_features = vision_encoder.extract_features(
                decode_for_clip(vae, denoised_latents, clip_mean, clip_std)
            )
            raw_diff = compute_perceptual_discrepancy(pred_features, target_features)

            alpha = _blend_alpha(step, STEPS)
            mask = _blend_masks(attn_mask, raw_diff, latents.shape[2:], alpha=alpha)

        blur_sigma = mask_builder.blur_sigma_for_step(step, STEPS)
        mask = mask_builder.build_mask(mask, blur_sigma)

        t_norm = t.float() / 1000.0
        noise_scale = compute_spatial_noise_scale(
            mask,
            t_norm,
            subject_power=SCHEDULER_SUBJECT_POWER,
            bg_scale=SCHEDULER_BG_SCALE,
        )
        noisy_latents = scheduler.add_noise(latents, noise, t, noise_scale=noise_scale)

        loss, spec, div, recon = compute_loss(
            unet, noisy_latents, noise, t, text_emb, mask, scheduler, noise_scale
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            spec=f"{spec.item():.4f}",
            div=f"{div.item():.4f}",
            recon=f"{recon.item():.4f}",
            blend=f"{alpha:.2f}",
            t=t[0].item(),
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

    print("Building latent cache...")
    cached = build_cache(
        samples,
        transform,
        vae,
        text_encoder,
        tokenizer,
        vision_encoder,
        clip_mean,
        clip_std,
    )

    # pyrefly: ignore [missing-attribute]
    base_unet = models["pipe"].unet

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
