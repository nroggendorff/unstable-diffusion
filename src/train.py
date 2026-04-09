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

STEPS = 20000
BATCH_SIZE = 6
NOISE_LOSS_WEIGHT = 1.0
UNSPECIFIED_WEIGHT = 0.4
VISION_ENCODER_MODEL = "openai/clip-vit-base-patch32"
FEATURE_LAYERS = [2, 4, 6, 8]
MASK_BLUR_SIGMA_START = 5.0
MASK_BLUR_SIGMA_END = 1.0
MASK_MIN_VALUE = 0.1
SCHEDULER_GAMMA = 2.25
SCHEDULER_K = 5.0

STEPS = STEPS // BATCH_SIZE

SEGMENTS = [
    ("early", range(0, EARLY_SEG)),
    ("mid", range(EARLY_SEG, EARLY_SEG + MID_SEG)),
    ("late", range(EARLY_SEG + MID_SEG, NUM_INFERENCE_STEPS)),
]


def decode_for_clip(vae, latents, clip_mean, clip_std):
    decoded = vae.decode(
        (latents / vae.config.scaling_factor).to(dtype=torch.float16)
    ).sample
    img = F.interpolate(
        decoded.float(), size=(224, 224), mode="bilinear", align_corners=False
    )
    img = img * 0.5 + 0.5
    return (img - clip_mean) / clip_std


def compute_loss(unet, noisy_latents, noise, t, text_emb, mask):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        pred = unet(
            noisy_latents,
            t,
            encoder_hidden_states=text_emb,
        ).sample

    per_pixel = (pred.float() - noise.float()).pow(2)
    mask_f = mask.float().to(per_pixel.device)

    specified_loss = (per_pixel * mask_f).mean()
    unspecified_loss = (per_pixel * (1.0 - mask_f)).mean()
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

            target_features = vision_encoder.extract_features(
                decode_for_clip(vae, latents, clip_mean, clip_std)
            )

        cached.append(
            {
                "latents": latents.cpu().float(),
                "text_emb": text_emb.cpu(),
                "target_features": [f.cpu() for f in target_features],
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

    for step in (pbar := tqdm(range(STEPS))):
        items = random.choices(cached, k=BATCH_SIZE)
        latents = torch.cat([x["latents"] for x in items]).to(DEVICE)
        text_emb = torch.cat([x["text_emb"] for x in items]).to(DEVICE)
        target_features = [
            torch.cat([x["target_features"][i] for x in items]).to(DEVICE)
            for i in range(len(items[0]["target_features"]))
        ]

        noise = torch.randn_like(latents)
        t_idx = random.choice(t_indices_list)
        t = timesteps[t_idx].unsqueeze(0).expand(BATCH_SIZE).clone()

        with torch.no_grad():
            uniform_noisy = scheduler.add_noise(latents, noise, t)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                init_pred = unet(
                    uniform_noisy.to(dtype=torch.float16),
                    t,
                    encoder_hidden_states=text_emb.to(dtype=torch.float16),
                ).sample.float()

            a = alphas[t.long()].view(-1, 1, 1, 1) ** 0.5
            b = (1 - alphas[t.long()]).view(-1, 1, 1, 1) ** 0.5
            denoised_latents = (uniform_noisy.float() - b * init_pred) / a

            pred_features = vision_encoder.extract_features(
                decode_for_clip(vae, denoised_latents, clip_mean, clip_std)
            )

            raw_diff = compute_perceptual_discrepancy(pred_features, target_features)

        blur_sigma = mask_builder.blur_sigma_for_step(step, STEPS)
        mask = mask_builder.build_mask(raw_diff, blur_sigma)
        mask = F.interpolate(
            mask, size=latents.shape[2:], mode="bilinear", align_corners=False
        )

        t_norm = t.float() / 1000.0
        noise_scale = compute_spatial_noise_scale(
            mask, t_norm, gamma=SCHEDULER_GAMMA, k=SCHEDULER_K
        )
        noisy_latents = scheduler.add_noise(latents, noise, t, noise_scale=noise_scale)

        loss, spec, unspec = compute_loss(unet, noisy_latents, noise, t, text_emb, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            spec=f"{spec.item():.4f}",
            unspec=f"{unspec.item():.4f}",
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
