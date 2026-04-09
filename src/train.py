import random
import copy
import os

import torch
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
from .encoder import SubjectMaskBuilder, CrossAttentionCapture
from .scheduler import SpatiallyVaryingDDPMScheduler, compute_spatial_noise_scale

STEPS = 200
BATCH_SIZE = 4
NOISE_LOSS_WEIGHT = 1.0
DIVERSITY_WEIGHT = 0.2

STEPS //= BATCH_SIZE

MASK_BLUR_SIGMA_START = 5.0
MASK_BLUR_SIGMA_END = 1.0
MASK_MIN_VALUE = 0.1
SCHEDULER_SUBJECT_POWER = 1.5
SCHEDULER_BG_SCALE = 0.4

SEGMENTS = [
    ("early", range(0, EARLY_SEG)),
    ("mid", range(EARLY_SEG, EARLY_SEG + MID_SEG)),
    ("late", range(EARLY_SEG + MID_SEG, NUM_INFERENCE_STEPS)),
]


def compute_loss(unet, noisy_latents, noise, t, text_emb, mask):
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

    loss = NOISE_LOSS_WEIGHT * specified_loss + DIVERSITY_WEIGHT * diversity_loss
    return loss, specified_loss.detach(), diversity_loss.detach()


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


def build_cache(samples, transform, vae, text_encoder, tokenizer):
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

        cached.append(
            {
                "latents": latents.cpu().float(),
                "text_emb": text_emb.cpu(),
                "token_attention_mask": inputs.attention_mask.cpu(),
            }
        )

    return cached


def train_segment(
    segment_name,
    t_indices,
    base_unet,
    timesteps,
    scheduler,
    mask_builder,
):
    print(f"\nTraining segment: {segment_name}")

    unet = get_peft_model(copy.deepcopy(base_unet), get_lora_config()).train()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LR)
    t_indices_list = list(t_indices)

    capture = CrossAttentionCapture(unet)

    for step in (pbar := tqdm(range(STEPS))):
        items = random.choices(cached_global, k=BATCH_SIZE)
        latents = torch.cat([x["latents"] for x in items]).to(DEVICE)
        text_emb = torch.cat([x["text_emb"] for x in items]).to(DEVICE)
        token_mask = torch.cat([x["token_attention_mask"] for x in items]).to(DEVICE)

        noise = torch.randn_like(latents)
        t_idx = random.choice(t_indices_list)
        t = timesteps[t_idx].unsqueeze(0).expand(BATCH_SIZE).clone()

        with torch.no_grad():
            uniform_noisy = scheduler.add_noise(latents, noise, t)

            with capture:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    unet(
                        uniform_noisy.to(dtype=torch.float16),
                        t,
                        encoder_hidden_states=text_emb.to(dtype=torch.float16),
                    )

            mask = capture.build_mask(token_mask, latents.shape[2:])

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

        loss, spec, div = compute_loss(unet, noisy_latents, noise, t, text_emb, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            spec=f"{spec.item():.4f}",
            div=f"{div.item():.4f}",
            t=t[0].item(),
        )

        if step % 50 == 0:
            torch.cuda.empty_cache()

    save_lora(unet, f"creative-lora/{segment_name}")

    del unet, optimizer
    torch.cuda.empty_cache()


cached_global = []


def train():
    global cached_global

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

    mask_builder = SubjectMaskBuilder(
        blur_sigma_start=MASK_BLUR_SIGMA_START,
        blur_sigma_end=MASK_BLUR_SIGMA_END,
        min_mask_value=MASK_MIN_VALUE,
    )

    samples = get_samples(100)
    transform = get_transform()

    print("Building latent cache...")
    cached_global = build_cache(samples, transform, vae, text_encoder, tokenizer)

    # pyrefly: ignore [missing-attribute]
    base_unet = models["pipe"].unet

    for segment_name, t_indices in SEGMENTS:
        train_segment(
            segment_name=segment_name,
            t_indices=t_indices,
            base_unet=base_unet,
            timesteps=timesteps,
            scheduler=scheduler,
            mask_builder=mask_builder,
        )


if __name__ == "__main__":
    train()
