import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .model import load_model, DEVICE
from .dataset import get_samples, get_transform, prepare_sample, IMAGE_SIZE


STEPS = 200


def compute_subject_mask(latents):
    down = F.interpolate(
        latents, scale_factor=0.25, mode="bilinear", align_corners=False
    )
    up = F.interpolate(
        down, size=latents.shape[-2:], mode="bilinear", align_corners=False
    )
    importance = (latents - up).abs()
    importance = importance / (importance.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)
    return importance


def apply_structured_noise(noisy_latents, latents, t, t_max):
    importance = latents.abs().mean(dim=1, keepdim=True)
    background_mask = 1.0 - importance / (
        importance.amax(dim=(1, 2, 3), keepdim=True) + 1e-6
    )

    subject_mask = compute_subject_mask(latents)
    combined_mask = (background_mask + (1.0 - subject_mask)) * 0.5

    strength = t.float() / t_max
    noisy_latents = noisy_latents + strength * 0.1 * combined_mask * torch.randn_like(
        noisy_latents
    )
    return noisy_latents


def compute_loss(
    unet_creative,
    unet_base,
    scheduler,
    noisy_latents,
    noise,
    t,
    text_emb,
    pooled_emb,
    time_ids,
):
    noisy_latents = noisy_latents.to(dtype=torch.float16)
    t = t.to(dtype=torch.float16)
    added_cond_kwargs = {
        "text_embeds": pooled_emb,
        "time_ids": time_ids,
    }
    pred_noise = unet_creative(
        noisy_latents,
        t,
        encoder_hidden_states=text_emb,
        added_cond_kwargs=added_cond_kwargs,
    ).sample
    t_int = t.long()
    x_prev_pred = scheduler.step(pred_noise, t_int.item(), noisy_latents).prev_sample

    with torch.no_grad():
        base_noise = unet_base(
            noisy_latents,
            t,
            encoder_hidden_states=text_emb,
            added_cond_kwargs=added_cond_kwargs,
        ).sample
        x_prev_target = scheduler.step(
            base_noise, t_int.item(), noisy_latents
        ).prev_sample

    noise_loss = F.mse_loss(pred_noise, noise)
    step_loss = F.mse_loss(x_prev_pred, x_prev_target)
    divergence = (pred_noise - base_noise).abs().mean()

    loss = noise_loss + 0.5 * step_loss - 0.05 * divergence

    return loss, pred_noise, base_noise


def train():
    models = load_model()
    vae = models["vae"]
    unet_base = models["unet_base"]
    unet_creative = models["unet_creative"]
    text_encoder = models["text_encoder"]
    text_encoder_2 = models["text_encoder_2"]
    tokenizer = models["tokenizer"]
    tokenizer_2 = models["tokenizer_2"]
    optimizer = models["optimizer"]
    scheduler = models["scheduler"]
    early_timesteps = models["early_timesteps"]
    pipe = models["pipe"]

    t_max = scheduler.timesteps[0].float()

    samples = get_samples(100)
    transform = get_transform()

    for step in tqdm(range(STEPS)):
        sample = random.choice(samples)
        image, prompt = prepare_sample(sample, transform, DEVICE)

        if image is None:
            continue

        with torch.no_grad():
            latents = vae.encode(image).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        noise = torch.randn_like(latents)

        t = early_timesteps[torch.randint(0, len(early_timesteps), (1,))]

        noisy_latents = scheduler.add_noise(latents, noise, t)

        with torch.no_grad():
            noisy_latents = apply_structured_noise(noisy_latents, latents, t, t_max)

        inputs_1 = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).to(DEVICE)
        inputs_2 = tokenizer_2(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
        ).to(DEVICE)
        with torch.no_grad():
            text_emb_1 = text_encoder(**inputs_1, output_hidden_states=True)
            text_emb_2 = text_encoder_2(**inputs_2, output_hidden_states=True)
            text_emb = torch.cat(
                [text_emb_1.hidden_states[-2], text_emb_2.hidden_states[-2]], dim=-1
            )
            pooled_emb = text_emb_2.text_embeds
            time_ids = torch.tensor(
                [[IMAGE_SIZE, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE]], device=DEVICE
            )

        loss, pred, base_pred = compute_loss(
            unet_creative,
            unet_base,
            scheduler,
            noisy_latents,
            noise,
            t.float(),
            text_emb,
            pooled_emb,
            time_ids,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            diff = (pred - base_pred).abs().mean().item()
            print(f"step {step} loss {loss.item():.4f} diff {diff:.4f}")

        if step % 50 == 0:
            torch.cuda.empty_cache()

    pipe.save_pretrained("./creative-early-step")
    unet_creative.save_pretrained("./creative-early-step/lora")


if __name__ == "__main__":
    train()
