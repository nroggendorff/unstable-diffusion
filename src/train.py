import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import load_model, DEVICE
from dataset import get_samples, get_transform, prepare_sample


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
    unet_creative, unet_base, scheduler, noisy_latents, noise, t, text_emb
):
    pred_noise = unet_creative(noisy_latents, t, encoder_hidden_states=text_emb).sample
    x_prev_pred = scheduler.step(pred_noise, t.item(), noisy_latents).prev_sample

    with torch.no_grad():
        base_noise = unet_base(noisy_latents, t, encoder_hidden_states=text_emb).sample
        x_prev_target = scheduler.step(base_noise, t.item(), noisy_latents).prev_sample

    noise_loss = F.mse_loss(pred_noise, noise)
    step_loss = F.mse_loss(x_prev_pred, x_prev_target)
    divergence = (pred_noise - base_noise).abs().mean()

    loss = noise_loss + 0.5 * step_loss + 0.05 * divergence

    return loss, pred_noise, base_noise


def train():
    models = load_model()
    vae = models["vae"]
    unet_base = models["unet_base"]
    unet_creative = models["unet_creative"]
    text_encoder = models["text_encoder"]
    tokenizer = models["tokenizer"]
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

        inputs = tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)
        with torch.no_grad():
            text_emb = text_encoder(**inputs).last_hidden_state

        loss, pred, base_pred = compute_loss(
            unet_creative, unet_base, scheduler, noisy_latents, noise, t, text_emb
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
    unet_creative.save_adapter("./creative-early-step", "default")


if __name__ == "__main__":
    train()
