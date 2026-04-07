import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .model import load_model, DEVICE
from .dataset import get_samples, get_transform, prepare_sample


STEPS = 200


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

        noisy_latents = noisy_latents + 0.1 * torch.randn_like(noisy_latents)

        inputs = tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)
        with torch.no_grad():
            text_emb = text_encoder(**inputs).last_hidden_state

        pred = unet_creative(noisy_latents, t, encoder_hidden_states=text_emb).sample

        with torch.no_grad():
            base_pred = unet_base(
                noisy_latents, t, encoder_hidden_states=text_emb
            ).sample

        loss = F.mse_loss(pred, noise) + 0.1 * F.mse_loss(pred, base_pred)

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
