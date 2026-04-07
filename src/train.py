import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .model import load_model, DEVICE
from .dataset import get_samples, get_transform, prepare_sample, IMAGE_SIZE
from .directions import compute_diversity_loss, LAYER_PATTERNS
from .snapshot import SnapshotBuffer


STEPS = 20

NOISE_LOSS_WEIGHT = 1.0
DIVERSITY_LOSS_WEIGHT = 0.3
SNAPSHOT_LOSS_WEIGHT = 0.1
SNAPSHOT_INTERVAL = 50
DIVERSITY_SCHEDULE_START = 0.1
DIVERSITY_SCHEDULE_END = 0.5


def compute_loss(
    unet_creative, noisy_latents, noise, t, text_emb, pooled_emb, time_ids
):
    noisy_latents = noisy_latents.to(dtype=torch.float16)
    t = t.to(dtype=torch.float16)
    added_cond_kwargs = {"text_embeds": pooled_emb, "time_ids": time_ids}

    pred_noise = unet_creative(
        noisy_latents,
        t,
        encoder_hidden_states=text_emb,
        added_cond_kwargs=added_cond_kwargs,
    ).sample

    noise_loss = F.mse_loss(pred_noise, noise)
    return noise_loss


def train():
    models = load_model()
    vae = models["vae"]
    unet_creative = models["unet_creative"]
    text_encoder = models["text_encoder"]
    text_encoder_2 = models["text_encoder_2"]
    tokenizer = models["tokenizer"]
    tokenizer_2 = models["tokenizer_2"]
    optimizer = models["optimizer"]
    scheduler = models["scheduler"]
    early_timesteps = models["early_timesteps"]
    pipe = models["pipe"]

    snapshot_buffer = SnapshotBuffer(unet_creative)

    samples = get_samples(100)
    transform = get_transform()

    for step in (pbar := tqdm(range(STEPS))):
        sample = random.choice(samples)
        image, prompt = prepare_sample(sample, transform, DEVICE)

        if image is None:
            continue

        with torch.no_grad():
            latents = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor

        noise_a = torch.randn_like(latents)
        noise_b = torch.randn_like(latents)
        t = early_timesteps[torch.randint(0, len(early_timesteps), (1,))]

        # pyrefly: ignore [missing-attribute]
        noisy_a = scheduler.add_noise(latents, noise_a, t)
        # pyrefly: ignore [missing-attribute]
        noisy_b = scheduler.add_noise(latents, noise_b, t)

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

        diversity_weight = DIVERSITY_SCHEDULE_START + (
            DIVERSITY_SCHEDULE_END - DIVERSITY_SCHEDULE_START
        ) * (step / STEPS)

        noise_loss = compute_loss(
            unet_creative,
            noisy_a,
            noise_a,
            t.float(),
            text_emb,
            pooled_emb,
            time_ids,
        )
        diversity_loss = compute_diversity_loss(
            unet_creative,
            noisy_a,
            noisy_b,
            t.float(),
            text_emb,
            pooled_emb,
            time_ids,
            LAYER_PATTERNS,
        )
        snapshot_loss = snapshot_buffer.compute_distance_loss(unet_creative)

        loss = (
            NOISE_LOSS_WEIGHT * noise_loss
            + diversity_weight * diversity_loss
            + SNAPSHOT_LOSS_WEIGHT * snapshot_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            noise=f"{noise_loss.item():.4f}",
            div=f"{diversity_loss.item():.4f}",
            snap=f"{snapshot_loss.item():.4f}",
        )

        if step % SNAPSHOT_INTERVAL == 0 and step > 0:
            snapshot_buffer.push(unet_creative)

        if step % 50 == 0:
            torch.cuda.empty_cache()

    pipe.save_pretrained("./creative-early-step")
    unet_creative.save_pretrained("./creative-early-step/lora")


if __name__ == "__main__":
    train()
