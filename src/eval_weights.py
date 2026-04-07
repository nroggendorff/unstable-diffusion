import csv
import random

import lpips
import torch
from tqdm import tqdm

from .model import load_model, DEVICE
from .dataset import get_samples, get_transform, prepare_sample, IMAGE_SIZE
from .snapshot import SnapshotBuffer
from .directions import compute_diversity_loss, LAYER_NAMES


NOISE_LOSS_WEIGHT = 1.0
DIVERSITY_LOSS_WEIGHT = 0.3
SNAPSHOT_LOSS_WEIGHT = 0.1
SNAPSHOT_INTERVAL = 50
DIVERSITY_SCHEDULE_START = 0.1
DIVERSITY_SCHEDULE_END = 0.5
NUM_INFERENCE_STEPS = 30
EARLY_STEPS = 4
LR = 1e-5
STEPS = 25
GRID = [
    (1.0, 0.1, 0.05),
    (1.0, 0.3, 0.1),
    (1.0, 0.5, 0.1),
    (1.0, 0.5, 0.2),
    (1.0, 0.8, 0.1),
]


def encode_text(tokenizer, tokenizer_2, text_encoder, text_encoder_2, prompt):
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
    return text_emb, pooled_emb, time_ids


def generate_images(
    pipe,
    vae,
    unet,
    latents,
    noise_a,
    noise_b,
    scheduler,
    early_timesteps,
    text_emb,
    pooled_emb,
    time_ids,
):
    t = early_timesteps[torch.randint(0, len(early_timesteps), (1,))]
    noisy_a = scheduler.add_noise(latents, noise_a, t)
    noisy_b = scheduler.add_noise(latents, noise_b, t)

    added_cond_kwargs = {"text_embeds": pooled_emb, "time_ids": time_ids}

    with torch.no_grad():
        pred_a = unet(
            noisy_a,
            t,
            encoder_hidden_states=text_emb,
            added_cond_kwargs=added_cond_kwargs,
        ).sample
        pred_b = unet(
            noisy_b,
            t,
            encoder_hidden_states=text_emb,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

    return pred_a, pred_b


def compute_lpips_distance(img1, img2, lpips_model):
    return lpips_model(img1, img2).item()


def run_grid_search():
    lpips_model = lpips.LPIPS(net="alex").to(DEVICE)
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

    samples = get_samples(20)
    transform = get_transform()

    prompts = [
        "a girl in a dress",
        "a dragon flying over mountains",
        "a robot in a city",
        "a magical forest",
    ]

    results = []

    for noise_w, div_w, snap_w in GRID:
        print(f"\nEvaluating: noise={noise_w}, diversity={div_w}, snapshot={snap_w}")

        for _ in range(3):
            torch.cuda.empty_cache()

        snapshot_buffer = SnapshotBuffer(unet_creative)

        step_losses = []
        for step in tqdm(range(STEPS)):
            sample = random.choice(samples)
            image, prompt = prepare_sample(sample, transform, DEVICE)
            if image is None:
                continue

            with torch.no_grad():
                latents = (
                    vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
                )

            noise_a = torch.randn_like(latents)
            noise_b = torch.randn_like(latents)
            t = early_timesteps[torch.randint(0, len(early_timesteps), (1,))]

            noisy_a = scheduler.add_noise(latents, noise_a, t)
            noisy_b = scheduler.add_noise(latents, noise_b, t)

            text_emb, pooled_emb, time_ids = encode_text(
                tokenizer, tokenizer_2, text_encoder, text_encoder_2, prompt
            )

            diversity_weight = DIVERSITY_SCHEDULE_START + (
                DIVERSITY_SCHEDULE_END - DIVERSITY_SCHEDULE_START
            ) * (step / STEPS)

            pred_noise = unet_creative(
                noisy_a,
                t.float(),
                encoder_hidden_states=text_emb,
                added_cond_kwargs={"text_embeds": pooled_emb, "time_ids": time_ids},
            ).sample

            noise_loss = torch.nn.functional.mse_loss(pred_noise, noise_a)
            diversity_loss = compute_diversity_loss(
                unet_creative,
                noisy_a,
                noisy_b,
                t.float(),
                text_emb,
                pooled_emb,
                time_ids,
                LAYER_NAMES,
            )
            snapshot_loss = snapshot_buffer.compute_distance_loss(unet_creative)

            loss = (
                noise_w * noise_loss
                + diversity_weight * div_w * diversity_loss
                + snap_w * snapshot_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_losses.append(loss.item())

            if step % SNAPSHOT_INTERVAL == 0 and step > 0:
                snapshot_buffer.push(unet_creative)

        lpips_distances = []
        for prompt in prompts:
            with torch.no_grad():
                image, _ = prepare_sample(
                    {"image_url": samples[0]["image_url"], "caption": prompt},
                    transform,
                    DEVICE,
                )
                if image is None:
                    continue
                latents = (
                    vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
                )

            text_emb, pooled_emb, time_ids = encode_text(
                tokenizer, tokenizer_2, text_encoder, text_encoder_2, prompt
            )

            gen_images = []
            for seed_i in range(4):
                noise = torch.randn_like(latents)
                t = early_timesteps[0]
                noisy = scheduler.add_noise(latents, noise, t)
                pred = unet_creative(
                    noisy,
                    t.float(),
                    encoder_hidden_states=text_emb,
                    added_cond_kwargs={"text_embeds": pooled_emb, "time_ids": time_ids},
                ).sample
                latents_pred = scheduler.step(pred, t.item(), noisy).prev_sample
                img = vae.decode(latents_pred / vae.config.scaling_factor).sample
                img = (img / 2 + 0.5).clamp(0, 1)
                gen_images.append(img)

            for i in range(len(gen_images)):
                for j in range(i + 1, len(gen_images)):
                    dist = compute_lpips_distance(
                        gen_images[i], gen_images[j], lpips_model
                    )
                    lpips_distances.append(dist)

        mean_lpips = (
            sum(lpips_distances) / len(lpips_distances) if lpips_distances else 0
        )
        mean_loss = sum(step_losses) / len(step_losses)

        results.append(
            {
                "noise_weight": noise_w,
                "diversity_weight": div_w,
                "snapshot_weight": snap_w,
                "mean_loss": mean_loss,
                "mean_lpips": mean_lpips,
            }
        )
        print(f"  Mean loss: {mean_loss:.4f}, Mean LPIPS: {mean_lpips:.4f}")

    with open("weight_search_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "noise_weight",
                "diversity_weight",
                "snapshot_weight",
                "mean_loss",
                "mean_lpips",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nResults written to weight_search_results.csv")
    return results


if __name__ == "__main__":
    run_grid_search()
