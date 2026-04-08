import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .model import load_model, DEVICE
from .dataset import get_samples, get_transform, prepare_sample, IMAGE_SIZE


STEPS = 200
NOISE_LOSS_WEIGHT = 1.0
UNSPECIFIED_WEIGHT = 0.4
LR = 5e-6


def get_attention_mask(unet, noisy_latents, t, text_emb, pooled_emb, time_ids):
    attention_maps = []
    hooks = []

    def make_attn_hook():
        def hook(module, input, output):  # noqa: ARG001
            if hasattr(module, "heads"):
                out = (
                    output[1] if isinstance(output, tuple) and len(output) > 1 else None
                )
                if out is not None and out.ndim == 4:
                    attention_maps.append(out.detach().float().mean(dim=1))

        return hook

    for name, module in unet.named_modules():
        if "attn2" in name and name.endswith("attn2"):
            hooks.append(module.register_forward_hook(make_attn_hook()))

    added_cond_kwargs = {"text_embeds": pooled_emb, "time_ids": time_ids}
    with torch.no_grad():
        unet(
            noisy_latents.to(dtype=torch.float16),
            t.to(dtype=torch.float16),
            encoder_hidden_states=text_emb,
            added_cond_kwargs=added_cond_kwargs,
        )

    for hook in hooks:
        hook.remove()

    if not attention_maps:
        h = noisy_latents.shape[2]
        w = noisy_latents.shape[3]
        return torch.ones(1, 1, h, w, device=noisy_latents.device)

    maps = []
    target_h = noisy_latents.shape[2]
    target_w = noisy_latents.shape[3]
    for m in attention_maps:
        spatial = m.mean(dim=-1)
        side = int(spatial.shape[-1] ** 0.5)
        if side * side == spatial.shape[-1]:
            spatial = spatial.reshape(spatial.shape[0], side, side).unsqueeze(1)
            spatial = F.interpolate(
                spatial.float(),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            maps.append(spatial)

    if not maps:
        return torch.ones(1, 1, target_h, target_w, device=noisy_latents.device)

    mask = torch.stack(maps).mean(dim=0)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
    return mask


def compute_loss(
    unet_creative,
    noisy_latents,
    noise,
    t,
    text_emb,
    pooled_emb,
    time_ids,
    attention_mask,
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

    pred_noise_f = pred_noise.float()
    noise_f = noise.float()
    mask = attention_mask.to(pred_noise_f.device)

    per_pixel_loss = (pred_noise_f - noise_f).pow(2)

    specified_loss = (per_pixel_loss * mask).mean()
    unspecified_loss = (per_pixel_loss * (1.0 - mask)).mean()

    loss = NOISE_LOSS_WEIGHT * specified_loss - UNSPECIFIED_WEIGHT * unspecified_loss

    return loss, specified_loss.detach(), unspecified_loss.detach()


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

    samples = get_samples(100)
    transform = get_transform()

    for step in (pbar := tqdm(range(STEPS))):
        sample = random.choice(samples)
        image, prompt = prepare_sample(sample, transform, DEVICE)
        if image is None:
            continue

        with torch.no_grad():
            latents = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor

        noise = torch.randn_like(latents)
        t = early_timesteps[torch.randint(0, len(early_timesteps), (1,))]
        # pyrefly: ignore [missing-attribute]
        noisy_latents = scheduler.add_noise(latents, noise, t)

        with torch.no_grad():
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
            text_emb_1 = text_encoder(**inputs_1, output_hidden_states=True)
            text_emb_2 = text_encoder_2(**inputs_2, output_hidden_states=True)
            text_emb = torch.cat(
                [text_emb_1.hidden_states[-2], text_emb_2.hidden_states[-2]], dim=-1
            )
            pooled_emb = text_emb_2.text_embeds
            time_ids = torch.tensor(
                [[IMAGE_SIZE, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE]], device=DEVICE
            )

        attention_mask = get_attention_mask(
            unet_creative, noisy_latents, t.float(), text_emb, pooled_emb, time_ids
        )

        loss, spec, unspec = compute_loss(
            unet_creative,
            noisy_latents,
            noise,
            t.float(),
            text_emb,
            pooled_emb,
            time_ids,
            attention_mask,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            spec=f"{spec.item():.4f}",
            unspec=f"{unspec.item():.4f}",
        )

        if step % 50 == 0:
            torch.cuda.empty_cache()

    unet_creative.save_pretrained("./creative-lora")


if __name__ == "__main__":
    train()
