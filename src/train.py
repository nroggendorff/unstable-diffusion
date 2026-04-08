import random
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file

from .model import load_model, DEVICE, NUM_INFERENCE_STEPS, EARLY_STEPS
from .dataset import get_samples, get_transform, prepare_sample, IMAGE_SIZE
from .encoder import CLIPVisionEncoder, SubjectMaskBuilder
from .encoder.feature_diff import compute_perceptual_discrepancy
from .scheduler import SpatiallyVaryingDDPMScheduler, compute_spatial_noise_scale


STEPS = 200
NOISE_LOSS_WEIGHT = 1.0
UNSPECIFIED_WEIGHT = 0.4
LR = 5e-6
VISION_ENCODER_MODEL = "openai/clip-vit-base-patch32"
FEATURE_LAYERS = [2, 4, 6, 8]
MASK_BLUR_SIGMA_START = 5.0
MASK_BLUR_SIGMA_END = 1.0
MASK_MIN_VALUE = 0.1
SCHEDULER_GAMMA = 2.25
SCHEDULER_K = 5.0


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

    scheduler = SpatiallyVaryingDDPMScheduler.from_config(
        models["pipe"].scheduler.config
    )
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    early_timesteps = scheduler.timesteps[:EARLY_STEPS].to(DEVICE)

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

    for step in (pbar := tqdm(range(STEPS))):
        sample = random.choice(samples)
        image, prompt = prepare_sample(sample, transform, DEVICE)
        if image is None:
            continue

        with torch.no_grad():
            latents = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor

        noise = torch.randn_like(latents)
        t = early_timesteps[torch.randint(0, len(early_timesteps), (1,))]

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
            added_cond_kwargs = {"text_embeds": pooled_emb, "time_ids": time_ids}

            uniform_noisy = scheduler.add_noise(latents, noise, t)
            init_pred = unet_creative(
                uniform_noisy.to(dtype=torch.float16),
                t.to(dtype=torch.float16),
                encoder_hidden_states=text_emb,
                added_cond_kwargs=added_cond_kwargs,
            ).sample.float()
            alphas_cumprod = scheduler.alphas_cumprod.to(DEVICE)
            sqrt_alpha_prod = alphas_cumprod[t.item()] ** 0.5
            sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[t.item()]) ** 0.5
            denoised_latents = (
                uniform_noisy - sqrt_one_minus_alpha_prod * init_pred
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

        blur_sigma = mask_builder.blur_sigma_for_step(step, STEPS)

        with torch.no_grad():
            raw_discrepancy = compute_perceptual_discrepancy(
                pred_features, target_features
            )

        subject_mask = mask_builder.build_mask(raw_discrepancy, blur_sigma)
        subject_mask = F.interpolate(
            subject_mask, size=latents.shape[2:], mode="bilinear", align_corners=False
        )

        t_norm = t.float() / 1000.0
        noise_scale = compute_spatial_noise_scale(
            subject_mask, t_norm, gamma=SCHEDULER_GAMMA, k=SCHEDULER_K
        )
        noisy_latents = scheduler.add_noise(latents, noise, t, noise_scale=noise_scale)

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

    state_dict = get_peft_model_state_dict(unet_creative)

    converted = {
        k.replace("base_model.model.", "unet."): v for k, v in state_dict.items()
    }

    os.makedirs("creative-lora", exist_ok=True)
    save_file(converted, "creative-lora/adapter_model.safetensors")


if __name__ == "__main__":
    train()
