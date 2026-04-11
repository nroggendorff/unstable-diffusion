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
GROUNDING_WEIGHT = 0.5
DRIFT_WEIGHT = 0.3
DIVERSITY_WEIGHT = 0.1
BASE_DRIFT_TARGET_SIM = 0.1

TRAIN_STEPS = STEPS // BATCH_SIZE

MASK_BLUR_SIGMA_START = 5.0
MASK_BLUR_SIGMA_END = 1.0
MASK_MIN_VALUE = 0.05
SCHEDULER_SUBJECT_POWER = 0.6
SCHEDULER_BG_SCALE = 0.9
SCHEDULER_MIN_SCALE = 0.0

VISION_ENCODER_MODEL = "openai/clip-vit-base-patch32"
FEATURE_LAYERS = [2, 4, 6, 8]
MASK_BLEND_ALPHA_MAX = 0.2

LATE_START = EARLY_SEG + MID_SEG

SEGMENTS = [
    ("early", range(0, EARLY_SEG)),
    ("mid", range(EARLY_SEG, LATE_START)),
    ("late", range(LATE_START, NUM_INFERENCE_STEPS)),
    ("final", range(LATE_START, NUM_INFERENCE_STEPS)),
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


def compute_loss(
    unet,
    noisy_latents,
    t,
    text_emb,
    mask,
    noise_scale,
    alphas_cumprod,
    t_normalized,
    clean_latents,
    base_pred,
    uniform_noisy,
    grounding_weight=GROUNDING_WEIGHT,
    drift_weight=DRIFT_WEIGHT,
    diversity_weight=DIVERSITY_WEIGHT,
):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        pred = unet(noisy_latents, t, encoder_hidden_states=text_emb).sample

    pred_f = pred.float()
    clean_f = clean_latents.float().to(pred_f.device)
    base_f = base_pred.float().to(pred_f.device)
    mask_f = mask.float().to(pred_f.device).clamp(0.0, 1.0)

    a = alphas_cumprod[t.long()].view(-1, 1, 1, 1).sqrt()
    b = (1 - alphas_cumprod[t.long()]).view(-1, 1, 1, 1).sqrt()
    ns = noise_scale.float().to(pred_f.device) if noise_scale is not None else 1.0

    pred_x0 = (noisy_latents.float() - b * ns * pred_f) / a
    base_x0 = (uniform_noisy.float().to(pred_f.device) - b * base_f) / a
    del base_f

    t_scalar = float(t_normalized.mean().item())
    gated = 0.15 <= t_scalar <= 0.85

    if gated and grounding_weight > 0:
        lora_delta = ((pred_x0 - base_x0) * mask_f).flatten(1)
        target_delta = ((clean_f - base_x0) * mask_f).flatten(1)
        grounding_loss = -F.cosine_similarity(
            lora_delta, target_delta, dim=1, eps=1e-6
        ).mean()
        del lora_delta, target_delta
    else:
        grounding_loss = pred_f.new_tensor(0.0)

    del clean_f

    pred_subj = (pred_x0 * mask_f).flatten(1)
    if gated and drift_weight > 0:
        base_subj = (base_x0 * mask_f).flatten(1)
        subj_sim = F.cosine_similarity(pred_subj, base_subj, dim=1, eps=1e-6)
        drift_loss = F.relu(subj_sim - BASE_DRIFT_TARGET_SIM).mean()
        del base_subj
    else:
        drift_loss = pred_f.new_tensor(0.0)

    del base_x0, mask_f

    if pred_f.shape[0] > 1 and diversity_weight > 0:
        pred_subj_norm = F.normalize(pred_subj, dim=1)
        sim_matrix = pred_subj_norm @ pred_subj_norm.T
        n = pred_subj_norm.shape[0]
        pair_mask = torch.triu(
            torch.ones(n, n, dtype=torch.bool, device=pred_subj_norm.device),
            diagonal=1,
        )
        batch_diversity_loss = sim_matrix[pair_mask].mean()
    else:
        batch_diversity_loss = pred_f.new_tensor(0.0)

    del pred_subj

    loss = (
        grounding_weight * grounding_loss
        + drift_weight * drift_loss
        + diversity_weight * batch_diversity_loss
    )
    return (
        loss,
        grounding_loss.detach(),
        drift_loss.detach(),
        batch_diversity_loss.detach(),
    )


def save_lora(model, path):
    state_dict = get_peft_model_state_dict(model)

    converted = {}
    for k, v in state_dict.items():
        k = k.replace("base_model.model.", "unet.")
        k = k.replace(".early", str())
        k = k.replace(".mid", str())
        k = k.replace(".late", str())
        k = k.replace(".final", str())
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
                f.cpu().half()
                for f in vision_encoder.extract_features(
                    decode_for_clip(vae, latents, clip_mean, clip_std)
                )
            ]

        cached.append(
            {
                "latents": latents.cpu().half(),
                "text_emb": text_emb.cpu().half(),
                "token_attention_mask": inputs.attention_mask.cpu(),
                "target_features": target_features,
            }
        )

    return cached


def _make_inputs_require_grad(module, input, output):  # noqa: ARG001
    output.requires_grad_(True)


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

    unet = get_peft_model(copy.deepcopy(base_unet), get_lora_config())
    unet.base_model.model.conv_in.register_forward_hook(_make_inputs_require_grad)
    # pyrefly: ignore [not-callable]
    unet.enable_gradient_checkpointing()
    unet.train()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=LR)
    t_indices_list = list(t_indices)
    alphas = scheduler.alphas_cumprod.to(DEVICE)

    capture = CrossAttentionCapture(base_unet)

    for step in (pbar := tqdm(range(TRAIN_STEPS))):
        indices = list(range(len(cached)))
        random.shuffle(indices)
        items = [cached[i] for i in indices[:BATCH_SIZE]]

        latents = torch.cat([x["latents"] for x in items]).float().to(DEVICE)
        text_emb = torch.cat([x["text_emb"] for x in items]).to(
            DEVICE, dtype=torch.float16
        )
        token_mask = torch.cat([x["token_attention_mask"] for x in items]).to(DEVICE)
        target_features = [
            torch.cat([x["target_features"][i] for x in items]).float().to(DEVICE)
            for i in range(len(items[0]["target_features"]))
        ]

        noise = torch.randn_like(latents)
        t_idx = random.choice(t_indices_list)
        t = timesteps[t_idx].unsqueeze(0).expand(BATCH_SIZE).clone()

        with torch.no_grad():
            uniform_noisy = scheduler.add_noise(latents, noise, t)

            with capture:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    base_pred = base_unet(
                        uniform_noisy.to(dtype=torch.float16),
                        t,
                        encoder_hidden_states=text_emb,
                    ).sample.float()

                # pyrefly: ignore [bad-argument-type]
                attn_mask = capture.build_mask(token_mask, latents.shape[2:])

            a = alphas[t.long()].view(-1, 1, 1, 1) ** 0.5
            b = (1 - alphas[t.long()]).view(-1, 1, 1, 1) ** 0.5
            denoised_latents = (uniform_noisy.float() - b * base_pred) / a

            pred_features = vision_encoder.extract_features(
                decode_for_clip(vae, denoised_latents, clip_mean, clip_std)
            )
            del denoised_latents

            raw_diff = compute_perceptual_discrepancy(pred_features, target_features)
            del pred_features

            alpha = _blend_alpha(step, TRAIN_STEPS)
            mask = _blend_masks(attn_mask, raw_diff, latents.shape[2:], alpha=alpha)
            del attn_mask, raw_diff

        blur_sigma = mask_builder.blur_sigma_for_step(step, TRAIN_STEPS)
        mask = mask_builder.build_mask(mask, blur_sigma)

        t_norm = t.float() / 1000.0
        noise_scale = compute_spatial_noise_scale(
            mask,
            t_norm,
            subject_power=SCHEDULER_SUBJECT_POWER,
            bg_scale=SCHEDULER_BG_SCALE,
            min_scale=SCHEDULER_MIN_SCALE,
        )
        noisy_latents = scheduler.add_noise(latents, noise, t, noise_scale=noise_scale)

        loss, ground, drift, diversity = compute_loss(
            unet,
            noisy_latents,
            t,
            text_emb,
            mask,
            noise_scale,
            alphas_cumprod=alphas,
            t_normalized=t_norm,
            clean_latents=latents,
            base_pred=base_pred,
            uniform_noisy=uniform_noisy,
            grounding_weight=GROUNDING_WEIGHT,
            drift_weight=0.0 if segment_name == "final" else DRIFT_WEIGHT,
            diversity_weight=0.0 if segment_name == "final" else DIVERSITY_WEIGHT,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del noisy_latents, noise_scale, mask, base_pred, uniform_noisy

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            ground=f"{ground.item():.4f}",
            drift=f"{drift.item():.4f}",
            div=f"{diversity.item():.4f}",
            blend=f"{alpha:.2f}",
            t=t[0].item(),
        )

        if step % 20 == 0:
            torch.cuda.empty_cache()

    save_lora(unet, f"creative-lora/{segment_name}")

    del unet, optimizer
    torch.cuda.empty_cache()


def train():
    models = load_model()
    vae = models["vae"]
    text_encoder = models["text_encoder"]
    tokenizer = models["tokenizer"]

    # pyrefly: ignore [missing-attribute]
    base_unet = models["pipe"].unet.eval()
    for param in base_unet.parameters():
        param.requires_grad_(False)

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

    samples = get_samples(TRAIN_STEPS)
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
