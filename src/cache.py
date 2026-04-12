import torch
import torch.nn.functional as F
from tqdm import tqdm

from .dataset import prepare_sample
from .encoding import encode_prompt, decode_for_clip


TARGET_SIZE = 1024
_TIME_IDS_BASE = [TARGET_SIZE, TARGET_SIZE, 0, 0, TARGET_SIZE, TARGET_SIZE]

MASK_BLEND_ALPHA_MAX = 0.2


def make_time_ids(batch_size: int, device: torch.device) -> torch.Tensor:
    ids = torch.tensor(_TIME_IDS_BASE, dtype=torch.float32, device=device)
    return ids.unsqueeze(0).expand(batch_size, -1)


def blend_alpha(step: int, total_steps: int) -> float:
    return MASK_BLEND_ALPHA_MAX * (step / max(total_steps - 1, 1))


def blend_masks(
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


def build_cache(
    samples,
    transform,
    vae,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    vision_encoder,
    clip_mean,
    clip_std,
    device,
):
    cached = []
    for sample in tqdm(samples, desc="Caching"):
        image, prompt = prepare_sample(sample, transform, device)
        if image is None:
            continue

        with torch.no_grad():
            latents = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor

            text_emb, pooled, attn_mask = encode_prompt(
                prompt,
                text_encoder,
                text_encoder_2,
                tokenizer,
                tokenizer_2,
                device,
            )

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
                "pooled_text_emb": pooled.cpu().half(),
                "token_attention_mask": attn_mask.cpu(),
                "target_features": target_features,
            }
        )

    return cached
