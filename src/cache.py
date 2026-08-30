import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .dataset import prepare_sample
from .encoding import encode_prompt
from .encoder.mask_builder import percentile_normalize

MASK_BLEND_ALPHA_MAX = 0.2


def blend_alpha(step: int, total_steps: int) -> float:
    return MASK_BLEND_ALPHA_MAX * (step / max(total_steps - 1, 1))


def blend_masks(
    attn_mask: torch.Tensor,
    latent_diff: torch.Tensor,
    spatial_size: tuple,
    alpha: float | torch.Tensor,
) -> torch.Tensor:
    if isinstance(alpha, float) and alpha <= 0.0:
        return attn_mask

    diff_resized = F.interpolate(
        latent_diff, size=spatial_size, mode="bilinear", align_corners=False
    )
    return (1.0 - alpha) * attn_mask + alpha * percentile_normalize(diff_resized)


def build_cache(samples, vae, text_encoder, tokenizer, device):
    cached = []
    content_cache: dict = {}

    for sample in tqdm(samples, desc="Caching"):
        image, prompt, bucket = prepare_sample(sample, device)
        if image is None:
            continue

        with torch.no_grad():
            latents = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
            text_emb, content = encode_prompt(
                prompt, text_encoder, tokenizer, device, content_cache
            )

        cached.append(
            {
                "latents": latents.cpu().half(),
                "text_emb": text_emb.cpu().half(),
                "token_content_mask": content.cpu(),
                "bucket": bucket,
            }
        )

    return cached


def group_by_bucket(cached: list, min_size: int) -> dict:
    groups: dict = {}
    for index, item in enumerate(cached):
        groups.setdefault(item["bucket"], []).append(index)
    return {b: idxs for b, idxs in groups.items() if len(idxs) >= min_size}


def sample_minibatch(cached: list, groups: dict, mini_batch_size: int) -> list:
    buckets = list(groups.keys())
    weights = [len(groups[b]) for b in buckets]
    bucket = random.choices(buckets, weights=weights, k=1)[0]
    return [cached[i] for i in random.sample(groups[bucket], mini_batch_size)]
