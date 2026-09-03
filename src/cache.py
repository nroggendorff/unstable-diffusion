import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .dataset import CACHE_WORKERS, prepared_samples
from .encoding import encode_prompt, subset_caption
from .encoder.mask_builder import normalize_mask
from .model import X0_CLAMP

MASK_BLEND_ALPHA_MAX = 0.2


def blend_alpha(step: int, total_steps: int) -> float:
    return MASK_BLEND_ALPHA_MAX * (step / max(total_steps - 1, 1))


def blend_masks(
    attn_mask: torch.Tensor,
    latent_diff: torch.Tensor,
    spatial_size: tuple,
    alpha: float | torch.Tensor,
    gain: float = 0.0,
) -> torch.Tensor:
    if isinstance(alpha, float) and alpha <= 0.0:
        return attn_mask

    diff_resized = F.interpolate(
        latent_diff, size=spatial_size, mode="bilinear", align_corners=False
    )
    return (1.0 - alpha) * attn_mask + alpha * normalize_mask(diff_resized, gain)


def stack_added_cond(items: list, device, dtype=torch.float16) -> tuple:
    pooled = torch.cat([x["pooled"] for x in items]).to(device, dtype=dtype)
    time_ids = torch.stack([x["time_ids"] for x in items]).to(device, dtype=dtype)
    return pooled, time_ids


class LatentStats:
    def __init__(self):
        self.sum_sq: torch.Tensor | None = None
        self.over: torch.Tensor | None = None
        self.max_abs: torch.Tensor | None = None
        self.count = 0

    def absorb(self, latents: torch.Tensor) -> None:
        values = latents.detach().float()
        absolute = values.abs()

        sum_sq = values.pow(2).sum().double()
        over = (absolute > X0_CLAMP).sum().double()
        max_abs = absolute.max()

        if self.sum_sq is None or self.over is None or self.max_abs is None:
            self.sum_sq, self.over, self.max_abs = sum_sq, over, max_abs
        else:
            self.sum_sq = self.sum_sq + sum_sq
            self.over = self.over + over
            self.max_abs = torch.maximum(self.max_abs, max_abs)

        self.count += values.numel()

    def report(self) -> None:
        if (
            self.sum_sq is None
            or self.over is None
            or self.max_abs is None
            or not self.count
        ):
            return

        rms = (self.sum_sq / self.count).sqrt().item()
        print(
            f"Latent stats: rms {rms:.3f}, max|x| {self.max_abs.item():.2f}, "
            f"|x| over X0_CLAMP={X0_CLAMP} on "
            f"{self.over.item() / self.count:.4%} of elements"
        )


def build_cache(
    samples,
    vae,
    text_encoders,
    tokenizers,
    device,
    total=None,
    subset_prob=0.0,
    subset_min=0.15,
    seed=0,
    workers=CACHE_WORKERS,
):
    cached = []
    content_cache: dict = {}
    rng = random.Random(seed)
    subset_count = 0
    words_before = 0
    words_after = 0
    stats = LatentStats()

    prepared = prepared_samples(samples, workers)
    for image, prompt, bucket, time_ids in tqdm(prepared, desc="Caching", total=total):
        if image is None or prompt is None:
            continue

        conditioning = prompt
        if subset_prob > 0.0 and rng.random() < subset_prob:
            conditioning = subset_caption(prompt, rng, subset_min)
            subset_count += int(conditioning != prompt)

        words_before += len(prompt.split())
        words_after += len(conditioning.split())

        with torch.no_grad():
            latents = (
                vae.encode(image.to(device)).latent_dist.sample()
                * vae.config.scaling_factor
            )
            text_emb, pooled, content = encode_prompt(
                conditioning, text_encoders, tokenizers, device, content_cache
            )

        stats.absorb(latents)
        cached.append(
            {
                "latents": latents.cpu().half(),
                "text_emb": text_emb.cpu().half(),
                "pooled": pooled.cpu().half(),
                "time_ids": torch.tensor(time_ids, dtype=torch.float32),
                "token_content_mask": content.cpu(),
                "bucket": bucket,
            }
        )

    stats.report()

    if subset_prob > 0.0 and cached:
        print(
            f"Caption subsetting: {subset_count}/{len(cached)} truncated, "
            f"mean words {words_before / len(cached):.1f} -> "
            f"{words_after / len(cached):.1f}"
        )

    return cached


def group_by_bucket(cached: list, min_size: int) -> dict:
    groups: dict = {}
    for index, item in enumerate(cached):
        groups.setdefault(item["bucket"], []).append(index)
    return {b: idxs for b, idxs in groups.items() if len(idxs) >= min_size}


class BucketSampler:
    def __init__(self, cached: list, min_size: int):
        self.cached = cached
        self.groups = group_by_bucket(cached, min_size)
        self.buckets = list(self.groups)
        self.weights = [len(self.groups[b]) for b in self.buckets]
        self._order = {b: list(idxs) for b, idxs in self.groups.items()}
        self._cursor = dict.fromkeys(self.buckets, 0)
        for bucket in self.buckets:
            self._reshuffle(bucket)

    def __bool__(self) -> bool:
        return bool(self.buckets)

    @property
    def usable(self) -> int:
        return sum(self.weights)

    def _reshuffle(self, bucket) -> None:
        random.shuffle(self._order[bucket])
        self._cursor[bucket] = 0

    def draw(self, mini_batch_size: int) -> list:
        bucket = random.choices(self.buckets, weights=self.weights, k=1)[0]
        order = self._order[bucket]

        if self._cursor[bucket] + mini_batch_size > len(order):
            self._reshuffle(bucket)

        start = self._cursor[bucket]
        self._cursor[bucket] = start + mini_batch_size
        return [self.cached[i] for i in order[start : start + mini_batch_size]]
