import argparse
import ast
import os
import pathlib
import random
import string
import sys
import tempfile
from types import SimpleNamespace

import torch
from PIL import Image

from .cache import BucketSampler, blend_masks, load_cache, save_cache
from .chain import (
    active_adapters,
    normalize_grad,
    rollout_chain,
    sample_grad_window,
    save_chain_lora,
)
from .dataset import (
    build_time_ids,
    has_alpha,
    prepare_sample,
    shuffle_buffer_size,
)
from .encoder.mask_builder import (
    _gaussian_blur,
    normalize_mask,
    percentile_normalize,
    relative_normalize,
)
from .encoder.attn_mask import _CapturingProcessor
from .encoding import (
    CHUNK_LENGTH,
    CHUNK_TOKENS,
    HIDDEN_SPLIT,
    MAX_CHUNKS,
    POOLED_DIM,
    active_chunks,
    encode_prompt,
    pack_chunks,
    rms_scaled_noise,
    sampling_chunk_mask,
    split_clauses,
    subset_caption,
)
from .model import EARLY_SEG, MID_SEG, NUM_INFERENCE_STEPS, SEGMENT_TIMESTEP_RANGES
from .eval import (
    PROMPTS,
    _base_row,
    adherence,
    angular_spread,
    mean_se,
    paired_deltas,
    radius,
    reusable_base,
)
from .inference import make_two_row_grid
from .loss import compute_diffusion_loss, min_snr_weight
from .rl import (
    adapter_exposure,
    clip_by_exposure,
    composed_prompt,
    decode_pixels,
    floor_penalty,
    scaled_bucket,
)
from .sampler import (
    _BLEND_HALF,
    _MAX_RAW_GUIDANCE,
    ALL_SEGMENTS,
    NEGATIVE_PROMPT,
    adapter_weights,
    lied_timestep,
    prior_deviation,
    prior_latents,
)
from .scheduler import (
    LATENT_DC_STD,
    LATENT_MEAN,
    NOISE_LF_STRENGTH,
    PRIOR_DC,
    PRIOR_LEVELS,
    TERMINAL_LEAK,
    SpatiallyVaryingDDPMScheduler,
    compute_spatial_noise_scale,
    lowfreq_field,
    lowfreq_noise,
)
from .scorer import square_views
from .train import (
    apply_conditioning_augmentation,
    cache_path,
    finite_stages,
    first_nonfinite,
    segment_plan,
    TIMESTEP_RANGES,
)

INIT_SIGMA = 11.5203

MEASURED_BLOCK_VARIANCE = {8: 0.5978, 16: 0.5075, 32: 0.3966, 64: 0.2995}
MEASURED_DC_VARIANCE = 0.2554

_failures: list[str] = []


def check(name: str, condition, detail: str = "") -> None:
    passed = bool(condition)
    print(("PASS  " if passed else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not passed:
        _failures.append(name)


def make_scheduler():
    scheduler = SpatiallyVaryingDDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        steps_offset=1,
        timestep_spacing="leading",
    )
    scheduler.set_timesteps(30)
    return scheduler


class _DummyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.seen_added_cond = "unset"

    def forward(self, sample, t, encoder_hidden_states=None, **kwargs):  # noqa: ARG002
        self.seen_added_cond = kwargs.get("added_cond_kwargs", "missing")
        self.seen_mask = kwargs.get("encoder_attention_mask", "missing")
        return SimpleNamespace(sample=self.conv(sample.float()))


class _DummyTokenizer:
    bos_token_id = 49406
    eos_token_id = 49407

    def __init__(self, pad_token_id=49407):
        self.pad_token_id = pad_token_id
        self._ids: dict = {}
        self._words: dict = {}

    def __call__(
        self,
        text,
        add_special_tokens=False,  # noqa: ARG002
        truncation=True,  # noqa: ARG002
        max_length=150,  # noqa: ARG002
    ):
        ids = []
        for word in text.lower().split():
            if word not in self._ids:
                self._ids[word] = len(self._ids) + 1
                self._words[self._ids[word]] = word
            ids.append(self._ids[word])
        return {"input_ids": ids[:max_length]}

    def convert_ids_to_tokens(self, token_id):
        return self._words.get(int(token_id), "")


class _DummyTextEncoder(torch.nn.Module):
    def __init__(self, hidden, projection=None):
        super().__init__()
        self.embedding = torch.nn.Embedding(49408, hidden)
        self.projection = (
            None if projection is None else torch.nn.Linear(hidden, projection)
        )

    def forward(self, input_ids, output_hidden_states=False):  # noqa: ARG002
        self.seen_input_ids = input_ids
        hidden = self.embedding(input_ids)
        output = SimpleNamespace(
            hidden_states=[hidden, hidden, hidden], last_hidden_state=hidden
        )
        if self.projection is not None:
            output.text_embeds = self.projection(hidden.mean(1))
        return output


def _block_variance(x: torch.Tensor, stride: int) -> float:
    if stride >= min(x.shape[-2:]):
        return x.mean(dim=(2, 3)).pow(2).mean().item()
    return torch.nn.functional.avg_pool2d(x, stride).pow(2).mean().item()


def test_lowfreq_noise(scheduler):
    print("\n--- low-frequency noise prior ---")
    latents = torch.zeros(6, 4, 128, 128)
    noise = lowfreq_noise(latents)

    check("shape preserved", noise.shape == latents.shape, str(tuple(noise.shape)))
    check("dtype preserved", noise.dtype == latents.dtype)

    rms = noise.pow(2).flatten(1).mean(1).sqrt()
    check(
        "per-sample RMS is 1 (variance invariant held)",
        torch.allclose(rms, torch.ones_like(rms), atol=1e-4),
        f"rms={[round(v, 5) for v in rms.tolist()[:3]]}",
    )

    terminal = scheduler.alphas_cumprod[-1].item()
    check(
        "TERMINAL_LEAK is the t=999 signal/noise amplitude ratio",
        abs(TERMINAL_LEAK - (terminal / (1 - terminal)) ** 0.5) < 5e-4,
        f"{TERMINAL_LEAK} vs {(terminal / (1 - terminal)) ** 0.5:.5f}",
    )

    torch.manual_seed(0)
    std = torch.tensor(LATENT_DC_STD).view(1, 4, 1, 1)
    field = lowfreq_field((400, 4, 128, 128)) / std
    profile = {s: _block_variance(field, s) for s in (8, 16, 32, 64, 128)}
    targets = {s: v / MEASURED_DC_VARIANCE for s, v in MEASURED_BLOCK_VARIANCE.items()}
    targets[128] = 1.0
    worst = max(abs(profile[s] / targets[s] - 1.0) for s in targets)
    check(
        "the field reproduces the measured latent spectrum, stride 8 to DC",
        worst < 0.12,
        " ".join(f"s{s}:{profile[s]:.2f}/{targets[s]:.2f}" for s in targets),
    )

    for height, width in ((104, 152), (192, 80)):
        odd = lowfreq_field((300, 4, height, width)) / std
        dc = odd.mean(dim=(2, 3)).pow(2).mean().item()
        check(
            f"DC variance holds on a {width}x{height} latent",
            abs(dc - 1.0) < 0.2,
            f"{dc:.3f}",
        )

    bulk = torch.zeros(400, 4, 128, 128)
    torch.manual_seed(1)
    plain = lowfreq_noise(bulk, strength=0.0)
    torch.manual_seed(1)
    prior = lowfreq_noise(bulk)
    check(
        "strength=0 reduces to plain randn",
        abs(plain.mean(dim=(2, 3)).var().item() - 1.0 / (128 * 128)) < 1e-5,
    )

    leak_dc = terminal * MEASURED_DC_VARIANCE
    noise_dc = (1 - terminal) * (
        prior.mean(dim=(2, 3)).pow(2).mean().item()
        - plain.mean(dim=(2, 3)).pow(2).mean().item()
    )
    check(
        "strength=1 puts the added DC noise at parity with the terminal-SNR leak",
        0.7 < noise_dc / leak_dc < 1.4,
        f"added/leak {noise_dc / leak_dc:.2f}x",
    )

    leak_mid = terminal * MEASURED_BLOCK_VARIANCE[8]
    excess_mid = (1 - terminal) * (
        _block_variance(prior, 8) - _block_variance(plain, 8)
    )
    check(
        "mid frequencies stay near the leak too (the old pyramid put 35x)",
        excess_mid / leak_mid < 2.0,
        f"added/leak at stride 8 {excess_mid / leak_mid:.2f}x",
    )
    check(
        "survives latents smaller than every level",
        lowfreq_noise(torch.zeros(2, 4, 8, 6)).shape == (2, 4, 8, 6),
    )
    check(
        "the measured constants cover every SDXL latent channel",
        len(LATENT_MEAN) == len(LATENT_DC_STD) == 4,
    )


def test_min_snr(scheduler):
    print("\n--- min_snr_weight ---")
    t = torch.tensor([1, 100, 364, 628, 958])
    weight = min_snr_weight(scheduler.alphas_cumprod, t, 5.0)

    check(
        "batch mean is 1",
        abs(weight.mean().item() - 1.0) < 1e-5,
        f"mean={weight.mean():.6f}",
    )
    check(
        "high-SNR steps downweighted, low-SNR steps left flat",
        weight[0] < weight[1] < weight[2]
        and abs(weight[2].item() - weight[4].item()) < 1e-6,
        f"t=1:{weight[0]:.4f} t=100:{weight[1]:.4f} "
        f"t=364:{weight[2]:.4f} t=958:{weight[4]:.4f}",
    )

    late = min_snr_weight(scheduler.alphas_cumprod, torch.tensor([1, 364]), 5.0)
    check(
        "late segment 627x SNR span compressed",
        (late[1] / late[0]).item() > 50,
        f"ratio={(late[1] / late[0]).item():.1f}x",
    )


def test_loss_weighting(scheduler):
    print("\n--- loss weighting ---")
    unet = _DummyUNet()
    noisy = torch.randn(4, 4, 16, 16)
    noise = torch.randn(4, 4, 16, 16)
    embeddings = torch.randn(4, 154, 2048)
    added_cond = {"text_embeds": torch.randn(4, 1280), "time_ids": torch.zeros(4, 6)}
    t = torch.full((4,), 300, dtype=torch.long)

    def loss_for(mask_value):
        torch.manual_seed(1)
        return compute_diffusion_loss(
            unet,
            noisy,
            t,
            embeddings,
            noise,
            added_cond_kwargs=added_cond,
            mask=torch.full((4, 1, 16, 16), mask_value),
            bg_weight=0.25,
        ).item()

    sparse, dense = loss_for(0.05), loss_for(0.95)
    check(
        "added_cond_kwargs reaches the UNet",
        unet.seen_added_cond is added_cond,
        str(unet.seen_added_cond if isinstance(unet.seen_added_cond, str) else "dict"),
    )

    chunk_mask = torch.ones(4, 154)
    compute_diffusion_loss(
        unet,
        noisy,
        t,
        embeddings,
        noise,
        added_cond_kwargs=added_cond,
        encoder_attention_mask=chunk_mask,
    )
    check("encoder_attention_mask reaches the UNet", unet.seen_mask is chunk_mask)
    check(
        "loss scale no longer tracks mask coverage",
        abs(sparse - dense) < 1e-5,
        f"sparse={sparse:.6f} dense={dense:.6f}",
    )
    check(
        "unnormalized weighting would have differed 4x",
        abs((0.25 + 0.05) / (0.25 + 0.95) - 0.25) < 1e-9,
    )

    mask = torch.rand(4, 1, 16, 16)
    loss = compute_diffusion_loss(
        unet,
        noisy,
        t,
        embeddings,
        noise,
        mask=mask,
        bg_weight=0.25,
        alphas_cumprod=scheduler.alphas_cumprod,
        snr_gamma=5.0,
    )
    loss.backward()
    grad = unet.conv.weight.grad
    check(
        "loss backward reaches parameters",
        grad is not None and grad.abs().sum() > 0,
    )
    check(
        "snr_gamma=0 path still works",
        compute_diffusion_loss(
            unet, noisy, t, embeddings, noise, mask=mask, snr_gamma=0.0
        ).item()
        > 0,
    )

    scale = compute_spatial_noise_scale(mask, torch.tensor([0.9, 0.5, 0.2, 0.05]))
    scale_rms = scale.pow(2).flatten(1).mean(1).sqrt()
    check(
        "spatial noise scale still has E[scale^2]=1 per sample",
        torch.allclose(scale_rms, torch.ones_like(scale_rms), atol=1e-5),
    )


def test_bucket_sampler():
    print("\n--- BucketSampler ---")
    cached = [{"bucket": (1216, 832) if i < 100 else (832, 1216)} for i in range(140)]
    sampler = BucketSampler(cached, 2)

    check("buckets grouped", set(sampler.groups) == {(1216, 832), (832, 1216)})
    check("usable count", sampler.usable == 140)

    seen: dict[int, int] = {}
    duplicate_in_batch = False
    for _ in range(2000):
        batch = sampler.draw(2)
        ids = [id(item) for item in batch]
        duplicate_in_batch |= len(set(ids)) != len(ids)
        for key in ids:
            seen[key] = seen.get(key, 0) + 1

    check("no duplicates within a mini-batch", not duplicate_in_batch)
    check("every cached sample drawn", len(seen) == 140, f"{len(seen)}/140")

    counts = sorted(seen.values())
    check(
        "coverage near-uniform",
        counts[-1] / counts[0] <= 2.2,
        f"min={counts[0]} max={counts[-1]}",
    )


def test_conditioning_augmentation():
    print("\n--- conditioning augmentation ---")
    embeddings = torch.randn(8, 154, 2048)
    pooled = torch.randn(8, 1280)
    content = torch.ones(8, 154)
    empty = torch.zeros(1, 154, 2048)
    empty_pooled = torch.zeros(1, 1280)

    dropped, dropped_pooled, dropped_content = apply_conditioning_augmentation(
        embeddings,
        pooled,
        content,
        empty,
        empty_pooled,
        argparse.Namespace(
            cond_dropout_prob=1.0, cond_partial_prob=0.0, cond_partial_max=0.6
        ),
    )
    check("full dropout gives the empty embedding", dropped.abs().max() < 1e-6)
    check(
        "full dropout also empties the pooled embedding",
        dropped_pooled.abs().max() < 1e-6,
    )
    check("full dropout zeroes the content mask", dropped_content.abs().max() < 1e-6)

    torch.manual_seed(3)
    partial, partial_pooled, partial_content = apply_conditioning_augmentation(
        embeddings,
        pooled,
        content,
        empty,
        empty_pooled,
        argparse.Namespace(
            cond_dropout_prob=0.0, cond_partial_prob=1.0, cond_partial_max=0.6
        ),
    )
    scales = (partial / embeddings).flatten(1).mean(1)
    check(
        "partial conditioning lerps toward empty within cond_partial_max",
        bool(((scales > 0) & (scales <= 0.6001)).all()),
        f"scales={[round(v, 3) for v in scales.tolist()[:4]]}",
    )
    check(
        "content mask scaled by the same factor",
        torch.allclose(partial_content[:, 0], scales, atol=1e-3),
    )
    check(
        "pooled embedding is lerped by the same factor",
        torch.allclose((partial_pooled / pooled).flatten(1).mean(1), scales, atol=1e-2),
    )

    same, same_pooled, same_content = apply_conditioning_augmentation(
        embeddings,
        pooled,
        content,
        empty,
        empty_pooled,
        argparse.Namespace(
            cond_dropout_prob=0.0, cond_partial_prob=0.0, cond_partial_max=0.6
        ),
    )
    check(
        "both disabled is a no-op",
        torch.equal(same, embeddings)
        and torch.equal(same_pooled, pooled)
        and torch.equal(same_content, content),
    )


def test_predict_x0(scheduler):
    print("\n--- scheduler: predict_x0 inverts add_noise ---")
    grid = scheduler.timesteps
    reference = torch.randn(3, 4, 16, 16)
    noise = torch.randn(3, 4, 16, 16)
    t_end = grid[10].unsqueeze(0).expand(3)

    x_t = scheduler.add_noise(reference, noise, t_end)
    recovered = scheduler.predict_x0(x_t, noise, t_end)

    error_x0 = (recovered - reference).pow(2).mean().item()
    error_xt = (x_t - reference).pow(2).mean().item()
    check(
        "predict_x0 recovers the reference where raw x_t does not",
        error_x0 < 1e-8 < error_xt,
        f"err(x0_pred)={error_x0:.2e} err(x_t)={error_xt:.3f} t={int(t_end[0])}",
    )


def test_sampler_parity():
    print("\n--- sampler parity with the app.py mirror ---")

    def app_weights(step_index, strength):
        boundaries, half = [10, 20], 2
        for i, boundary in enumerate(boundaries):
            distance = step_index - boundary
            if abs(distance) <= half:
                t = (distance + half) / (2 * half)
                weights = [0.0, 0.0, 0.0]
                weights[i] = (1.0 - t) * strength
                weights[i + 1] = t * strength
                return weights
        weights = [0.0, 0.0, 0.0]
        if step_index < boundaries[0]:
            weights[0] = strength
        elif step_index < boundaries[1]:
            weights[1] = strength
        else:
            weights[2] = strength
        return weights

    check(
        "sampler.adapter_weights matches the app.py copy across the schedule",
        all(adapter_weights(i, 0.8) == app_weights(i, 0.8) for i in range(30)),
    )

    def app_prior_latents(shape, generators, device, dtype, init_sigma, lf=1.0):
        def app_field(shape, generator, device):
            batch, channels, height, width = shape
            field = PRIOR_DC * torch.randn(
                (batch, channels, 1, 1), generator=generator, device=device
            )
            field = field.expand(batch, channels, height, width)
            for stride, amplitude in PRIOR_LEVELS:
                low = torch.randn(
                    (batch, channels, -(-height // stride), -(-width // stride)),
                    generator=generator,
                    device=device,
                )
                field = field + amplitude * torch.nn.functional.interpolate(
                    low, size=(height, width), mode="bilinear", align_corners=False
                )
            std = torch.tensor(LATENT_DC_STD[:channels], device=device)
            return field * std.view(1, channels, 1, 1)

        batch, channels, height, width = shape
        sigma = max(init_sigma**2 - 1.0, 0.0) ** 0.5
        deviation = (1.0 + NOISE_LF_STRENGTH * (TERMINAL_LEAK * sigma) ** 2) ** 0.5
        mean = torch.tensor(LATENT_MEAN[:channels], device=device).view(
            1, channels, 1, 1
        )
        samples = []
        for index in range(batch):
            generator = generators[index]
            noise = torch.randn(
                (1, channels, height, width),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            if lf > 0.0:
                field = app_field((1, channels, height, width), generator, device)
                noise = noise + lf * (mean + deviation * field) / init_sigma
            samples.append(noise)
        return torch.cat(samples).to(dtype)

    shape = (2, 4, 64, 48)
    matches = True
    for lf in (0.0, 0.5, 1.0):
        mine = prior_latents(
            shape,
            [torch.Generator().manual_seed(7 + i) for i in range(shape[0])],
            "cpu",
            torch.float32,
            init_sigma=INIT_SIGMA,
            lf=lf,
        )
        theirs = app_prior_latents(
            shape,
            [torch.Generator().manual_seed(7 + i) for i in range(shape[0])],
            "cpu",
            torch.float32,
            init_sigma=INIT_SIGMA,
            lf=lf,
        )
        matches = matches and bool(torch.equal(mine, theirs))
    check("sampler.prior_latents matches the app.py copy across lf", matches)

    plain = prior_latents(
        shape,
        [torch.Generator().manual_seed(3 + i) for i in range(shape[0])],
        "cpu",
        torch.float32,
        init_sigma=INIT_SIGMA,
        lf=0.0,
    )
    reference = torch.cat(
        [
            torch.randn((1, 4, 64, 48), generator=torch.Generator().manual_seed(3 + i))
            for i in range(2)
        ]
    )
    check("lf=0 is exactly the generator's plain randn", torch.equal(plain, reference))

    def original_lie(schedule, timestep, offset, gamma):
        last = len(schedule) - 1
        told = min(
            1.0,
            max(
                0.0,
                (1.0 - (schedule - timestep).abs().argmin().item() / max(last, 1))
                ** gamma
                + offset,
            ),
        )
        return torch.lerp(
            schedule[int((1.0 - told) * last)].float(),
            schedule[min(int((1.0 - told) * last) + 1, last)].float(),
            (1.0 - told) * last % 1.0,
        )

    schedule = make_scheduler().timesteps.float()
    same = all(
        torch.allclose(
            lied_timestep(schedule, index, offset, gamma),
            original_lie(schedule, schedule[index], offset, gamma),
        )
        for offset, gamma in ((0.0, 0.8), (0.1, 1.0), (-0.2, 1.5))
        for index in range(len(schedule))
    )
    check("lied_timestep reproduces the original lie_about_noise", same)


def test_eval_metrics():
    print("\n--- eval metrics ---")
    torch.manual_seed(5)
    clustered = torch.nn.functional.normalize(
        torch.randn(1, 64).repeat(8, 1) + 0.01 * torch.randn(8, 64), dim=-1
    )
    varied = torch.nn.functional.normalize(torch.randn(8, 64), dim=-1)

    check(
        "angular spread separates clustered from varied",
        angular_spread(clustered) < 0.05 < angular_spread(varied),
        f"clustered={angular_spread(clustered):.4f} varied={angular_spread(varied):.4f}",
    )

    basin = torch.nn.functional.normalize(torch.randn(64), dim=-1)
    near = torch.nn.functional.normalize(
        basin.unsqueeze(0) + 0.01 * torch.randn(4, 64), dim=-1
    )
    check(
        "radius is near zero at the basin and large away from it",
        radius(near, basin) < 0.01 < radius(varied, basin),
        f"near={radius(near, basin):.5f} far={radius(varied, basin):.4f}",
    )
    check("adherence is a cosine", -1.0 <= adherence(varied, basin) <= 1.0)
    check(
        "single-sample angular spread is defined", angular_spread(clustered[:1]) == 0.0
    )


def test_paired_deltas():
    print("\n--- eval: paired deltas ---")
    torch.manual_seed(11)
    normalize = torch.nn.functional.normalize
    mean, se = mean_se(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    check(
        "mean_se matches the textbook values",
        abs(mean - 2.5) < 1e-12 and abs(se - 0.6454972) < 1e-6,
        f"mean={mean} se={se:.7f}",
    )

    basin = normalize(torch.randn(64), dim=-1)
    text = normalize(torch.randn(5, 64), dim=-1)
    base = [normalize(torch.randn(6, 64), dim=-1) for _ in range(5)]
    same = paired_deltas(base, base, basin, text)
    check(
        "identical sets give zero deltas and zero displacement",
        all(abs(row["delta"]) < 1e-6 for row in same.values()),
        str({k: round(v["delta"], 8) for k, v in same.items()}),
    )

    toward = [normalize(f + 0.6 * basin, dim=-1) for f in base]
    moved = paired_deltas(toward, base, basin, text)
    check(
        "moving every image toward the basin gives a negative radius delta",
        moved["radius"]["delta"] < 0 and moved["displacement"]["delta"] > 0,
        f"radius={moved['radius']['delta']:+.4f} "
        f"displacement={moved['displacement']['delta']:.4f}",
    )

    nudged = [normalize(f + 0.05 * torch.randn_like(f), dim=-1) for f in base]
    deltas = paired_deltas(nudged, base, basin, text)
    radius_a = 1 - torch.stack(nudged) @ basin
    radius_b = 1 - torch.stack(base) @ basin
    unpaired = (mean_se(radius_a)[1] ** 2 + mean_se(radius_b)[1] ** 2) ** 0.5
    check(
        "pairing shrinks the radius SE when images move little per seed",
        deltas["radius"]["se"] < unpaired / 3,
        f"paired={deltas['radius']['se']:.5f} unpaired={unpaired:.5f}",
    )
    check(
        "radius delta equals the difference of the unpaired means",
        abs(
            deltas["radius"]["delta"]
            - (radius(torch.cat(nudged), basin) - radius(torch.cat(base), basin))
        )
        < 1e-6,
    )


def test_base_reuse():
    print("\n--- eval: reusing a prior base pass ---")
    prompts = ["a girl", "a boy"]
    seeds = [[1000, 1001], [1100, 1101]]
    args = {"guidance": 7.0, "strength": 0.8, "adapter_path": "a", "base_from": ""}
    prior = {"prompts": prompts, "seed_list": seeds, "args": dict(args)}

    candidate = dict(args, strength=1.0, adapter_path="b", base_from="x")
    check(
        "a prior differing only in adapter, strength and base_from is reusable",
        reusable_base(prior, candidate, prompts, seeds) is prior,
    )
    refused = []
    for name, call in (
        (
            "guidance",
            lambda: reusable_base(prior, dict(args, guidance=6.0), prompts, seeds),
        ),
        ("seeds", lambda: reusable_base(prior, args, prompts, [[1, 2], [3, 4]])),
        ("prompts", lambda: reusable_base(prior, args, prompts[:1], seeds[:1])),
    ):
        try:
            call()
        except ValueError:
            refused.append(name)
    check(
        "a prior differing in a base setting, seeds or prompts is refused",
        refused == ["guidance", "seeds", "prompts"],
        str(refused),
    )

    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    top = [Image.new("RGB", (8, 12), (9, 9, 9)) for _ in colours]
    bottom = [Image.new("RGB", (8, 12), c) for c in colours]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "00.png")
        make_two_row_grid(top, bottom).save(path)
        row = _base_row(path, (8, 12), 3)
        missing = _base_row(os.path.join(tmp, "01.png"), (8, 12), 3)
    check(
        "_base_row recovers the bottom row of a saved sheet, tile for tile",
        [im.getpixel((4, 6)) for im in row] == colours and not missing,
        str([im.getpixel((4, 6)) for im in row]),
    )


def test_segment_plan():
    print("\n--- train: segment plan ---")
    full = segment_plan(["early", "mid", "late"], True)
    check(
        "default plan trains all three with cache seeds 0/1/2",
        full == [(0, "early", 0), (1, "mid", 1), (2, "late", 2)],
        str(full),
    )
    check(
        "without refresh every segment shares cache seed 0",
        [seed for _, _, seed in segment_plan(["early", "mid", "late"], False)]
        == [0, 0, 0],
    )
    check(
        "a single segment keeps the seed it has in a full run",
        segment_plan(["late"], True) == [(2, "late", 2)],
    )
    check(
        "segment order follows SEGMENTS, not the flag",
        [name for _, name, _ in segment_plan(["late", "early"], True)]
        == ["early", "late"],
    )
    check(
        "all trains one adapter on cache seed 0",
        segment_plan(["all"], True) == [(0, "all", 0)],
    )
    low, high = TIMESTEP_RANGES["all"]
    check(
        "all covers every segment's timesteps",
        all(low <= a and b <= high for a, b in SEGMENT_TIMESTEP_RANGES.values()),
        str(TIMESTEP_RANGES["all"]),
    )
    for bad in (["middle"], [], ["all", "early"]):
        try:
            segment_plan(bad, True)
            raised = False
        except ValueError:
            raised = True
        check(f"segment plan rejects {bad}", raised)


def test_latent_cache_persistence():
    print("\n--- cache: persistence ---")
    cfg = argparse.Namespace(
        cache_dir="/tmp/cache",
        cache_size=128,
        shuffle_buffer=1024,
        caption_subset_prob=0.5,
        caption_subset_min=0.15,
    )
    base_path = cache_path(cfg, 0)
    check("cache path is stable", base_path == cache_path(cfg, 0), base_path)
    for field, value in (
        ("seed", None),
        ("cache_size", 256),
        ("shuffle_buffer", 512),
        ("caption_subset_prob", 0.0),
        ("caption_subset_min", 0.3),
    ):
        changed = argparse.Namespace(**vars(cfg))
        seed = 1 if field == "seed" else 0
        if value is not None:
            setattr(changed, field, value)
        check(
            f"cache path changes with {field}", cache_path(changed, seed) != base_path
        )

    cached: list[dict] = [
        {
            "latents": torch.randn(1, 4, 8, 6).half(),
            "text_emb": torch.randn(1, 154, 16).half(),
            "pooled": torch.randn(1, 8).half(),
            "time_ids": torch.tensor([512.0, 384.0, 0.0, 16.0, 1216.0, 832.0]),
            "token_content_mask": torch.rand(1, 154) > 0.5,
            "token_valid": torch.ones(1, 154, dtype=torch.bool),
            "bucket": (832, 1216),
        }
        for _ in range(3)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nested", "latents.pt")
        save_cache(cached, path)
        loaded = load_cache(path)
        check("no partial file is left behind", not os.path.exists(path + ".partial"))
    check(
        "a saved cache round-trips exactly",
        len(loaded) == len(cached)
        and all(
            torch.equal(a[k], b[k])
            for a, b in zip(cached, loaded)
            for k in a
            if k != "bucket"
        ),
    )
    check(
        "buckets come back as hashable tuples",
        all(isinstance(item["bucket"], tuple) for item in loaded)
        and BucketSampler(loaded, 1).buckets == [(832, 1216)],
    )


class _DummyVAE:
    dtype = torch.float32
    config = SimpleNamespace(scaling_factor=1.0)

    def decode(self, latents):
        return SimpleNamespace(sample=latents.expand(-1, 3, -1, -1))


def test_straight_through_clamp():
    print("\n--- RL: straight-through clamp in decode_pixels ---")
    vae = _DummyVAE()

    saturated = torch.full((1, 1, 8, 8), 3.0, requires_grad=True)
    saturated.clamp(-1.0, 1.0).sum().backward()
    hard_grad = saturated.grad.abs().sum().item()

    saturated_st = torch.full((1, 1, 8, 8), 3.0, requires_grad=True)
    decode_pixels(vae, saturated_st).sum().backward()
    st_grad = saturated_st.grad.abs().sum().item()

    check("hard clamp kills gradient where the decoder saturates", hard_grad == 0.0)
    check("straight-through preserves it", st_grad > 0.0, f"{st_grad:.3f}")

    values = torch.linspace(-3.0, 3.0, 64).view(1, 1, 8, 8)
    expected = values.expand(-1, 3, -1, -1).clamp(-1.0, 1.0)
    check(
        "forward value equals the hard clamp",
        torch.equal(decode_pixels(vae, values), expected),
    )


def test_normalize_grad():
    print("\n--- chain: gradient normalizer ---")
    x = torch.randn(3, 4, 8, 8, requires_grad=True)
    upstream = torch.randn(3, 4, 8, 8) * torch.tensor([1e-6, 1.0, 1e6]).view(3, 1, 1, 1)

    normalize_grad(x, 1.0).backward(upstream)
    rms = x.grad.pow(2).flatten(1).mean(1).sqrt()

    check(
        "per-sample gradient RMS is driven to the target",
        torch.allclose(rms, torch.ones(3), atol=1e-4),
        f"rms={[round(v, 4) for v in rms.tolist()]}",
    )

    cos = torch.nn.functional.cosine_similarity(
        x.grad.flatten(1), upstream.flatten(1), dim=1
    )
    check(
        "direction is preserved (magnitude only)",
        torch.allclose(cos, torch.ones(3), atol=1e-5),
    )

    y = torch.randn(2, 4, 8, 8, requires_grad=True)
    up = torch.randn(2, 4, 8, 8)
    normalize_grad(y, 0.0).backward(up)
    check("target_rms=0 disables the normalizer", torch.equal(y.grad, up))


def test_chain_lora_keys():
    print("\n--- chain: adapter key rewrite ---")
    named = [
        ("down_blocks.0.attentions.0.attn1.to_q.lora_A.early.weight", torch.zeros(2)),
        ("down_blocks.0.attentions.0.attn1.to_q.lora_B.early.weight", torch.zeros(2)),
        ("down_blocks.0.attentions.0.attn1.to_q.lora_A.late.weight", torch.zeros(2)),
        ("down_blocks.0.attentions.0.attn1.to_q.base_layer.weight", torch.zeros(2)),
    ]
    unet = SimpleNamespace(named_parameters=lambda: iter(named))

    with tempfile.TemporaryDirectory() as tmp:
        count = save_chain_lora(unet, tmp, "early")
        from safetensors.torch import load_file

        keys = set(load_file(os.path.join(tmp, "early.safetensors")))

    check("only the requested adapter is written", count == 2, f"{count} tensors")
    check(
        "keys match the flat layout app.py loads",
        keys
        == {
            "unet.down_blocks.0.attentions.0.attn1.to_q.lora_A.weight",
            "unet.down_blocks.0.attentions.0.attn1.to_q.lora_B.weight",
        },
        str(sorted(keys)[0]),
    )


_CAPTION = (
    "A young woman with long blonde hair, wearing a purple and white dress "
    "with a red bow, stands on a glowing blue platform. She holds a small "
    "object near her mouth. The background features a surreal, dreamlike "
    "landscape with a large, stylized eye floating above her."
)


def test_caption_subsetting():
    print("\n--- caption subsetting ---")

    clauses = split_clauses(_CAPTION)
    check("caption splits into clauses", len(clauses) == 7, str(len(clauses)))

    single = "A young woman standing"
    check(
        "single-clause captions pass through",
        subset_caption(single, random.Random(0)) == single,
    )
    check(
        "min_keep=1.0 keeps the whole caption",
        subset_caption(_CAPTION, random.Random(0), min_keep=1.0) == _CAPTION,
    )
    check(
        "subsetting is deterministic per seeded rng",
        subset_caption(_CAPTION, random.Random(7))
        == subset_caption(_CAPTION, random.Random(7)),
    )

    rng = random.Random(0)
    draws = [subset_caption(_CAPTION, rng) for _ in range(200)]
    full_words = _CAPTION.split()

    check("no draw is empty", all(d.strip() for d in draws))
    check(
        "no draw is longer than the caption",
        all(len(d.split()) <= len(full_words) for d in draws),
    )

    prefix_ok = True
    for draw in draws:
        for index, word in enumerate(draw.split()):
            if word.strip(string.punctuation) != full_words[index].strip(
                string.punctuation
            ):
                prefix_ok = False
    check("every draw is a word-prefix of the caption", prefix_ok)

    lengths = [len(d.split()) for d in draws]
    mean_length = sum(lengths) / len(lengths)
    check(
        "the draw distribution shortens captions",
        mean_length < 0.75 * len(full_words),
        f"{mean_length:.1f} of {len(full_words)}",
    )
    check(
        "the draw distribution reaches short prompts",
        min(lengths) <= 8,
        f"min {min(lengths)}",
    )
    check(
        "the draw distribution still includes the full caption",
        max(lengths) == len(full_words),
        f"max {max(lengths)}",
    )


def test_shuffle_buffer_size():
    print("\n--- shuffle buffer sizing ---")

    check(
        "config value caps the buffer on a large cache",
        shuffle_buffer_size(6000, 1000) == 1000,
        str(shuffle_buffer_size(6000, 1000)),
    )
    check(
        "smoke-sized runs keep the 256 floor",
        shuffle_buffer_size(96, 1000) == 256,
        str(shuffle_buffer_size(96, 1000)),
    )
    check(
        "the buffer never exceeds the cache it feeds, above the floor",
        shuffle_buffer_size(400, 3000) == 400,
        str(shuffle_buffer_size(400, 3000)),
    )
    check(
        "peak RAM is bounded by the buffer, not cache_size",
        shuffle_buffer_size(20000, 1000) == shuffle_buffer_size(6000, 1000),
    )


def test_mask_normalization():
    print("\n--- mask normalization ---")

    flat = torch.full((4, 1, 16, 16), 0.37)
    noisy_flat = flat + torch.randn_like(flat) * 1e-6

    stretched = percentile_normalize(noisy_flat)
    check(
        "percentile_normalize manufactures full contrast from noise",
        stretched.flatten(1).std(1).mean() > 0.2,
        f"std {stretched.flatten(1).std(1).mean():.3f}",
    )

    relative = relative_normalize(noisy_flat, gain=1.5)
    check(
        "relative_normalize leaves a flat map flat",
        relative.flatten(1).std(1).max() < 1e-3,
        f"std {relative.flatten(1).std(1).max():.2e}",
    )
    check(
        "a flat map normalizes to the neutral 0.5",
        (relative - 0.5).abs().max() < 1e-3,
    )

    pattern = torch.randn(4, 1, 16, 16)
    pattern = pattern - pattern.flatten(1).median(dim=1).values.view(-1, 1, 1, 1)
    pattern = pattern / pattern.flatten(1).abs().max(dim=1).values.view(-1, 1, 1, 1)

    weak = relative_normalize(1.0 + 0.05 * pattern, gain=1.5)
    strong = relative_normalize(1.0 + 0.10 * pattern, gain=1.5)

    weak_std = weak.flatten(1).std(1)
    strong_std = strong.flatten(1).std(1)
    ratio = (strong_std / weak_std.clamp(min=1e-8)).mean()
    check(
        "mask contrast is proportional to signal contrast",
        abs(float(ratio) - 2.0) < 0.05,
        f"ratio {float(ratio):.3f}",
    )

    scaled = relative_normalize(7.5 * (1.0 + 0.05 * pattern), gain=1.5)
    check(
        "normalization is invariant to the map's absolute scale",
        torch.allclose(weak, scaled, atol=1e-5),
    )
    check(
        "output stays inside [0, 1]",
        float(strong.min()) >= 0.0 and float(strong.max()) <= 1.0,
    )
    check(
        "gain=0 falls back to the percentile stretch",
        torch.equal(normalize_mask(noisy_flat, 0.0), percentile_normalize(noisy_flat)),
    )

    constant = torch.full((2, 1, 32, 48), 0.5)
    blurred = _gaussian_blur(constant, 43, 7.0)
    check(
        "blur preserves shape",
        blurred.shape == constant.shape,
        str(tuple(blurred.shape)),
    )
    check(
        "blur does not darken the edges of a constant map",
        (blurred - 0.5).abs().max() < 1e-5,
        f"max drift {float((blurred - 0.5).abs().max()):.2e}",
    )


def test_noise_prior_match():
    print("\n--- noise prior: trainer and sampler agree ---")
    shape = (256, 4, 64, 64)
    torch.manual_seed(2)

    sigma = (INIT_SIGMA**2 - 1.0) ** 0.5
    data = torch.tensor(LATENT_MEAN).view(1, 4, 1, 1) + lowfreq_field(shape)
    training = data + sigma * lowfreq_noise(torch.zeros(shape))
    inference = INIT_SIGMA * prior_latents(
        shape,
        [torch.Generator().manual_seed(500 + i) for i in range(shape[0])],
        "cpu",
        torch.float32,
        init_sigma=INIT_SIGMA,
    )

    dc_train = training.mean(dim=(2, 3))
    dc_infer = inference.mean(dim=(2, 3))
    mean_gap = (dc_train.mean(0) - dc_infer.mean(0)).abs().max().item()
    var_ratio = (dc_infer.var(0) / dc_train.var(0)).tolist()
    check(
        "inference x_T has the per-channel DC mean training x_T has",
        mean_gap < 0.12,
        f"max gap {mean_gap:.3f}",
    )
    check(
        "inference x_T has the per-channel DC spread training x_T has",
        all(0.7 < r < 1.4 for r in var_ratio),
        f"ratios {[round(r, 2) for r in var_ratio]}",
    )
    check(
        "deviation gain at the 30-step start sigma",
        abs(prior_deviation(INIT_SIGMA) - 1.272) < 0.01,
        f"{prior_deviation(INIT_SIGMA):.3f}",
    )


def test_dual_encoder_encoding():
    print("\n--- SDXL prompt encoding ---")
    torch.manual_seed(0)
    tokenizers = [_DummyTokenizer(49407), _DummyTokenizer(0)]
    encoders = [_DummyTextEncoder(768), _DummyTextEncoder(1280, projection=1280)]

    prompts = ["a girl with long pink hair", "a quiet room"]
    embeddings, pooled, content, valid = encode_prompt(
        prompts, encoders, tokenizers, "cpu"
    )

    check(
        "both encoders concatenate to cross_attention_dim 2048",
        tuple(embeddings.shape) == (2, 154, 2048),
        str(tuple(embeddings.shape)),
    )
    check(
        "the split point separates the two encoders",
        HIDDEN_SPLIT == 768 and embeddings.shape[-1] - HIDDEN_SPLIT == 1280,
    )
    check(
        "pooled embedding is one 1280-vector per prompt",
        tuple(pooled.shape) == (2, 1280),
        str(tuple(pooled.shape)),
    )
    check(
        "content mask still indexes the 154-token sequence",
        tuple(content.shape) == (2, 154),
        str(tuple(content.shape)),
    )

    check(
        "BOS and EOS are excluded from the content mask",
        content[0, 0] == 0 and content[0, 76] == 0 and content[0, 77] == 0,
    )
    check(
        "stopwords are excluded and content words are kept",
        content[0, 1:7].tolist() == [0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        str(content[0, 1:7].tolist()),
    )
    check(
        "padding and the unused second chunk contribute nothing",
        content[0, 7:76].sum() == 0 and content[0, 77:].sum() == 0,
    )

    ids = encoders[1].seen_input_ids
    check(
        "EOS closes the content, before the padding",
        ids[0, 0] == 49406
        and ids[0, 7] == 49407
        and (ids[0, 1:7] != 49407).all()
        and (ids[0, 8:] == 0).all(),
        str(ids[0, :10].tolist()),
    )
    check(
        "an empty chunk is BOS, EOS, then padding only",
        ids[1, 0] == 49406 and ids[1, 1] == 49407 and (ids[1, 2:] == 0).all(),
        str(ids[1, :4].tolist()),
    )

    words = [f"word{i}" for i in range(75)]
    short = " ".join(words)
    long = " ".join(words + ["extra"] * 5)
    _, pooled_short, _, _ = encode_prompt([short], encoders, tokenizers, "cpu")
    _, pooled_long, _, _ = encode_prompt([long], encoders, tokenizers, "cpu")
    check(
        "pooled comes from the first chunk only",
        torch.allclose(pooled_short, pooled_long, atol=1e-5),
    )

    check(
        "a short prompt marks only its first chunk valid",
        bool(valid[:, :CHUNK_LENGTH].all()) and not bool(valid[:, CHUNK_LENGTH:].any()),
    )
    check("a short batch packs to one chunk", active_chunks(valid) == 1)

    mixed_emb, _, mixed_content, mixed_valid = encode_prompt(
        [prompts[0], long], encoders, tokenizers, "cpu"
    )
    check(
        "an 80-token prompt spills into the second chunk",
        active_chunks(mixed_valid[1:]) == 2,
    )

    packed, mask, (packed_content,) = pack_chunks(embeddings, valid, content)
    check(
        "a one-chunk batch trims to 77 tokens and needs no mask",
        packed.shape[1] == CHUNK_LENGTH and mask is None,
        str(tuple(packed.shape)),
    )
    check(
        "the content mask is trimmed alongside",
        packed_content.shape[1] == CHUNK_LENGTH,
    )

    packed, mask, _ = pack_chunks(mixed_emb, mixed_valid, mixed_content)
    check(
        "a mixed batch keeps both chunks and masks the short row's empty one",
        packed.shape[1] == 2 * CHUNK_LENGTH
        and mask is not None
        and float(mask[0, CHUNK_LENGTH:].sum()) == 0.0
        and float(mask[1].sum()) == 2 * CHUNK_LENGTH,
    )


def test_capture_mask():
    print("\n--- masked cross-attention equals trimmed cross-attention ---")
    from diffusers.models.attention_processor import Attention

    torch.manual_seed(4)
    attn = Attention(query_dim=8, cross_attention_dim=6, heads=2, dim_head=4)
    hidden = torch.randn(2, 5, 8)
    context = torch.randn(2, 6, 6)
    keep = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.float32)
    bias = ((1 - keep) * -10000.0).unsqueeze(1)

    seen: list = []
    processor = _CapturingProcessor(SimpleNamespace(absorb=seen.append))
    with torch.no_grad():
        masked = processor(
            attn, hidden, encoder_hidden_states=context, attention_mask=bias
        )
        trimmed = processor(attn, hidden[:1], encoder_hidden_states=context[:1, :3])
        full = processor(attn, hidden[1:], encoder_hidden_states=context[1:])
    weights = seen[0].view(2, 2, 5, 6)

    check(
        "masked tokens receive no attention",
        float(weights[0, :, :, 3:].abs().max()) < 1e-6,
    )
    check(
        "masking the empty chunk is the same as removing it",
        torch.allclose(masked[:1], trimmed, atol=1e-5),
        f"max diff {float((masked[:1] - trimmed).abs().max()):.2e}",
    )
    check("unmasked rows are untouched", torch.allclose(masked[1:], full, atol=1e-5))


def test_nan_guards():
    print("\n--- NaN guards: capture scores, discrepancy gate, stage report ---")
    from diffusers.models.attention_processor import Attention

    attn = Attention(query_dim=8, cross_attention_dim=6, heads=1, dim_head=64).half()
    with torch.no_grad():
        attn.to_q.weight.fill_(5.7)
        attn.to_k.weight.fill_(5.7)
    hidden = torch.ones(1, 4, 8, dtype=torch.float16)
    context = torch.ones(1, 3, 6, dtype=torch.float16)
    seen: list = []
    processor = _CapturingProcessor(SimpleNamespace(absorb=seen.append))
    with torch.no_grad():
        q, k = attn.to_q(hidden), attn.to_k(context)
        unscaled = torch.bmm(q, k.transpose(-1, -2))
        processor(attn, hidden, encoder_hidden_states=context)
    check(
        "capture scores stay finite where the unscaled fp16 product overflows",
        bool(torch.isinf(unscaled).any())
        and bool(torch.isfinite(seen[0]).all())
        and torch.allclose(seen[0].sum(-1), torch.ones(1, 4)),
        f"unscaled max {float(unscaled.float().max()):.0f}",
    )

    attn_mask = torch.rand(2, 1, 4, 4)
    diff = torch.full((2, 1, 4, 4), float("nan"))
    blended = blend_masks(
        attn_mask, diff, (4, 4), torch.tensor([0.0, 0.5]).view(2, 1, 1, 1), gain=1.5
    )
    check(
        "a gated-off discrepancy cannot leak NaN into the mask",
        torch.equal(blended[0], attn_mask[0]) and bool(torch.isnan(blended[1]).all()),
    )

    bad = finite_stages(
        base_pred=torch.ones(2),
        attn_mask=torch.tensor([1.0, float("inf")]),
        noise_scale=torch.tensor([float("nan")]),
    )
    check(
        "the stage report names the first non-finite stage, else the LoRA forward",
        first_nonfinite(bad) == "attn_mask"
        and first_nonfinite(finite_stages(x=torch.ones(3))) == "the LoRA forward",
    )


def test_chain_window():
    print("\n--- chain: a grad window anywhere in the chain reaches the LoRA ---")
    steps = 8
    sigmas = torch.linspace(10.0, 0.0, steps + 1)
    timesteps = torch.linspace(900.0, 20.0, steps)
    embeds = torch.randn(2, 5, 16)
    pooled = torch.randn(2, 12)

    def anchor_at(index):  # noqa: ARG001
        return embeds, pooled

    def run(window):
        unet = _DummyUNet()
        pipe = SimpleNamespace(
            unet=unet,
            enable_lora=lambda: None,
            disable_lora=lambda: None,
            set_adapters=lambda *args, **kwargs: None,  # noqa: ARG005
        )
        out = rollout_chain(
            pipe,
            10.05,
            sigmas,
            timesteps,
            torch.randn(2, 4, 8, 8),
            embeds,
            pooled,
            torch.zeros(4, 6),
            anchor_at,
            lambda index: 5.0,  # noqa: ARG005
            1.0,
            True,
            window,
            0.0,
        )
        if not out.requires_grad:
            return 0.0
        out.pow(2).mean().backward()
        grad = unet.conv.weight.grad
        return 0.0 if grad is None else float(grad.abs().sum())

    middle, start, end = run({3, 4}), run({0, 1}), run({6, 7})
    check(
        "windows at the start, middle and end all produce gradient",
        middle > 0 and start > 0 and end > 0,
        f"start {start:.2e} middle {middle:.2e} end {end:.2e}",
    )
    check("an empty window produces none", run(set()) == 0.0)


class _BlendUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.lora = torch.nn.ParameterDict(
            {
                name: torch.nn.Parameter(torch.full((1, 4, 1, 1), 0.1))
                for name in ALL_SEGMENTS
            }
        )
        self.active: dict = {}
        self.calls: list = []

    def forward(self, sample, t, encoder_hidden_states=None, **kwargs):  # noqa: ARG002
        out = self.conv(sample.float())
        for name, weight in self.active.items():
            out = out + weight * self.lora[name] * sample.float()
        return SimpleNamespace(sample=out)


def _blend_pipe(unet):
    def set_adapters(names, weights):
        unet.active = dict(zip(names, weights))
        unet.calls.append(dict(unet.active))

    def disable_lora():
        unet.active = {}
        unet.calls.append({})

    return SimpleNamespace(
        unet=unet,
        enable_lora=lambda: None,
        disable_lora=disable_lora,
        set_adapters=set_adapters,
    )


def test_chain_blend_weights():
    print("\n--- chain: each adapter's gradient is weighted by its blend weight ---")
    steps = NUM_INFERENCE_STEPS
    sigmas = torch.linspace(10.0, 0.0, steps + 1)
    timesteps = torch.linspace(958.0, 1.0, steps)
    embeds = torch.randn(1, 5, 16)
    pooled = torch.randn(1, 12)

    def anchor_at(index):  # noqa: ARG001
        return embeds, pooled

    def run(window, strength=1.0):
        torch.manual_seed(0)
        unet = _BlendUNet()
        out = rollout_chain(
            _blend_pipe(unet),
            10.05,
            sigmas,
            timesteps,
            torch.randn(1, 4, 8, 8),
            embeds,
            pooled,
            torch.zeros(2, 6),
            anchor_at,
            lambda index: 5.0,  # noqa: ARG005
            strength,
            True,
            window,
            0.0,
        )
        out.pow(2).mean().backward()
        grads = {name: unet.lora[name].grad for name in ALL_SEGMENTS}
        return grads, unet.calls

    check(
        "active_adapters drops zero-weight adapters",
        active_adapters(ALL_SEGMENTS, [0.75, 0.25, 0.0])
        == (["early", "mid"], [0.75, 0.25])
        and active_adapters(ALL_SEGMENTS, [0.0, 0.0, 0.0]) == ([], []),
    )

    blend_step = EARLY_SEG - 1
    weights = adapter_weights(blend_step, 1.0)
    grads, calls = run({blend_step})
    early, mid = grads["early"], grads["mid"]
    assert early is not None and mid is not None
    ratio = float(mid.abs().sum() / early.abs().sum())
    check(
        "at a blend step the gradient ratio equals the weight ratio",
        abs(ratio - weights[1] / weights[0]) < 1e-4,
        f"step {blend_step} weights {weights[:2]} grad ratio {ratio:.5f}",
    )
    check(
        "an adapter with zero weight at every grad step gets no gradient at all",
        grads["late"] is None,
    )
    check(
        "the final step runs with every adapter off",
        calls[steps - 1] == {},
        str(calls[steps - 1]),
    )
    check(
        "the backward recompute re-applies the grad step's blend, not the last step's",
        len(calls) == steps + 1
        and calls[-1] == {"early": weights[0], "mid": weights[1]},
        str(calls[-1]),
    )

    only_early, _ = run({0, 1, 2})
    check(
        "a window inside one segment trains only that adapter",
        only_early["early"] is not None
        and only_early["mid"] is None
        and only_early["late"] is None,
    )


def test_grad_window_coverage():
    print("\n--- RL: grad windows cover every trainable step equally ---")
    trainable, window, draws = NUM_INFERENCE_STEPS - 1, 10, 40000
    rng = random.Random(0)
    counts = [0] * NUM_INFERENCE_STEPS
    contiguous = True
    for _ in range(draws):
        steps = sample_grad_window(rng, trainable, window)
        contiguous = contiguous and steps == set(range(min(steps), max(steps) + 1))
        for index in steps:
            counts[index] += 1
    expected = draws * window / (trainable + window - 1)
    worst = max(abs(c - expected) / expected for c in counts[:trainable])
    check(
        "per-step coverage is uniform within 3%",
        worst < 0.03,
        f"worst {worst:.3f}, ends {counts[0]}/{counts[trainable - 1]}, mid {counts[14]}",
    )
    check("the final, adapter-free step is never in a window", counts[trainable] == 0)
    check("windows are contiguous and non-empty", contiguous)
    check(
        "window >= trainable steps covers everything",
        sample_grad_window(rng, trainable, 40) == set(range(trainable))
        and sample_grad_window(rng, trainable, 0) == set(range(trainable)),
    )
    exposure = [0.0, 0.0, 0.0]
    for index in range(trainable):
        for segment, value in enumerate(
            adapter_exposure({index}, 1.0, NUM_INFERENCE_STEPS)
        ):
            exposure[segment] += value
    check(
        "under uniform coverage no adapter gets more than 1.25x another's exposure",
        max(exposure) / min(exposure) < 1.25,
        " / ".join(f"{e:.2f}" for e in exposure),
    )


def test_clip_by_exposure():
    print("\n--- RL: per-adapter clipping scales with blend exposure ---")
    big = torch.nn.Parameter(torch.zeros(4))
    small = torch.nn.Parameter(torch.zeros(4))
    idle = torch.nn.Parameter(torch.zeros(4))
    big.grad = torch.tensor([6.0, 8.0, 0.0, 0.0])
    small.grad = torch.tensor([0.06, 0.08, 0.0, 0.0])
    direction = big.grad / big.grad.norm()
    before, after = clip_by_exposure(
        {"early": [big], "mid": [small], "late": [idle]},
        {"early": 2.0, "mid": 0.5, "late": 0.0},
        1.0,
    )
    check(
        "an adapter over its limit is scaled to clip x exposure, direction kept",
        abs(float(big.grad.norm()) - 2.0) < 1e-6
        and torch.allclose(big.grad / big.grad.norm(), direction)
        and before["early"] == 10.0
        and after["early"] == 2.0,
        f"{before['early']} -> {float(big.grad.norm()):.4f}",
    )
    check(
        "an adapter under its limit is untouched",
        abs(float(small.grad.norm()) - 0.1) < 1e-6,
    )
    check(
        "an adapter with no gradient stays None",
        idle.grad is None and before["late"] is None,
    )


def test_sampling_chunk_mask():
    print("\n--- sampling: per-row chunk mask for mixed-length batches ---")

    def valid(*chunks):
        rows = []
        for count in chunks:
            rows.append(
                [1] * (count * CHUNK_LENGTH)
                + [0] * ((MAX_CHUNKS - count) * CHUNK_LENGTH)
            )
        return torch.tensor(rows, dtype=torch.bool)

    check(
        "a single prompt needs no mask",
        sampling_chunk_mask(valid(1), valid(1), 1) is None
        and sampling_chunk_mask(valid(2), valid(1), 2) is None,
    )
    mixed = sampling_chunk_mask(valid(1, 2), valid(1), 2)
    assert mixed is not None
    check(
        "a short row in a long batch masks only its own empty chunk",
        tuple(mixed.shape) == (2, 2 * CHUNK_LENGTH)
        and bool(mixed[0, :CHUNK_LENGTH].all())
        and not bool(mixed[0, CHUNK_LENGTH:].any())
        and bool(mixed[1].all()),
    )
    check(
        "a two-chunk negative keeps both chunks for every row, as unbatched sampling does",
        sampling_chunk_mask(valid(1, 2), valid(2), 2) is None,
    )


def test_rl_reward():
    print("\n--- RL: floor reward and sampling helpers ---")
    floor, temperature = 5.0, 0.25
    values = torch.tensor([3.0, 4.5, 5.0, 6.0, 8.0], requires_grad=True)
    penalty = floor_penalty(values, floor, temperature)
    penalty.sum().backward()
    grad = values.grad
    assert grad is not None

    check(
        "far below the floor the penalty is linear (slope -1)",
        abs(grad[0].item() + 1.0) < 1e-3 and abs(penalty[0].item() - 2.0) < 1e-3,
        f"grad {grad[0].item():.4f} penalty {penalty[0].item():.4f}",
    )
    check(
        "well above the floor there is effectively no push",
        abs(grad[4].item()) < 1e-4 and penalty[4].item() < 1e-4,
        f"grad {grad[4].item():.2e}",
    )
    check(
        "the push fades monotonically across the floor",
        bool((grad[:-1].abs() >= grad[1:].abs()).all()),
    )

    check(
        "scaled buckets stay multiples of 32",
        all(
            side % 32 == 0
            for bucket in ((832, 1216), (1024, 1024), (1344, 768))
            for scale in (1.0, 0.75, 0.5)
            for side in scaled_bucket(bucket, scale)
        ),
    )
    check("scale 1 keeps the bucket", scaled_bucket((832, 1216), 1.0) == (832, 1216))

    rng = random.Random(0)
    composed = {composed_prompt(rng) for _ in range(300)}
    check(
        "composed RL prompts never reuse an eval prompt",
        not composed & set(PROMPTS),
        f"{len(composed)} distinct",
    )

    portrait = torch.randn(1, 3, 600, 400, requires_grad=True)
    views = square_views(portrait, views=3)
    check(
        "portrait views are 224 squares",
        tuple(views.shape) == (1, 3, 3, 224, 224),
        str(tuple(views.shape)),
    )
    views.sum().backward()
    view_grad = portrait.grad
    assert view_grad is not None
    check(
        "views cover both ends of the long axis",
        float(view_grad[..., :20, :].abs().sum()) > 0
        and float(view_grad[..., -20:, :].abs().sum()) > 0,
    )
    check(
        "a square image is a single view",
        square_views(torch.randn(2, 3, 300, 300)).shape[1] == 1,
    )


def test_rms_split():
    print("\n--- embedding jitter across two encoder blocks ---")
    reference = torch.cat(
        [torch.full((1, 154, 768), 0.1), torch.full((1, 154, 1280), 10.0)], dim=-1
    )

    torch.manual_seed(0)
    split_noise = rms_scaled_noise(reference, 1.0, split=HIDDEN_SPLIT)
    head = split_noise[..., :HIDDEN_SPLIT].float().std().item()
    tail = split_noise[..., HIDDEN_SPLIT:].float().std().item()

    check(
        "each encoder block is jittered by its own RMS",
        abs(head / 0.1 - 1.0) < 0.05 and abs(tail / 10.0 - 1.0) < 0.05,
        f"head={head:.4f} (want 0.1) tail={tail:.3f} (want 10.0)",
    )

    global_noise = rms_scaled_noise(reference, 1.0)
    global_head = global_noise[..., :HIDDEN_SPLIT].float().std().item()
    check(
        "a single global RMS would swamp the smaller block",
        global_head > 10.0 * head,
        f"{global_head:.3f} vs {head:.4f}",
    )
    check(
        "split=None keeps the original single-RMS behaviour",
        abs(global_head / (10.0 * (1280 / 2048) ** 0.5) - 1.0) < 0.1,
        f"{global_head:.3f}",
    )


def test_size_conditioning():
    print("\n--- SDXL size micro-conditioning ---")

    ids = build_time_ids((512, 384), (12, 0), (1216, 832))
    check(
        "time ids are original, crop, target in that order",
        ids == (512.0, 384.0, 12.0, 0.0, 1216.0, 832.0),
        str(ids),
    )
    check("six values feed the 6 x 256 time embedding", len(ids) == 6)

    tensor, _, bucket, sample_ids = prepare_sample(
        {"image": Image.new("RGB", (400, 300)), "text": "a girl"}, "cpu"
    )
    assert tensor is not None and sample_ids is not None
    check(
        "a 4:3 source lands in the nearest SDXL bucket",
        bucket == (1152, 896),
        str(bucket),
    )
    check(
        "the tensor matches the bucket",
        tuple(tensor.shape) == (1, 3, 896, 1152),
        str(tuple(tensor.shape)),
    )
    check(
        "original_size reports the true pre-upscale size, not the bucket",
        sample_ids[0] == 300.0 and sample_ids[1] == 400.0,
        str(sample_ids[:2]),
    )
    check(
        "the discarded centre-crop offset is carried through",
        sample_ids[2] == 0.0 and sample_ids[3] == 21.0,
        str(sample_ids[2:4]),
    )
    check(
        "target_size is the bucket in (height, width) order",
        sample_ids[4] == 896.0 and sample_ids[5] == 1152.0,
        str(sample_ids[4:]),
    )


def test_alpha_skip():
    print("\n--- Images with alpha are skipped ---")

    palette = Image.new("P", (2, 1))
    palette.info["transparency"] = 0
    keyed = Image.new("RGB", (2, 1))
    keyed.info["transparency"] = (0, 0, 0)
    alpha = {
        "RGBA": Image.new("RGBA", (4, 4)),
        "LA": Image.new("LA", (4, 4)),
        "P with a transparent index": palette,
        "RGB with a transparent colour key": keyed,
    }
    for name, image in alpha.items():
        check(f"{name} is detected", has_alpha(image))
        result = prepare_sample({"image": image, "text": "a girl"}, "cpu")
        check(f"{name} is skipped", result == (None, None, None, None))

    for mode in ("RGB", "L", "P", "CMYK"):
        image = Image.new(mode, (4, 4))
        check(f"plain {mode} is not alpha", not has_alpha(image))
        tensor, _, _, _ = prepare_sample({"image": image, "text": "a girl"}, "cpu")
        check(f"plain {mode} is kept", tensor is not None)


_APP_PATH = pathlib.Path(__file__).resolve().parent.parent / "app.py"

_APP_MIRRORED_CONSTANTS = {
    "EARLY_SEG": EARLY_SEG,
    "MID_SEG": MID_SEG,
    "NUM_INFERENCE_STEPS": NUM_INFERENCE_STEPS,
    "_BLEND_HALF": _BLEND_HALF,
    "_MAX_RAW_GUIDANCE": _MAX_RAW_GUIDANCE,
    "CHUNK_TOKENS": CHUNK_TOKENS,
    "MAX_CHUNKS": MAX_CHUNKS,
    "HIDDEN_SPLIT": HIDDEN_SPLIT,
    "POOLED_DIM": POOLED_DIM,
    "LATENT_MEAN": LATENT_MEAN,
    "LATENT_DC_STD": LATENT_DC_STD,
    "PRIOR_LEVELS": PRIOR_LEVELS,
    "PRIOR_DC": PRIOR_DC,
    "TERMINAL_LEAK": TERMINAL_LEAK,
    "NOISE_LF_STRENGTH": NOISE_LF_STRENGTH,
    "NEGATIVE_PROMPT": NEGATIVE_PROMPT,
}


def _module_constants(path: pathlib.Path) -> dict:
    found = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            found[target.id] = ast.literal_eval(node.value)
        except ValueError:
            continue
    return found


def test_app_constant_parity():
    print("\n--- app.py constant parity ---")

    if not _APP_PATH.exists():
        check(
            "app.py is absent, so there is nothing to diff (expected in the image)",
            True,
            str(_APP_PATH),
        )
        return

    found = _module_constants(_APP_PATH)

    for name, expected in _APP_MIRRORED_CONSTANTS.items():
        present = name in found
        check(
            f"app.py defines {name}",
            present,
            "" if present else "not found as a module-level literal",
        )
        if present:
            check(
                f"app.py {name} matches src",
                found[name] == expected,
                f"app={found[name]!r} src={expected!r}",
            )


def main() -> int:
    torch.manual_seed(0)
    random.seed(0)
    scheduler = make_scheduler()

    test_lowfreq_noise(scheduler)
    test_noise_prior_match()
    test_dual_encoder_encoding()
    test_capture_mask()
    test_nan_guards()
    test_rms_split()
    test_size_conditioning()
    test_alpha_skip()
    test_min_snr(scheduler)
    test_loss_weighting(scheduler)
    test_bucket_sampler()
    test_conditioning_augmentation()
    test_caption_subsetting()
    test_shuffle_buffer_size()
    test_mask_normalization()
    test_predict_x0(scheduler)
    test_sampler_parity()
    test_app_constant_parity()
    test_eval_metrics()
    test_paired_deltas()
    test_base_reuse()
    test_segment_plan()
    test_latent_cache_persistence()
    test_straight_through_clamp()
    test_normalize_grad()
    test_chain_lora_keys()
    test_chain_window()
    test_chain_blend_weights()
    test_grad_window_coverage()
    test_sampling_chunk_mask()
    test_clip_by_exposure()
    test_rl_reward()

    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {_failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
