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

from .cache import BucketSampler
from .chainrl import normalize_grad, save_chain_lora
from .dataset import build_time_ids, prepare_sample, shuffle_buffer_size
from .encoder.mask_builder import (
    _gaussian_blur,
    normalize_mask,
    percentile_normalize,
    relative_normalize,
)
from .encoding import (
    CHUNK_TOKENS,
    HIDDEN_SPLIT,
    MAX_CHUNKS,
    POOLED_DIM,
    decode_for_clip,
    encode_prompt,
    rms_scaled_noise,
    split_clauses,
    subset_caption,
)
from .model import EARLY_SEG, MID_SEG, NUM_INFERENCE_STEPS
from .eval import adherence, angular_spread, radius
from .loss import compute_diffusion_loss, min_snr_weight
from .rl import _ddpm_posterior, rollout_segment
from .sampler import (
    LATENT_LF_DECAY,
    LATENT_LF_LEVELS,
    _BLEND_HALF,
    _MAX_RAW_GUIDANCE,
    adapter_weights,
    pyramid_latents,
)
from .scheduler import (
    SpatiallyVaryingDDPMScheduler,
    compute_spatial_noise_scale,
    pyramid_noise,
)
from .train import apply_conditioning_augmentation

TERMINAL_ALPHA_BAR = 0.004660

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
        hidden = self.embedding(input_ids)
        output = SimpleNamespace(
            hidden_states=[hidden, hidden, hidden], last_hidden_state=hidden
        )
        if self.projection is not None:
            output.text_embeds = self.projection(hidden.mean(1))
        return output


def test_pyramid_noise():
    print("\n--- pyramid_noise ---")
    latents = torch.zeros(6, 4, 128, 128)
    noise = pyramid_noise(latents, levels=LATENT_LF_LEVELS, decay=LATENT_LF_DECAY)

    check("shape preserved", noise.shape == latents.shape, str(tuple(noise.shape)))
    check("dtype preserved", noise.dtype == latents.dtype)

    rms = noise.pow(2).flatten(1).mean(1).sqrt()
    check(
        "per-sample RMS is 1 (variance invariant held)",
        torch.allclose(rms, torch.ones_like(rms), atol=1e-4),
        f"rms={[round(v, 5) for v in rms.tolist()[:3]]}",
    )

    bulk = torch.zeros(600, 4, 128, 128)
    dc_plain = pyramid_noise(bulk, levels=1).mean(dim=(2, 3)).var().item()
    dc_pyramid = (
        pyramid_noise(bulk, levels=LATENT_LF_LEVELS, decay=LATENT_LF_DECAY)
        .mean(dim=(2, 3))
        .var()
        .item()
    )

    check(
        "defaults bring DC noise to parity with the terminal-SNR leak",
        0.5 < TERMINAL_ALPHA_BAR / dc_pyramid < 2.0,
        f"signal/noise DC {TERMINAL_ALPHA_BAR / dc_plain:.1f}x -> "
        f"{TERMINAL_ALPHA_BAR / dc_pyramid:.2f}x",
    )
    check(
        "levels=1 reduces to plain randn",
        abs(dc_plain - 1.0 / (128 * 128)) < 1e-5,
        f"dc={dc_plain:.6f} expected={1 / 16384:.6f}",
    )
    check(
        "the pyramid reaches the 1x1 floor at the SDXL latent size",
        LATENT_LF_LEVELS >= 8 and 128 // 2 ** (LATENT_LF_LEVELS - 1) == 1,
        f"levels={LATENT_LF_LEVELS}",
    )
    check(
        "survives latents too small for every level",
        pyramid_noise(torch.zeros(2, 4, 8, 6), levels=LATENT_LF_LEVELS).shape
        == (2, 4, 8, 6),
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


def test_x0_reward(scheduler):
    print("\n--- RL: x0 reward vs the noisy latent it replaced ---")
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


def test_rollout(scheduler):
    print("\n--- RL: rollout shapes, grads, determinism ---")
    grid = scheduler.timesteps
    policy, base = _DummyUNet(), _DummyUNet()
    start = torch.randn(8, 4, 16, 16)
    embeddings = torch.randn(8, 154, 2048)
    added_cond = {"text_embeds": torch.randn(8, 1280), "time_ids": torch.zeros(8, 6)}
    indices = list(range(10))

    torch.manual_seed(7)
    generated, base_generated, log_prob = rollout_segment(
        policy,
        base,
        scheduler,
        start,
        indices,
        grid,
        embeddings,
        added_cond,
        True,
        set(indices),
    )
    check("x0 prediction shape", generated.shape == start.shape)
    check(
        "base x0 prediction shape",
        base_generated is not None and base_generated.shape == start.shape,
    )
    check("log_prob is per-sample", log_prob.shape == (8,), str(tuple(log_prob.shape)))
    check("log_prob carries grad", log_prob.requires_grad)

    (-(torch.randn(8) * log_prob).mean()).backward()
    policy_grad = policy.conv.weight.grad
    check(
        "policy grads flow, base stays untouched",
        policy_grad is not None
        and policy_grad.abs().sum() > 0
        and base.conv.weight.grad is None,
    )

    policy.zero_grad()
    _, _, subsampled = rollout_segment(
        policy,
        None,
        scheduler,
        start,
        indices,
        grid,
        embeddings,
        added_cond,
        False,
        {3, 7},
    )
    check("subsampled rollout is per-sample", subsampled.shape == (8,))
    check("subsampled rollout carries grad", subsampled.requires_grad)

    torch.manual_seed(11)
    first = rollout_segment(
        policy,
        base,
        scheduler,
        start,
        indices,
        grid,
        embeddings,
        added_cond,
        True,
        set(indices),
    )
    torch.manual_seed(11)
    second = rollout_segment(
        policy,
        base,
        scheduler,
        start,
        indices,
        grid,
        embeddings,
        added_cond,
        True,
        set(indices),
    )
    check(
        "rollout is deterministic given the seed (shared-noise wiring intact)",
        torch.allclose(first[0], second[0])
        and first[1] is not None
        and second[1] is not None
        and torch.allclose(first[1], second[1]),
    )

    late = rollout_segment(
        policy,
        None,
        scheduler,
        start,
        list(range(20, 30)),
        grid,
        embeddings,
        added_cond,
        False,
        {i for i in range(20, 30) if i < 29},
    )
    check("late segment runs (index 29 has no t_prev)", late[0].shape == start.shape)

    mu, variance = _ddpm_posterior(
        torch.randn(2, 4, 8, 8),
        torch.tensor([1, 1]),
        torch.tensor([-1, -1]),
        torch.randn(2, 4, 8, 8),
        scheduler,
    )
    check(
        "terminal t_prev=-1 handled",
        bool(torch.isfinite(mu).all()) and bool((variance > 0).all()),
    )


def test_group_advantage():
    print("\n--- RL: group-relative advantage ---")
    reward = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0, 101.0, 102.0, 103.0])
    grouped = reward.view(2, 4)
    advantage = (
        (grouped - grouped.mean(dim=1, keepdim=True))
        / (grouped.std(dim=1, keepdim=True) + 1e-6)
    ).flatten()

    check(
        "reference difficulty cancels inside the group",
        torch.allclose(advantage[:4], advantage[4:], atol=1e-4),
        f"easy={[round(v, 3) for v in advantage[:4].tolist()]} "
        f"hard={[round(v, 3) for v in advantage[4:].tolist()]}",
    )
    check(
        "advantage is zero-mean per group",
        abs(advantage.view(2, 4).mean().item()) < 1e-6,
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

    def app_pyramid_latents(shape, generators, device, dtype, lf=1.0, eps=1e-8):
        levels, decay = 8, 0.66
        batch, channels, height, width = shape
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
                for level in range(1, max(levels, 1)):
                    stride = 2**level
                    low_h, low_w = height // stride, width // stride
                    if low_h < 1 or low_w < 1:
                        break
                    low = torch.randn(
                        (1, channels, low_h, low_w),
                        generator=generator,
                        device=device,
                        dtype=torch.float32,
                    )
                    noise = noise + torch.nn.functional.interpolate(
                        low, size=(height, width), mode="bilinear", align_corners=False
                    ) * (decay**level * lf)
            rms = noise.pow(2).flatten(1).mean(1).sqrt().view(-1, 1, 1, 1)
            samples.append(noise / (rms + eps))
        return torch.cat(samples).to(dtype)

    shape = (2, 4, 64, 48)
    matches = True
    for lf in (0.0, 0.5, 1.0):
        mine = pyramid_latents(
            shape,
            [torch.Generator().manual_seed(7 + i) for i in range(shape[0])],
            "cpu",
            torch.float32,
            lf=lf,
        )
        theirs = app_pyramid_latents(
            shape,
            [torch.Generator().manual_seed(7 + i) for i in range(shape[0])],
            "cpu",
            torch.float32,
            lf=lf,
        )
        matches = matches and bool(torch.equal(mine, theirs))

    check("sampler.pyramid_latents matches the app.py copy across lf", matches)

    latents = pyramid_latents(
        shape,
        [torch.Generator().manual_seed(3 + i) for i in range(shape[0])],
        "cpu",
        torch.float32,
        lf=1.0,
    )
    rms = latents.pow(2).flatten(1).mean(1).sqrt()
    check(
        "pyramid_latents is unit RMS per sample",
        bool((rms - 1.0).abs().max().item() < 1e-4),
        f"max deviation {float((rms - 1.0).abs().max()):.2e}",
    )

    flat = pyramid_latents(
        shape,
        [torch.Generator().manual_seed(11 + i) for i in range(shape[0])],
        "cpu",
        torch.float32,
        lf=0.0,
    )
    dc_pyramid = latents.mean(dim=(2, 3)).pow(2).mean().item()
    dc_flat = flat.mean(dim=(2, 3)).pow(2).mean().item()
    check(
        "lf=1 carries more DC energy than lf=0",
        dc_pyramid > 5.0 * dc_flat,
        f"pyramid {dc_pyramid:.5f} vs flat {dc_flat:.5f}",
    )


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


class _DummyVAE:
    dtype = torch.float32
    config = SimpleNamespace(scaling_factor=1.0)

    def decode(self, latents):
        return SimpleNamespace(sample=latents.expand(-1, 3, -1, -1))


def test_straight_through_clamp():
    print("\n--- chain RL: straight-through clamp on decode_for_clip ---")
    vae = _DummyVAE()

    saturated = torch.full((1, 1, 8, 8), 3.0, requires_grad=True)
    decode_for_clip(vae, saturated).sum().backward()
    hard_grad = saturated.grad.abs().sum().item()

    saturated_st = torch.full((1, 1, 8, 8), 3.0, requires_grad=True)
    decode_for_clip(vae, saturated_st, straight_through=True).sum().backward()
    st_grad = saturated_st.grad.abs().sum().item()

    check("hard clamp kills gradient where the decoder saturates", hard_grad == 0.0)
    check("straight-through preserves it", st_grad > 0.0, f"{st_grad:.3f}")

    inside = torch.zeros(1, 1, 8, 8)
    a = decode_for_clip(vae, inside)
    b = decode_for_clip(vae, inside, straight_through=True)
    check("forward value is unchanged inside the clamp range", torch.equal(a, b))

    over = torch.full((1, 1, 8, 8), 3.0)
    check(
        "forward value is still clamped outside it",
        torch.equal(
            decode_for_clip(vae, over),
            decode_for_clip(vae, over, straight_through=True),
        ),
    )


def test_normalize_grad():
    print("\n--- chain RL: gradient normalizer ---")
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
    print("\n--- chain RL: adapter key rewrite ---")
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
    shape = (128, 4, 128, 128)

    inference = pyramid_latents(
        shape,
        [torch.Generator().manual_seed(100 + i) for i in range(shape[0])],
        "cpu",
        torch.float32,
        lf=1.0,
    )
    training = pyramid_noise(
        torch.zeros(shape), levels=LATENT_LF_LEVELS, decay=LATENT_LF_DECAY
    )

    dc_inference = inference.mean(dim=(2, 3)).var().item()
    dc_training = training.mean(dim=(2, 3)).var().item()

    check(
        "inference latents carry the same DC energy the trainer taught",
        0.75 < dc_inference / dc_training < 1.35,
        f"inference {dc_inference:.5f} vs training {dc_training:.5f} "
        f"({dc_inference / dc_training:.2f}x)",
    )


def test_dual_encoder_encoding():
    print("\n--- SDXL prompt encoding ---")
    torch.manual_seed(0)
    tokenizers = [_DummyTokenizer(49407), _DummyTokenizer(0)]
    encoders = [_DummyTextEncoder(768), _DummyTextEncoder(1280, projection=1280)]

    prompts = ["a girl with long pink hair", "a quiet room"]
    embeddings, pooled, content = encode_prompt(prompts, encoders, tokenizers, "cpu")

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

    words = [f"word{i}" for i in range(75)]
    short = " ".join(words)
    long = " ".join(words + ["extra"] * 5)
    _, pooled_short, _ = encode_prompt([short], encoders, tokenizers, "cpu")
    _, pooled_long, _ = encode_prompt([long], encoders, tokenizers, "cpu")
    check(
        "pooled comes from the first chunk only",
        torch.allclose(pooled_short, pooled_long, atol=1e-5),
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
    "LATENT_LF_LEVELS": LATENT_LF_LEVELS,
    "LATENT_LF_DECAY": LATENT_LF_DECAY,
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

    test_pyramid_noise()
    test_noise_prior_match()
    test_dual_encoder_encoding()
    test_rms_split()
    test_size_conditioning()
    test_min_snr(scheduler)
    test_loss_weighting(scheduler)
    test_bucket_sampler()
    test_conditioning_augmentation()
    test_caption_subsetting()
    test_shuffle_buffer_size()
    test_mask_normalization()
    test_x0_reward(scheduler)
    test_rollout(scheduler)
    test_group_advantage()
    test_sampler_parity()
    test_app_constant_parity()
    test_eval_metrics()
    test_straight_through_clamp()
    test_normalize_grad()
    test_chain_lora_keys()

    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {_failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
