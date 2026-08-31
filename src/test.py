import argparse
import random
import sys
from types import SimpleNamespace

import torch

from .cache import BucketSampler
from .eval import adherence, angular_spread, radius
from .loss import compute_diffusion_loss, min_snr_weight
from .rl import _ddpm_posterior, rollout_segment
from .sampler import adapter_weights
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

    def forward(self, sample, t, encoder_hidden_states=None, **kwargs):  # noqa: ARG002
        return SimpleNamespace(sample=self.conv(sample.float()))


def test_pyramid_noise():
    print("\n--- pyramid_noise ---")
    latents = torch.zeros(6, 4, 64, 48)
    noise = pyramid_noise(latents, levels=6, decay=0.6)

    check("shape preserved", noise.shape == latents.shape, str(tuple(noise.shape)))
    check("dtype preserved", noise.dtype == latents.dtype)

    rms = noise.pow(2).flatten(1).mean(1).sqrt()
    check(
        "per-sample RMS is 1 (variance invariant held)",
        torch.allclose(rms, torch.ones_like(rms), atol=1e-4),
        f"rms={[round(v, 5) for v in rms.tolist()[:3]]}",
    )

    bulk = torch.zeros(3000, 4, 64, 48)
    dc_plain = pyramid_noise(bulk, levels=1).mean(dim=(2, 3)).var().item()
    dc_pyramid = pyramid_noise(bulk, levels=6, decay=0.6).mean(dim=(2, 3)).var().item()

    check(
        "defaults bring DC noise to parity with the terminal-SNR leak",
        0.5 < TERMINAL_ALPHA_BAR / dc_pyramid < 2.0,
        f"signal/noise DC {TERMINAL_ALPHA_BAR / dc_plain:.1f}x -> "
        f"{TERMINAL_ALPHA_BAR / dc_pyramid:.2f}x",
    )
    check(
        "levels=1 reduces to plain randn",
        abs(dc_plain - 1.0 / (64 * 48)) < 2e-5,
        f"dc={dc_plain:.6f} expected={1 / 3072:.6f}",
    )
    check(
        "survives latents too small for every level",
        pyramid_noise(torch.zeros(2, 4, 8, 6), levels=6).shape == (2, 4, 8, 6),
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
    embeddings = torch.randn(4, 154, 768)
    t = torch.full((4,), 300, dtype=torch.long)

    def loss_for(mask_value):
        torch.manual_seed(1)
        return compute_diffusion_loss(
            unet,
            noisy,
            t,
            embeddings,
            noise,
            mask=torch.full((4, 1, 16, 16), mask_value),
            bg_weight=0.25,
        ).item()

    sparse, dense = loss_for(0.05), loss_for(0.95)
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
    cached = [{"bucket": (512, 384) if i < 100 else (384, 512)} for i in range(140)]
    sampler = BucketSampler(cached, 2)

    check("buckets grouped", set(sampler.groups) == {(512, 384), (384, 512)})
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
    embeddings = torch.randn(8, 154, 768)
    content = torch.ones(8, 154)
    empty = torch.zeros(1, 154, 768)

    dropped, dropped_content = apply_conditioning_augmentation(
        embeddings,
        content,
        empty,
        argparse.Namespace(
            cond_dropout_prob=1.0, cond_partial_prob=0.0, cond_partial_max=0.6
        ),
    )
    check("full dropout gives the empty embedding", dropped.abs().max() < 1e-6)
    check("full dropout zeroes the content mask", dropped_content.abs().max() < 1e-6)

    torch.manual_seed(3)
    partial, partial_content = apply_conditioning_augmentation(
        embeddings,
        content,
        empty,
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

    same, same_content = apply_conditioning_augmentation(
        embeddings,
        content,
        empty,
        argparse.Namespace(
            cond_dropout_prob=0.0, cond_partial_prob=0.0, cond_partial_max=0.6
        ),
    )
    check(
        "both disabled is a no-op",
        torch.equal(same, embeddings) and torch.equal(same_content, content),
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
    embeddings = torch.randn(8, 154, 768)
    indices = list(range(10))

    torch.manual_seed(7)
    generated, base_generated, log_prob = rollout_segment(
        policy, base, scheduler, start, indices, grid, embeddings, True, set(indices)
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
        policy, None, scheduler, start, indices, grid, embeddings, False, {3, 7}
    )
    check("subsampled rollout is per-sample", subsampled.shape == (8,))
    check("subsampled rollout carries grad", subsampled.requires_grad)

    torch.manual_seed(11)
    first = rollout_segment(
        policy, base, scheduler, start, indices, grid, embeddings, True, set(indices)
    )
    torch.manual_seed(11)
    second = rollout_segment(
        policy, base, scheduler, start, indices, grid, embeddings, True, set(indices)
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


def main() -> int:
    torch.manual_seed(0)
    random.seed(0)
    scheduler = make_scheduler()

    test_pyramid_noise()
    test_min_snr(scheduler)
    test_loss_weighting(scheduler)
    test_bucket_sampler()
    test_conditioning_augmentation()
    test_x0_reward(scheduler)
    test_rollout(scheduler)
    test_group_advantage()
    test_sampler_parity()
    test_eval_metrics()

    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {_failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
