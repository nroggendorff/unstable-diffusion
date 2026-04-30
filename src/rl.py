import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .encoder import CLIPVisionEncoder, compute_perceptual_discrepancy
from .cache import make_time_ids
from .model import DEVICE

_CLIP_MODEL = "openai/clip-vit-base-patch32"


def _decode_to_clip(vae, latents: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        decoded = vae.decode(
            latents.to(dtype=vae.dtype) / vae.config.scaling_factor
        ).sample
        decoded = (decoded.float().clamp(-1, 1) + 1) / 2
        decoded = F.interpolate(
            decoded, size=(224, 224), mode="bilinear", align_corners=False
        )
    return decoded


def _mu_theta(
    x_t: torch.Tensor,
    t: torch.Tensor,
    noise_pred: torch.Tensor,
    scheduler,
) -> tuple[torch.Tensor, torch.Tensor]:
    alphas = scheduler.alphas.to(x_t.device)
    alphas_cumprod = scheduler.alphas_cumprod.to(x_t.device)

    alpha_t = alphas[t.long()].view(-1, 1, 1, 1)
    alpha_bar_t = alphas_cumprod[t.long()].view(-1, 1, 1, 1)
    beta_t = 1.0 - alpha_t

    coef = beta_t / (1.0 - alpha_bar_t).sqrt()
    mu = (x_t - coef * noise_pred) / alpha_t.sqrt()
    return mu, beta_t


def compute_reward(
    gen_latents: torch.Tensor,
    ref_latents: torch.Tensor,
    base_latents: torch.Tensor | None,
    vae,
    clip_encoder: CLIPVisionEncoder,
    grounding_weight: float,
    diversity_weight: float,
) -> torch.Tensor:
    gen_img = _decode_to_clip(vae, gen_latents)
    ref_img = _decode_to_clip(vae, ref_latents)

    with torch.no_grad():
        gen_feats = clip_encoder.extract_features(gen_img)
        ref_feats = clip_encoder.extract_features(ref_img)

    discrepancy = compute_perceptual_discrepancy(gen_feats, ref_feats)
    perceptual_reward = -discrepancy.flatten(1).mean(1).mean()

    if base_latents is not None and grounding_weight > 0.0:
        base_img = _decode_to_clip(vae, base_latents)
        with torch.no_grad():
            base_feats = clip_encoder.extract_features(base_img)

        gen_last = gen_feats[-1].flatten(1)
        ref_last = ref_feats[-1].flatten(1)
        base_last = base_feats[-1].flatten(1)

        lora_delta = gen_last - base_last
        target_delta = ref_last - base_last
        grounding_reward = F.cosine_similarity(
            lora_delta, target_delta, dim=1, eps=1e-6
        ).mean()
    else:
        grounding_reward = gen_latents.new_tensor(0.0)

    if gen_latents.shape[0] > 1 and diversity_weight > 0.0:
        gen_flat = F.normalize(gen_feats[-1].flatten(1), dim=1)
        sim = gen_flat @ gen_flat.T
        n = gen_flat.shape[0]
        upper = torch.triu(
            torch.ones(n, n, dtype=torch.bool, device=gen_flat.device), diagonal=1
        )
        diversity_reward = -sim[upper].mean()
    else:
        diversity_reward = gen_latents.new_tensor(0.0)

    return (
        perceptual_reward
        + grounding_weight * grounding_reward
        + diversity_weight * diversity_reward
    )


def rollout_segment(
    unet,
    base_unet,
    scheduler,
    noisy_start: torch.Tensor,
    t_indices: list[int],
    timesteps: torch.Tensor,
    text_emb: torch.Tensor,
    pooled: torch.Tensor,
    mini_batch_size: int,
    run_base: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    x_t = noisy_start.detach().clone()
    log_prob_sum = noisy_start.new_tensor(0.0)

    time_ids = make_time_ids(mini_batch_size, x_t.device)
    added_cond_kwargs = {"text_embeds": pooled, "time_ids": time_ids}

    base_x_t = noisy_start.detach().clone() if run_base else None

    for t_idx in sorted(t_indices, reverse=True):
        t = timesteps[t_idx].unsqueeze(0).expand(mini_batch_size).to(x_t.device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            noise_pred = unet(
                x_t.to(dtype=torch.float16),
                t,
                encoder_hidden_states=text_emb,
                added_cond_kwargs=added_cond_kwargs,
            ).sample.float()

        with torch.no_grad():
            mu_env, beta_t = _mu_theta(x_t, t, noise_pred.detach(), scheduler)
            eps = torch.randn_like(x_t)
            x_t_next = mu_env + beta_t.sqrt() * eps

        mu_grad, _ = _mu_theta(x_t, t, noise_pred, scheduler)
        step_log_prob = -(
            (x_t_next.detach() - mu_grad).pow(2) / (2.0 * beta_t.detach() + 1e-8)
        ).mean()
        log_prob_sum = log_prob_sum + step_log_prob

        x_t = x_t_next.detach()

        if run_base and base_x_t is not None:
            with torch.no_grad():
                base_pred = base_unet(
                    base_x_t.to(dtype=torch.float16),
                    t,
                    encoder_hidden_states=text_emb,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample.float()
                base_mu, base_beta_t = _mu_theta(base_x_t, t, base_pred, scheduler)
                base_eps = torch.randn_like(base_x_t)
                base_x_t = (base_mu + base_beta_t.sqrt() * base_eps).detach()

    return x_t, base_x_t, log_prob_sum


def rl_segment(
    segment_name: str,
    unet,
    base_unet,
    t_indices,
    timesteps: torch.Tensor,
    scheduler,
    vae,
    cached: list,
    cfg,
) -> object:
    print(f"\nRL phase: {segment_name}")

    run_base = cfg.rl_grounding_weight > 0.0

    if run_base:
        base_unet.to(DEVICE)

    unet.train()
    clip_encoder = CLIPVisionEncoder(model_name=_CLIP_MODEL).to(DEVICE)

    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=cfg.rl_lr,
    )

    t_indices_desc = sorted(list(t_indices), reverse=True)
    t_start_idx = t_indices_desc[0]

    baseline: float | None = None

    for step in (pbar := tqdm(range(cfg.rl_steps), desc=f"RL {segment_name}")):
        optimizer.zero_grad(set_to_none=True)

        idxs = list(range(len(cached)))
        random.shuffle(idxs)
        items = [cached[i] for i in idxs[: cfg.mini_batch_size]]

        ref_latents = torch.cat([x["latents"] for x in items]).float().to(DEVICE)
        text_emb = torch.cat([x["text_emb"] for x in items]).to(
            DEVICE, dtype=torch.float16
        )
        pooled = torch.cat([x["pooled_text_emb"] for x in items]).to(
            DEVICE, dtype=torch.float16
        )

        t_start = timesteps[t_start_idx].unsqueeze(0).expand(cfg.mini_batch_size)
        noise = torch.randn_like(ref_latents)

        with torch.no_grad():
            noisy_start = scheduler.add_noise(ref_latents, noise, t_start)

        gen_latents, base_latents, log_prob_sum = rollout_segment(
            unet=unet,
            base_unet=base_unet if run_base else None,
            scheduler=scheduler,
            noisy_start=noisy_start,
            t_indices=t_indices_desc,
            timesteps=timesteps,
            text_emb=text_emb,
            pooled=pooled,
            mini_batch_size=cfg.mini_batch_size,
            run_base=run_base,
        )

        with torch.no_grad():
            reward = compute_reward(
                gen_latents=gen_latents,
                ref_latents=ref_latents,
                base_latents=base_latents,
                vae=vae,
                clip_encoder=clip_encoder,
                grounding_weight=cfg.rl_grounding_weight,
                diversity_weight=cfg.rl_diversity_weight,
            )

        r = reward.item()
        if baseline is None:
            baseline = r
        else:
            baseline = (
                cfg.rl_baseline_momentum * baseline
                + (1.0 - cfg.rl_baseline_momentum) * r
            )

        advantage = r - baseline
        rl_loss = -(advantage * log_prob_sum)
        rl_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in unet.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        pbar.set_postfix(
            reward=f"{r:.4f}",
            baseline=f"{baseline:.4f}",
            adv=f"{advantage:.4f}",
        )

        if step % 20 == 0:
            torch.cuda.empty_cache()

    if run_base:
        base_unet.to("cpu")

    del clip_encoder, optimizer
    torch.cuda.empty_cache()

    return unet
