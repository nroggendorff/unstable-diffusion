import torch
import torch.nn.functional as F
from tqdm import tqdm

from .encoder import CLIPVisionEncoder, compute_perceptual_discrepancy
from .encoding import decode_for_clip
from .cache import sample_minibatch
from .model import DEVICE

_CLIP_MODEL = "openai/clip-vit-base-patch32"


def _ddpm_posterior(
    x_t: torch.Tensor,
    t: torch.Tensor,
    t_prev: torch.Tensor,
    noise_pred: torch.Tensor,
    scheduler,
) -> tuple[torch.Tensor, torch.Tensor]:
    alphas_cumprod = scheduler.alphas_cumprod.to(x_t.device)

    alpha_bar_t = alphas_cumprod[t.long()]
    prev_clamped = t_prev.clamp(min=0).long()
    alpha_bar_prev = torch.where(
        t_prev >= 0,
        alphas_cumprod[prev_clamped],
        torch.ones_like(alphas_cumprod[prev_clamped]),
    )

    alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)
    alpha_bar_prev = alpha_bar_prev.view(-1, 1, 1, 1)

    beta = 1.0 - alpha_bar_t / alpha_bar_prev

    x0 = (x_t - (1.0 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
    x0 = x0.clamp(-4.0, 4.0)

    coef_x0 = alpha_bar_prev.sqrt() * beta / (1.0 - alpha_bar_t)
    coef_xt = (
        (alpha_bar_t / alpha_bar_prev).sqrt()
        * (1.0 - alpha_bar_prev)
        / (1.0 - alpha_bar_t)
    )

    mu = coef_x0 * x0 + coef_xt * x_t
    variance = (beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)).clamp(min=1e-12)

    return mu, variance


def compute_reward(
    gen_latents: torch.Tensor,
    ref_latents: torch.Tensor,
    base_latents: torch.Tensor | None,
    vae,
    clip_encoder: CLIPVisionEncoder,
    grounding_weight: float,
) -> torch.Tensor:
    with torch.no_grad():
        gen_img = decode_for_clip(vae, gen_latents)
        ref_img = decode_for_clip(vae, ref_latents)

        gen_feats = clip_encoder.extract_features(gen_img)
        ref_feats = clip_encoder.extract_features(ref_img)

        discrepancy = compute_perceptual_discrepancy(gen_feats, ref_feats)
        perceptual_reward = -discrepancy.flatten(1).mean(1).mean()

        if base_latents is not None and grounding_weight > 0.0:
            base_img = decode_for_clip(vae, base_latents)
            base_feats = clip_encoder.extract_features(base_img)

            gen_last = gen_feats[-1].flatten(1)
            ref_last = ref_feats[-1].flatten(1)
            base_last = base_feats[-1].flatten(1)

            grounding_reward = F.cosine_similarity(
                gen_last - base_last, ref_last - base_last, dim=1, eps=1e-6
            ).mean()
        else:
            grounding_reward = gen_latents.new_tensor(0.0)

    return perceptual_reward + grounding_weight * grounding_reward


def rollout_segment(
    unet,
    base_unet,
    scheduler,
    noisy_start: torch.Tensor,
    t_indices: list[int],
    timesteps: torch.Tensor,
    text_emb: torch.Tensor,
    run_base: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    x_t = noisy_start.detach().clone()
    log_prob_sum = noisy_start.new_tensor(0.0)

    base_x_t = noisy_start.detach().clone() if run_base else None
    last_index = timesteps.shape[0] - 1
    batch = x_t.shape[0]

    for t_idx in sorted(t_indices):
        t = timesteps[t_idx].unsqueeze(0).expand(batch).to(x_t.device)
        if t_idx < last_index:
            t_prev = timesteps[t_idx + 1].unsqueeze(0).expand(batch).to(x_t.device)
        else:
            t_prev = torch.full_like(t, -1)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            noise_pred = unet(
                x_t.to(dtype=torch.float16),
                t,
                encoder_hidden_states=text_emb,
            ).sample.float()

        with torch.no_grad():
            mu_env, variance = _ddpm_posterior(
                x_t, t, t_prev, noise_pred.detach(), scheduler
            )
            x_t_next = mu_env + variance.sqrt() * torch.randn_like(x_t)

        if t_idx < last_index:
            mu_grad, _ = _ddpm_posterior(x_t, t, t_prev, noise_pred, scheduler)

            step_log_prob = (
                -((x_t_next.detach() - mu_grad).pow(2) / (2.0 * variance.detach()))
                .flatten(1)
                .sum(1)
                .mean()
            )
            log_prob_sum = log_prob_sum + step_log_prob

        x_t = x_t_next.detach()

        if run_base and base_x_t is not None:
            with torch.no_grad():
                base_pred = base_unet(
                    base_x_t.to(dtype=torch.float16),
                    t,
                    encoder_hidden_states=text_emb,
                ).sample.float()
                base_mu, base_var = _ddpm_posterior(
                    base_x_t, t, t_prev, base_pred, scheduler
                )
                base_x_t = (
                    base_mu + base_var.sqrt() * torch.randn_like(base_x_t)
                ).detach()

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
    groups: dict,
    cfg,
) -> object:
    if cfg.rl_steps <= 0:
        return unet

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

    t_indices_asc = sorted(list(t_indices))
    t_start_idx = t_indices_asc[0]

    baseline: float | None = None

    for step in (pbar := tqdm(range(cfg.rl_steps), desc=f"RL {segment_name}")):
        optimizer.zero_grad(set_to_none=True)

        items = sample_minibatch(cached, groups, cfg.mini_batch_size)

        ref_latents = torch.cat([x["latents"] for x in items]).float().to(DEVICE)
        text_emb = torch.cat([x["text_emb"] for x in items]).to(
            DEVICE, dtype=torch.float16
        )

        t_start = (
            timesteps[t_start_idx].unsqueeze(0).expand(ref_latents.shape[0]).to(DEVICE)
        )
        noise = torch.randn_like(ref_latents)

        with torch.no_grad():
            noisy_start = scheduler.add_noise(ref_latents, noise, t_start)

        gen_latents, base_latents, log_prob_sum = rollout_segment(
            unet=unet,
            base_unet=base_unet if run_base else None,
            scheduler=scheduler,
            noisy_start=noisy_start,
            t_indices=t_indices_asc,
            timesteps=timesteps,
            text_emb=text_emb,
            run_base=run_base,
        )

        reward = compute_reward(
            gen_latents=gen_latents,
            ref_latents=ref_latents,
            base_latents=base_latents,
            vae=vae,
            clip_encoder=clip_encoder,
            grounding_weight=cfg.rl_grounding_weight,
        )

        r = reward.item()
        advantage = 0.0 if baseline is None else r - baseline
        baseline = (
            r
            if baseline is None
            else cfg.rl_baseline_momentum * baseline
            + (1.0 - cfg.rl_baseline_momentum) * r
        )

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
