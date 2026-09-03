import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .encoder import CLIPVisionEncoder, compute_perceptual_discrepancy
from .encoding import decode_for_clip
from .cache import stack_added_cond
from .model import DEVICE

_CLIP_MODEL = "openai/clip-vit-base-patch32"

X0_CLAMP = 4.0


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
    x0 = x0.clamp(-X0_CLAMP, X0_CLAMP)

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
    ref_feats: list[torch.Tensor],
    base_latents: torch.Tensor | None,
    vae,
    clip_encoder: CLIPVisionEncoder,
    grounding_weight: float,
) -> torch.Tensor:
    with torch.no_grad():
        gen_feats = clip_encoder.extract_features(decode_for_clip(vae, gen_latents))

        discrepancy = compute_perceptual_discrepancy(gen_feats, ref_feats)
        perceptual_reward = -discrepancy.flatten(1).mean(1)

        if base_latents is not None and grounding_weight > 0.0:
            base_feats = clip_encoder.extract_features(
                decode_for_clip(vae, base_latents)
            )

            base_last = base_feats[-1].flatten(1)
            grounding_reward = F.cosine_similarity(
                gen_feats[-1].flatten(1) - base_last,
                ref_feats[-1].flatten(1) - base_last,
                dim=1,
                eps=1e-6,
            )
        else:
            grounding_reward = gen_latents.new_zeros(gen_latents.shape[0])

    return perceptual_reward + grounding_weight * grounding_reward


def rollout_segment(
    unet,
    base_unet,
    scheduler,
    noisy_start: torch.Tensor,
    t_indices: list[int],
    timesteps: torch.Tensor,
    text_emb: torch.Tensor,
    added_cond_kwargs: dict,
    run_base: bool,
    grad_indices: set,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    x_t = noisy_start.detach().clone()
    log_prob = noisy_start.new_zeros(noisy_start.shape[0])

    base_x_t = noisy_start.detach().clone() if run_base else None
    last_index = timesteps.shape[0] - 1
    batch = x_t.shape[0]

    x0_pred = x_t
    base_x0_pred = base_x_t

    for t_idx in t_indices:
        t = timesteps[t_idx].unsqueeze(0).expand(batch).to(x_t.device)
        if t_idx < last_index:
            t_prev = timesteps[t_idx + 1].unsqueeze(0).expand(batch).to(x_t.device)
        else:
            t_prev = torch.full_like(t, -1)

        needs_grad = t_idx in grad_indices

        with torch.set_grad_enabled(needs_grad):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                noise_pred = unet(
                    x_t.to(dtype=torch.float16),
                    t,
                    encoder_hidden_states=text_emb,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample.float()

            mu, variance = _ddpm_posterior(x_t, t, t_prev, noise_pred, scheduler)

        z = torch.randn_like(x_t)
        x_t_next = (mu.detach() + variance.sqrt() * z).detach()

        if needs_grad:
            log_prob = log_prob + (
                -((x_t_next - mu).pow(2) / (2.0 * variance)).flatten(1).sum(1)
            )

        x0_pred = scheduler.predict_x0(x_t, noise_pred.detach(), t).clamp(
            -X0_CLAMP, X0_CLAMP
        )
        x_t = x_t_next

        if run_base and base_x_t is not None:
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    base_pred = base_unet(
                        base_x_t.to(dtype=torch.float16),
                        t,
                        encoder_hidden_states=text_emb,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample.float()

                base_mu, base_var = _ddpm_posterior(
                    base_x_t, t, t_prev, base_pred, scheduler
                )
                base_x0_pred = scheduler.predict_x0(base_x_t, base_pred, t).clamp(
                    -X0_CLAMP, X0_CLAMP
                )
                base_x_t = (base_mu + base_var.sqrt() * z).detach()

    return x0_pred, base_x0_pred, log_prob


def rl_segment(
    segment_name: str,
    unet,
    base_unet,
    t_indices,
    timesteps: torch.Tensor,
    scheduler,
    vae,
    sampler,
    cfg,
) -> object:
    if cfg.rl_steps <= 0:
        return unet

    print(f"\nRL phase: {segment_name}")

    run_base = cfg.rl_grounding_weight > 0.0 and base_unet is not None
    if run_base:
        base_unet.to(DEVICE)

    unet.train()
    clip_encoder = CLIPVisionEncoder(model_name=_CLIP_MODEL).to(DEVICE)

    params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.rl_lr)

    t_indices_asc = sorted(list(t_indices))
    t_start_idx = t_indices_asc[0]
    last_index = timesteps.shape[0] - 1
    eligible = [i for i in t_indices_asc if i < last_index]

    group = cfg.rl_group
    subsample = cfg.rl_logprob_subsample

    for step in (pbar := tqdm(range(cfg.rl_steps), desc=f"RL {segment_name}")):
        optimizer.zero_grad(set_to_none=True)

        items = sampler.draw(cfg.rl_refs)

        ref_latents = torch.cat([x["latents"] for x in items]).float().to(DEVICE)
        text_emb = torch.cat([x["text_emb"] for x in items]).to(
            DEVICE, dtype=torch.float16
        )
        pooled, time_ids = stack_added_cond(items, DEVICE)

        with torch.no_grad():
            ref_feats = [
                f.repeat_interleave(group, dim=0)
                for f in clip_encoder.extract_features(
                    decode_for_clip(vae, ref_latents)
                )
            ]

        ref_repeated = ref_latents.repeat_interleave(group, dim=0)
        emb_repeated = text_emb.repeat_interleave(group, dim=0)
        added_cond_kwargs = {
            "text_embeds": pooled.repeat_interleave(group, dim=0),
            "time_ids": time_ids.repeat_interleave(group, dim=0),
        }
        rollout_batch = ref_repeated.shape[0]

        t_start = timesteps[t_start_idx].unsqueeze(0).expand(rollout_batch).to(DEVICE)

        with torch.no_grad():
            noisy_start = scheduler.add_noise(
                ref_repeated, torch.randn_like(ref_repeated), t_start
            )

        if 0 < subsample < len(eligible):
            grad_indices = set(random.sample(eligible, subsample))
            logprob_scale = len(eligible) / subsample
        else:
            grad_indices = set(eligible)
            logprob_scale = 1.0

        gen_latents, base_latents, log_prob = rollout_segment(
            unet=unet,
            base_unet=base_unet if run_base else None,
            scheduler=scheduler,
            noisy_start=noisy_start,
            t_indices=t_indices_asc,
            timesteps=timesteps,
            text_emb=emb_repeated,
            added_cond_kwargs=added_cond_kwargs,
            run_base=run_base,
            grad_indices=grad_indices,
        )

        reward = compute_reward(
            gen_latents=gen_latents,
            ref_feats=ref_feats,
            base_latents=base_latents,
            vae=vae,
            clip_encoder=clip_encoder,
            grounding_weight=cfg.rl_grounding_weight,
        )

        grouped = reward.view(cfg.rl_refs, group)
        advantage = (
            (grouped - grouped.mean(dim=1, keepdim=True))
            / (grouped.std(dim=1, keepdim=True) + 1e-6)
        ).flatten()

        rl_loss = -(advantage.detach() * log_prob * logprob_scale).mean()
        rl_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
        optimizer.step()

        pbar.set_postfix(
            reward=f"{reward.mean().item():.4f}",
            spread=f"{grouped.std(dim=1).mean().item():.4f}",
            gnorm=f"{grad_norm.item():.3e}",
        )

        del gen_latents, base_latents, log_prob, ref_feats, noisy_start

        if step % 20 == 0:
            torch.cuda.empty_cache()

    if run_base:
        base_unet.to("cpu")

    del clip_encoder, optimizer
    torch.cuda.empty_cache()

    return unet
