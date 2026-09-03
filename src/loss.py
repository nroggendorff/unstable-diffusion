import torch


def min_snr_weight(
    alphas_cumprod: torch.Tensor, t: torch.Tensor, gamma: float
) -> torch.Tensor:
    alpha_bar = alphas_cumprod.to(t.device)[t.long()].float()
    snr = alpha_bar / (1.0 - alpha_bar)
    weight = snr.clamp(max=gamma) / snr
    return weight / weight.mean().clamp(min=1e-8)


def compute_diffusion_loss(
    unet,
    noisy_latents,
    t,
    text_emb,
    noise,
    added_cond_kwargs=None,
    noise_scale=None,
    mask=None,
    bg_weight=0.25,
    alphas_cumprod=None,
    snr_gamma=0.0,
):
    target = noise * noise_scale if noise_scale is not None else noise

    with torch.amp.autocast("cuda", dtype=torch.float16):
        pred = unet(
            noisy_latents,
            t,
            encoder_hidden_states=text_emb,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

    residual = (pred.float() - target.float()).pow(2)

    weight = None

    if mask is not None:
        weight = bg_weight + mask.float().clamp(0.0, 1.0)
        weight = weight / weight.flatten(1).mean(1).clamp(min=1e-8).view(-1, 1, 1, 1)

    if snr_gamma > 0.0 and alphas_cumprod is not None:
        snr = min_snr_weight(alphas_cumprod, t, snr_gamma).view(-1, 1, 1, 1)
        weight = snr if weight is None else weight * snr

    if weight is None:
        return residual.mean()

    return (residual * weight).mean()
