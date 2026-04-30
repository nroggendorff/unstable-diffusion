import torch


def compute_diffusion_loss(
    unet,
    noisy_latents,
    t,
    text_emb,
    added_cond_kwargs,
    noise,
    noise_scale=None,
    mask=None,
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

    if mask is not None:
        weight = 1.0 + mask.float().clamp(0.0, 1.0)
        return (residual * weight).mean()

    return residual.mean()
