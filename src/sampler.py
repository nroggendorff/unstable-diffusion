import contextlib

import torch
import torch.nn.functional as F

from .encoding import rms_scaled_noise, HIDDEN_SPLIT
from .model import EARLY_SEG, MID_SEG, NUM_INFERENCE_STEPS

ALL_SEGMENTS = ["early", "mid", "late"]

_BOUNDARIES = [EARLY_SEG, EARLY_SEG + MID_SEG]
_BLEND_HALF = 2
_MAX_RAW_GUIDANCE = 30.0

LATENT_LF_LEVELS = 8
LATENT_LF_DECAY = 0.66


def pyramid_latents(
    shape: tuple[int, int, int, int],
    generators: list[torch.Generator],
    device,
    dtype,
    lf: float = 1.0,
    levels: int = LATENT_LF_LEVELS,
    decay: float = LATENT_LF_DECAY,
    eps: float = 1e-8,
) -> torch.Tensor:
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
                noise = noise + F.interpolate(
                    low, size=(height, width), mode="bilinear", align_corners=False
                ) * (decay**level * lf)

        rms = noise.pow(2).flatten(1).mean(1).sqrt().view(-1, 1, 1, 1)
        samples.append(noise / (rms + eps))

    return torch.cat(samples).to(dtype)


def adapter_weights(step_index: int, strength: float = 1.0) -> list[float]:
    for i, boundary in enumerate(_BOUNDARIES):
        dist = step_index - boundary
        if abs(dist) <= _BLEND_HALF:
            t = (dist + _BLEND_HALF) / (2 * _BLEND_HALF)
            weights = [0.0, 0.0, 0.0]
            weights[i] = (1.0 - t) * strength
            weights[i + 1] = t * strength
            return weights

    weights = [0.0, 0.0, 0.0]
    if step_index < _BOUNDARIES[0]:
        weights[0] = strength
    elif step_index < _BOUNDARIES[1]:
        weights[1] = strength
    else:
        weights[2] = strength
    return weights


def progress(step_index: int, num_inference_steps: int) -> float:
    return 1.0 - step_index / max(num_inference_steps - 1, 1)


@contextlib.contextmanager
def lie_about_noise(pipe, offset: float, gamma: float):
    if offset == 0.0 and gamma == 1.0:
        yield
        return

    original = pipe.unet.forward

    def forward(sample, timestep, *args, **kwargs):
        schedule = pipe.scheduler.timesteps
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
        return original(
            sample,
            torch.lerp(
                schedule[int((1.0 - told) * last)].float(),
                schedule[min(int((1.0 - told) * last) + 1, last)].float(),
                (1.0 - told) * last % 1.0,
            ).to(timestep.dtype if torch.is_tensor(timestep) else torch.float32),
            *args,
            **kwargs,
        )

    pipe.unet.forward = forward
    try:
        yield
    finally:
        pipe.unet.forward = original


def anchor_generate(
    pipe,
    positive: torch.Tensor,
    positive_pooled: torch.Tensor,
    negative: torch.Tensor,
    negative_pooled: torch.Tensor,
    seeds: list[int],
    use_lora: bool = True,
    strength: float = 0.8,
    guidance: float = 7.0,
    width: int = 832,
    height: int = 1216,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    alpha: float = 0.4,
    sigma: float = 0.2,
    cutover: float = 0.8,
    sigma_cutover: float = 0.8,
    offset: float = 0.0,
    gamma: float = 1.0,
    latent_lf: float = 1.0,
):
    batch = len(seeds)
    last_step = num_inference_steps - 1

    positive = positive.expand(batch, -1, -1)
    negative = negative.expand(batch, -1, -1)
    positive_pooled = positive_pooled.expand(batch, -1)
    negative_pooled = negative_pooled.expand(batch, -1)

    def raw_guidance(step_index):
        if alpha == 0.0 or progress(step_index, num_inference_steps) <= cutover:
            return guidance
        return max(1.01, min(_MAX_RAW_GUIDANCE, (guidance - alpha) / (1.0 - alpha)))

    if use_lora:
        pipe.enable_lora()
        pipe.set_adapters(ALL_SEGMENTS, adapter_weights(0, strength))
    else:
        pipe.disable_lora()

    jitter_generator = torch.Generator(device=positive.device).manual_seed(
        seeds[0] ^ 0x5EED
    )
    embed_noise = rms_scaled_noise(
        positive, sigma, jitter_generator, split=HIDDEN_SPLIT
    )
    pooled_noise = rms_scaled_noise(positive_pooled, sigma, jitter_generator)

    def anchor_at(step_index):
        step_progress = progress(step_index, num_inference_steps)
        if step_progress <= cutover:
            return negative, negative_pooled

        anchor = torch.lerp(negative, positive, alpha)
        anchor_pooled = torch.lerp(negative_pooled, positive_pooled, alpha)
        if step_progress > sigma_cutover:
            return anchor + embed_noise, anchor_pooled + pooled_noise
        return anchor, anchor_pooled

    def callback(p, step_index, timestep, callback_kwargs):  # noqa: ARG001
        if use_lora:
            if step_index + 1 == last_step:
                p.set_adapters(ALL_SEGMENTS, [0.0, 0.0, 0.0])
            else:
                p.set_adapters(ALL_SEGMENTS, adapter_weights(step_index + 1, strength))

        p._guidance_scale = raw_guidance(min(step_index + 1, last_step))

        anchor, anchor_pooled = anchor_at(min(step_index + 1, last_step))
        callback_kwargs["prompt_embeds"] = torch.cat([anchor, positive])
        callback_kwargs["add_text_embeds"] = torch.cat([anchor_pooled, positive_pooled])
        return callback_kwargs

    generators = [torch.Generator(device="cuda").manual_seed(s) for s in seeds]
    scale = pipe.vae_scale_factor
    latents = pyramid_latents(
        (batch, pipe.unet.config.in_channels, height // scale, width // scale),
        generators,
        pipe.unet.device,
        pipe.unet.dtype,
        lf=latent_lf,
    )

    start_anchor, start_anchor_pooled = anchor_at(0)

    try:
        with lie_about_noise(pipe, offset, gamma):
            result = pipe(
                prompt_embeds=positive,
                pooled_prompt_embeds=positive_pooled,
                negative_prompt_embeds=start_anchor,
                negative_pooled_prompt_embeds=start_anchor_pooled,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=raw_guidance(0),
                generator=generators,
                latents=latents,
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=[
                    "latents",
                    "prompt_embeds",
                    "add_text_embeds",
                ],
            )
        return result.images
    finally:
        pipe.disable_lora()
        torch.cuda.empty_cache()
