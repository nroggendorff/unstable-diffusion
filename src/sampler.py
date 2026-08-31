import contextlib

import torch

from .encoding import rms_scaled_noise
from .model import EARLY_SEG, MID_SEG, NUM_INFERENCE_STEPS

ALL_SEGMENTS = ["early", "mid", "late"]

_BOUNDARIES = [EARLY_SEG, EARLY_SEG + MID_SEG]
_BLEND_HALF = 2
_MAX_RAW_GUIDANCE = 30.0


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
    negative: torch.Tensor,
    seeds: list[int],
    use_lora: bool = True,
    strength: float = 0.8,
    guidance: float = 7.0,
    width: int = 384,
    height: int = 512,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    alpha: float = 0.4,
    sigma: float = 0.2,
    cutover: float = 0.8,
    sigma_cutover: float = 0.8,
    offset: float = 0.0,
    gamma: float = 1.0,
):
    batch = len(seeds)
    last_step = num_inference_steps - 1

    positive = positive.expand(batch, -1, -1)
    negative = negative.expand(batch, -1, -1)

    def raw_guidance(step_index):
        if alpha == 0.0 or progress(step_index, num_inference_steps) <= cutover:
            return guidance
        return max(1.01, min(_MAX_RAW_GUIDANCE, (guidance - alpha) / (1.0 - alpha)))

    if use_lora:
        pipe.enable_lora()
        pipe.set_adapters(ALL_SEGMENTS, adapter_weights(0, strength))
    else:
        pipe.disable_lora()

    embed_noise = rms_scaled_noise(
        positive,
        sigma,
        torch.Generator(device=positive.device).manual_seed(seeds[0] ^ 0x5EED),
    )

    def anchor_at(step_index):
        if progress(step_index, num_inference_steps) <= cutover:
            return negative
        anchor = torch.lerp(negative, positive, alpha)
        if progress(step_index, num_inference_steps) > sigma_cutover:
            return anchor + embed_noise
        return anchor

    def callback(p, step_index, timestep, callback_kwargs):  # noqa: ARG001
        if use_lora:
            if step_index + 1 == last_step:
                p.set_adapters(ALL_SEGMENTS, [0.0, 0.0, 0.0])
            else:
                p.set_adapters(ALL_SEGMENTS, adapter_weights(step_index + 1, strength))

        p._guidance_scale = raw_guidance(min(step_index + 1, last_step))

        anchor = anchor_at(min(step_index + 1, last_step))
        callback_kwargs["prompt_embeds"] = torch.cat([anchor, positive])
        return callback_kwargs

    try:
        with lie_about_noise(pipe, offset, gamma):
            result = pipe(
                prompt_embeds=positive,
                negative_prompt_embeds=anchor_at(0),
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=raw_guidance(0),
                generator=[
                    torch.Generator(device="cuda").manual_seed(s) for s in seeds
                ],
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
            )
        return result.images
    finally:
        pipe.disable_lora()
        torch.cuda.empty_cache()
