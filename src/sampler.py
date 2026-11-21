import contextlib
import math
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import EulerDiscreteScheduler

from .encoding import rms_scaled_noise, HIDDEN_SPLIT
from .model import EARLY_SEG, MID_SEG, NUM_INFERENCE_STEPS
from .scheduler import LATENT_MEAN, NOISE_LF_STRENGTH, TERMINAL_LEAK, lowfreq_field

ALL_SEGMENTS = ["early", "mid", "late"]

NEGATIVE_PROMPT = (
    "nsfw, explicit, questionable, sensitive, suggestive, underwear, panties, "
    "lingerie, cleavage, nude, ugly, low quality, lowres, worst quality, "
    "text, watermark, comic, speech bubble, 4koma, multiple views, english text, logo"
)

_BOUNDARIES = [EARLY_SEG, EARLY_SEG + MID_SEG]
_BLEND_HALF = 2
_MAX_RAW_GUIDANCE = 30.0


def prior_deviation(init_sigma: float, strength: float = NOISE_LF_STRENGTH) -> float:
    sigma = max(init_sigma**2 - 1.0, 0.0) ** 0.5
    return (1.0 + strength * (TERMINAL_LEAK * sigma) ** 2) ** 0.5


def prior_latents(
    shape: tuple[int, int, int, int],
    generators: list[torch.Generator],
    device,
    dtype,
    init_sigma: float,
    lf: float = 1.0,
    strength: float = NOISE_LF_STRENGTH,
) -> torch.Tensor:
    batch, channels, height, width = shape
    deviation = prior_deviation(init_sigma, strength)
    mean = torch.tensor(LATENT_MEAN[:channels], device=device).view(1, channels, 1, 1)

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
            field = lowfreq_field((1, channels, height, width), generator, device)
            noise = noise + lf * (mean + deviation * field) / init_sigma
        samples.append(noise)

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


def lied_timestep(
    schedule: torch.Tensor, step_index: int, offset: float, gamma: float
) -> torch.Tensor:
    last = len(schedule) - 1
    told = min(1.0, max(0.0, (1.0 - step_index / max(last, 1)) ** gamma + offset))
    position = (1.0 - told) * last
    low = int(position)
    return torch.lerp(
        schedule[low].float(),
        schedule[min(low + 1, last)].float(),
        position % 1.0,
    )


@contextlib.contextmanager
def lie_about_noise(pipe, offset: float, gamma: float):
    if offset == 0.0 and gamma == 1.0:
        yield
        return

    original = pipe.unet.forward

    def forward(sample, timestep, *args, **kwargs):
        schedule = pipe.scheduler.timesteps
        index = int((schedule - timestep).abs().argmin().item())
        return original(
            sample,
            lied_timestep(schedule, index, offset, gamma).to(
                timestep.dtype if torch.is_tensor(timestep) else torch.float32
            ),
            *args,
            **kwargs,
        )

    pipe.unet.forward = forward
    try:
        yield
    finally:
        pipe.unet.forward = original


def anchor_plan(
    positive: torch.Tensor,
    positive_pooled: torch.Tensor,
    negative: torch.Tensor,
    negative_pooled: torch.Tensor,
    num_inference_steps: int,
    guidance: float,
    alpha: float,
    sigma: float,
    cutover: float,
    sigma_cutover: float,
    generator: torch.Generator | None = None,
):
    embed_noise = rms_scaled_noise(positive, sigma, generator, split=HIDDEN_SPLIT)
    pooled_noise = rms_scaled_noise(positive_pooled, sigma, generator)
    anchor = torch.lerp(negative, positive, alpha)
    anchor_pooled = torch.lerp(negative_pooled, positive_pooled, alpha)

    def raw_guidance(step_index: int) -> float:
        if alpha == 0.0 or progress(step_index, num_inference_steps) <= cutover:
            return guidance
        return max(1.01, min(_MAX_RAW_GUIDANCE, (guidance - alpha) / (1.0 - alpha)))

    def anchor_at(step_index: int):
        step_progress = progress(step_index, num_inference_steps)
        if step_progress <= cutover:
            return negative, negative_pooled
        if step_progress > sigma_cutover:
            return anchor + embed_noise, anchor_pooled + pooled_noise
        return anchor, anchor_pooled

    return anchor_at, raw_guidance


COND_NOISE_SALT = 0xC0DE
COND_LO = 0.6
COND_HI = 0.9


def early_weight(
    step_index: int, num_inference_steps: int, lo: float, hi: float
) -> float:
    step_progress = progress(step_index, num_inference_steps)
    if step_progress >= hi:
        return 1.0
    if step_progress <= lo:
        return 0.0
    return (step_progress - lo) / (hi - lo)


def cond_noise_plan(
    positive: torch.Tensor,
    positive_pooled: torch.Tensor,
    seeds: list[int],
    num_inference_steps: int,
    sigma: float,
    lo: float,
    hi: float,
):
    if sigma <= 0.0:
        return lambda step_index: (positive, positive_pooled)  # noqa: ARG005

    noise, pooled_noise = [], []
    for row, seed in enumerate(seeds):
        generator = torch.Generator(device=positive.device).manual_seed(
            seed ^ COND_NOISE_SALT
        )
        noise.append(
            rms_scaled_noise(
                positive[row : row + 1], sigma, generator, split=HIDDEN_SPLIT
            )
        )
        pooled_noise.append(
            rms_scaled_noise(positive_pooled[row : row + 1], sigma, generator)
        )
    embed_noise = torch.cat(noise)
    pooled_offset = torch.cat(pooled_noise)

    def positive_at(step_index: int):
        weight = early_weight(step_index, num_inference_steps, lo, hi)
        if weight == 0.0:
            return positive, positive_pooled
        return positive + weight * embed_noise, positive_pooled + weight * pooled_offset

    return positive_at


PG_SPACES = ("latent", "semantic")
PG_LO = 0.5
PG_HI = 0.75
PG_POOL = 8
PG_SEMANTIC_SIDE = 28
TAESD_ID = "madebyollin/taesdxl"
DINO_ID = "facebook/dinov2-small"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_DINO_PATCH = 14


def pg_grid(height: int, width: int, space: str) -> tuple[int, int]:
    if space == "latent":
        return max(1, height // PG_POOL), max(1, width // PG_POOL)
    scale = PG_SEMANTIC_SIDE / min(height, width)
    return max(1, round(height * scale)), max(1, round(width * scale))


class SemanticEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from diffusers import AutoencoderTiny
        from transformers import Dinov2Model

        taesd: Any = AutoencoderTiny.from_pretrained(TAESD_ID)
        self.decoder = taesd.decoder
        self.dino = Dinov2Model.from_pretrained(DINO_ID)
        self.eval().requires_grad_(False)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        images = self.decoder(latents.float())
        size = [
            _DINO_PATCH * max(1, round(side / _DINO_PATCH))
            for side in images.shape[-2:]
        ]
        images = F.interpolate(images, size=size, mode="bilinear", align_corners=False)
        mean = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
        pixels = ((images + 1.0) / 2.0 - mean) / std
        return F.normalize(self.dino(pixel_values=pixels).pooler_output, dim=-1)


_EMBEDDERS: dict[str, SemanticEmbedder] = {}


def semantic_embedder(device) -> SemanticEmbedder:
    key = str(device)
    if key not in _EMBEDDERS:
        _EMBEDDERS[key] = SemanticEmbedder().to(device)
    return _EMBEDDERS[key]


def _flat(coarse: torch.Tensor) -> torch.Tensor:
    return coarse.flatten(1)


def repulsion(x0: torch.Tensor, grid: tuple[int, int], embed=_flat) -> torch.Tensor:
    batch, _, height, width = x0.shape
    none = torch.zeros(x0.shape, dtype=torch.float32, device=x0.device)
    if batch < 2:
        return none

    coarse = F.adaptive_avg_pool2d(x0.detach().float(), grid)
    off_diagonal = ~torch.eye(batch, dtype=torch.bool, device=x0.device)
    with torch.enable_grad():
        coarse.requires_grad_(True)
        features = embed(coarse).float().flatten(1)
        diff = features[:, None, :] - features[None, :, :]
        dist2 = diff.pow(2).sum(-1)[off_diagonal]
        bandwidth = float(dist2.detach().median()) / math.log(batch)
        if bandwidth <= 0.0:
            return none
        potential = torch.exp(-dist2 / bandwidth).sum()
        (gradient,) = torch.autograd.grad(potential, coarse)

    field = F.interpolate(
        -gradient, size=(height, width), mode="bilinear", align_corners=False
    )
    rms = float(field.pow(2).mean().sqrt())
    if not math.isfinite(rms) or rms <= 0.0:
        return none
    return field / rms


def _epsilon_euler(scheduler) -> bool:
    prediction = getattr(getattr(scheduler, "config", None), "prediction_type", None)
    return isinstance(scheduler, EulerDiscreteScheduler) and prediction == "epsilon"


@contextlib.contextmanager
def particle_guidance(
    pipe,
    num_inference_steps: int,
    strength: float,
    lo: float,
    hi: float,
    space: str = "latent",
):
    if strength <= 0.0:
        yield
        return

    if space not in PG_SPACES:
        raise ValueError(f"pg space must be one of {PG_SPACES}, not {space!r}")
    scheduler = pipe.scheduler
    if not _epsilon_euler(scheduler):
        raise ValueError(
            "particle guidance reads x0 as sample - sigma * eps and needs the pinned "
            f"epsilon EulerDiscreteScheduler, not {type(scheduler).__name__}"
        )
    embed = semantic_embedder(pipe.unet.device) if space == "semantic" else _flat
    original = scheduler.step

    def step(model_output, timestep, sample, *args, **kwargs):
        index = int((scheduler.timesteps - timestep).abs().argmin().item())
        weight = early_weight(index, num_inference_steps, lo, hi)
        if weight == 0.0:
            return original(model_output, timestep, sample, *args, **kwargs)
        sigma = float(scheduler.sigmas[index])
        x0 = sample.float() - sigma * model_output.float()
        grid = pg_grid(x0.shape[-2], x0.shape[-1], space)
        shift = (strength * weight / sigma) * repulsion(x0, grid, embed)
        guided = (model_output.float() - shift).to(model_output.dtype)
        return original(guided, timestep, sample, *args, **kwargs)

    scheduler.step = step
    try:
        yield
    finally:
        scheduler.step = original


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
    cond_sigma: float = 0.0,
    cond_lo: float = COND_LO,
    cond_hi: float = COND_HI,
    pg_strength: float = 0.0,
    pg_lo: float = PG_LO,
    pg_hi: float = PG_HI,
    pg_space: str = "latent",
):
    batch = len(seeds)
    last_step = num_inference_steps - 1

    positive = positive.expand(batch, -1, -1)
    negative = negative.expand(batch, -1, -1)
    positive_pooled = positive_pooled.expand(batch, -1)
    negative_pooled = negative_pooled.expand(batch, -1)

    if use_lora:
        pipe.enable_lora()
        pipe.set_adapters(ALL_SEGMENTS, adapter_weights(0, strength))
    else:
        pipe.disable_lora()

    jitter_generator = torch.Generator(device=positive.device).manual_seed(
        seeds[0] ^ 0x5EED
    )
    anchor_at, raw_guidance = anchor_plan(
        positive,
        positive_pooled,
        negative,
        negative_pooled,
        num_inference_steps,
        guidance,
        alpha,
        sigma,
        cutover,
        sigma_cutover,
        jitter_generator,
    )
    positive_at = cond_noise_plan(
        positive,
        positive_pooled,
        seeds,
        num_inference_steps,
        cond_sigma,
        cond_lo,
        cond_hi,
    )

    def callback(p, step_index, timestep, callback_kwargs):  # noqa: ARG001
        if use_lora:
            if step_index + 1 == last_step:
                p.set_adapters(ALL_SEGMENTS, [0.0, 0.0, 0.0])
            else:
                p.set_adapters(ALL_SEGMENTS, adapter_weights(step_index + 1, strength))

        p._guidance_scale = raw_guidance(min(step_index + 1, last_step))

        anchor, anchor_pooled = anchor_at(min(step_index + 1, last_step))
        step_positive, step_pooled = positive_at(min(step_index + 1, last_step))
        callback_kwargs["prompt_embeds"] = torch.cat([anchor, step_positive])
        callback_kwargs["add_text_embeds"] = torch.cat([anchor_pooled, step_pooled])
        return callback_kwargs

    pipe.scheduler.set_timesteps(num_inference_steps, device=pipe.unet.device)
    generators = [torch.Generator(device="cuda").manual_seed(s) for s in seeds]
    scale = pipe.vae_scale_factor
    latents = prior_latents(
        (batch, pipe.unet.config.in_channels, height // scale, width // scale),
        generators,
        pipe.unet.device,
        pipe.unet.dtype,
        init_sigma=float(pipe.scheduler.init_noise_sigma),
        lf=latent_lf,
    )

    start_anchor, start_anchor_pooled = anchor_at(0)
    start_positive, start_positive_pooled = positive_at(0)

    try:
        with (
            lie_about_noise(pipe, offset, gamma),
            particle_guidance(
                pipe, num_inference_steps, pg_strength, pg_lo, pg_hi, pg_space
            ),
        ):
            result = pipe(
                prompt_embeds=start_positive,
                pooled_prompt_embeds=start_positive_pooled,
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
