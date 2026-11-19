import os

import torch
from safetensors.torch import save_file

from .sampler import ALL_SEGMENTS, adapter_weights


class _NormalizeGrad(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, x, target_rms):
        ctx.target_rms = target_rms
        return x

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad):
        rms = grad.float().pow(2).flatten(1).mean(1).sqrt().view(-1, 1, 1, 1)
        scale = torch.where(rms > 0, ctx.target_rms / rms, torch.ones_like(rms))
        return grad * scale.to(grad.dtype), None


def normalize_grad(x: torch.Tensor, target_rms: float) -> torch.Tensor:
    if target_rms <= 0.0:
        return x
    return _NormalizeGrad.apply(x, target_rms)


def active_adapters(adapters, weights) -> tuple[list[str], list[float]]:
    pairs = [(name, weight) for name, weight in zip(adapters, weights) if weight != 0.0]
    return [name for name, _ in pairs], [weight for _, weight in pairs]


_UNKNOWN = object()


class AdapterSwitch:
    def __init__(self, pipe):
        self.pipe = pipe
        self.state = _UNKNOWN

    def apply(self, names: list[str], scales: list[float]):
        state = (tuple(names), tuple(scales))
        if state == self.state:
            return
        if names:
            self.pipe.enable_lora()
            self.pipe.set_adapters(names, scales)
        else:
            self.pipe.disable_lora()
        self.state = state

    def forget(self):
        self.state = _UNKNOWN


def sample_grad_window(rng, trainable_steps: int, window: int) -> set[int]:
    if window <= 0 or window >= trainable_steps:
        return set(range(trainable_steps))
    start = rng.randrange(1 - window, trainable_steps)
    return set(range(max(0, start), min(trainable_steps, start + window)))


def lora_parameters(unet, segments=ALL_SEGMENTS):
    params = []
    for name, param in unet.named_parameters():
        if "lora_" in name and any(f".{seg}.weight" in name for seg in segments):
            param.data = param.data.float()
            param.requires_grad_(True)
            params.append(param)
    return params


def save_chain_lora(unet, path: str, segment: str) -> int:
    out = {}
    for name, param in unet.named_parameters():
        for kind in ("lora_A", "lora_B"):
            suffix = f".{kind}.{segment}.weight"
            if name.endswith(suffix):
                key = "unet." + name[: -len(suffix)] + f".{kind}.weight"
                out[key] = param.detach().float().cpu().contiguous()

    os.makedirs(path, exist_ok=True)
    save_file(out, os.path.join(path, f"{segment}.safetensors"))
    return len(out)


def _unet_eps(
    pipe,
    x,
    timestep,
    embeds,
    pooled,
    time_ids,
    sigma,
    guidance,
    weights,
    use_lora,
    use_ckpt,
    adapters=ALL_SEGMENTS,
    attention_mask=None,
    switch: AdapterSwitch | None = None,
):
    if switch is None:
        switch = AdapterSwitch(pipe)

    def run(sample, prompt_embeds, add_text_embeds):
        names, scales = active_adapters(adapters, weights) if use_lora else ([], [])
        switch.apply(names, scales)

        model_in = torch.cat([sample, sample]) / ((sigma**2 + 1) ** 0.5)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = pipe.unet(
                model_in.to(dtype=torch.float16),
                timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=attention_mask,
                added_cond_kwargs={
                    "text_embeds": add_text_embeds,
                    "time_ids": time_ids,
                },
            ).sample

        uncond, cond = out.float().chunk(2)
        return uncond + guidance * (cond - uncond)

    if use_ckpt:
        return torch.utils.checkpoint.checkpoint(
            run, x, embeds, pooled, use_reentrant=False
        )
    return run(x, embeds, pooled)


def rollout_chain(
    pipe,
    init_sigma: float,
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    latents: torch.Tensor,
    positive: torch.Tensor,
    positive_pooled: torch.Tensor,
    time_ids: torch.Tensor,
    anchor_at,
    raw_guidance,
    strength: float,
    use_lora: bool,
    grad_steps: set,
    grad_target_rms: float,
    adapters=ALL_SEGMENTS,
    timestep_at=None,
    attention_mask=None,
) -> torch.Tensor:
    steps = timesteps.shape[0]
    last = steps - 1
    x = latents.float() * init_sigma
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, attention_mask])
    sigma_values = sigmas[:steps].tolist()
    deltas = (sigmas[1 : steps + 1] - sigmas[:steps]).tolist()
    switch = AdapterSwitch(pipe)

    for index in range(steps):
        needs_grad = use_lora and index in grad_steps
        weights = [0.0, 0.0, 0.0] if index == last else adapter_weights(index, strength)
        anchor, anchor_pooled = anchor_at(index)
        embeds = torch.cat([anchor, positive])
        pooled = torch.cat([anchor_pooled, positive_pooled])

        sigma = sigma_values[index]
        delta = deltas[index]

        carries = needs_grad or (x.requires_grad and torch.is_grad_enabled())

        with torch.set_grad_enabled(needs_grad):
            x_in = normalize_grad(x, grad_target_rms) if needs_grad else x
            eps = _unet_eps(
                pipe,
                x_in,
                timesteps[index] if timestep_at is None else timestep_at(index),
                embeds,
                pooled,
                time_ids,
                sigma,
                raw_guidance(index),
                weights,
                use_lora,
                needs_grad,
                adapters,
                attention_mask,
                switch,
            )

        with torch.set_grad_enabled(carries):
            x_next = x_in + eps * delta

        x = x_next if carries else x_next.detach()

    switch.forget()
    return x
