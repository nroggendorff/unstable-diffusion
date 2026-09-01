import argparse
import os
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from safetensors.torch import save_file

from .cache import BucketSampler, build_cache
from .config import default_output_dir
from .dataset import get_samples
from .encoder import CLIPVisionEncoder, compute_perceptual_discrepancy
from .encoding import decode_for_clip, encode_prompt
from .eval import NEGATIVE_PROMPT
from .inference import ADAPTER_BASE_PATH, load_pipe
from .model import DEVICE, NUM_INFERENCE_STEPS
from .sampler import ALL_SEGMENTS, adapter_weights, progress, pyramid_latents

_CLIP_MODEL = "openai/clip-vit-base-patch32"
_MAX_RAW_GUIDANCE = 30.0


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


def _unet_eps(pipe, x, timestep, embeds, sigma, guidance, weights, use_lora, use_ckpt):
    def run(sample, prompt_embeds):
        if use_lora:
            pipe.enable_lora()
            pipe.set_adapters(ALL_SEGMENTS, weights)
        else:
            pipe.disable_lora()

        model_in = torch.cat([sample, sample]) / ((sigma**2 + 1) ** 0.5)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = pipe.unet(
                model_in.to(dtype=torch.float16),
                timestep,
                encoder_hidden_states=prompt_embeds,
            ).sample

        uncond, cond = out.float().chunk(2)
        return uncond + guidance * (cond - uncond)

    if use_ckpt:
        return torch.utils.checkpoint.checkpoint(run, x, embeds, use_reentrant=False)
    return run(x, embeds)


def rollout_chain(
    pipe,
    init_sigma: float,
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    latents: torch.Tensor,
    positive: torch.Tensor,
    anchor_at,
    raw_guidance,
    strength: float,
    use_lora: bool,
    grad_steps: set,
    grad_target_rms: float,
) -> torch.Tensor:
    steps = timesteps.shape[0]
    last = steps - 1
    x = latents.float() * init_sigma

    for index in range(steps):
        needs_grad = use_lora and index in grad_steps
        weights = [0.0, 0.0, 0.0] if index == last else adapter_weights(index, strength)
        embeds = torch.cat([anchor_at(index), positive])

        sigma = sigmas[index].item()
        delta = (sigmas[index + 1] - sigmas[index]).item()

        with torch.set_grad_enabled(needs_grad):
            x_in = normalize_grad(x, grad_target_rms) if needs_grad else x
            eps = _unet_eps(
                pipe,
                x_in,
                timesteps[index],
                embeds,
                sigma,
                raw_guidance(index),
                weights,
                use_lora,
                needs_grad,
            )
            x_next = x_in + eps * delta

        x = x_next if needs_grad else x_next.detach()

    return x


def chain_reward(
    gen_latents,
    ref_feats,
    base_latents,
    vae,
    clip_encoder,
    grounding_weight: float,
):
    gen_feats = clip_encoder.extract_features(
        decode_for_clip(vae, gen_latents, straight_through=True)
    )

    discrepancy = compute_perceptual_discrepancy(gen_feats, ref_feats)
    perceptual = -discrepancy.flatten(1).mean(1)

    if base_latents is not None and grounding_weight > 0.0:
        with torch.no_grad():
            base_feats = clip_encoder.extract_features(
                decode_for_clip(vae, base_latents, straight_through=True)
            )
        base_last = base_feats[-1].flatten(1)
        grounding = F.cosine_similarity(
            gen_feats[-1].flatten(1) - base_last,
            ref_feats[-1].flatten(1) - base_last,
            dim=1,
            eps=1e-6,
        )
    else:
        grounding = torch.zeros_like(perceptual)

    return perceptual + grounding_weight * grounding, perceptual, grounding


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default=ADAPTER_BASE_PATH)
    parser.add_argument("--output_dir", type=str, default=default_output_dir())
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--refs", type=int, default=2)
    parser.add_argument("--cache_size", type=int, default=512)
    parser.add_argument("--cache_seed", type=int, default=0)
    parser.add_argument("--grad_window", type=int, default=0)
    parser.add_argument("--grad_target_rms", type=float, default=1.0)
    parser.add_argument("--grounding_weight", type=float, default=0.4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--cutover", type=float, default=0.8)
    parser.add_argument("--sigma_cutover", type=float, default=0.8)
    parser.add_argument("--latent_lf", type=float, default=1.0)
    parser.add_argument("--unet_checkpoint", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=0)
    return parser.parse_args()


def main():
    args = get_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    window = args.grad_window if args.grad_window > 0 else "full"
    print(
        f"Chain RL: adapters={args.adapter_path} -> {args.output_dir}, "
        f"steps={args.steps}, lr={args.lr}, refs={args.refs}, "
        f"grad_window={window}, target_rms={args.grad_target_rms}, "
        f"grounding={args.grounding_weight}, cache_size={args.cache_size}"
    )

    pipe = load_pipe(args.adapter_path)
    pipe.set_progress_bar_config(disable=True)

    vae = pipe.vae
    unet = pipe.unet

    for param in unet.parameters():
        param.requires_grad_(False)
    params = lora_parameters(unet)
    print(f"Trainable LoRA tensors: {len(params)}")

    if args.unet_checkpoint:
        unet.enable_gradient_checkpointing()
        vae.enable_gradient_checkpointing()

    scheduler = pipe.scheduler
    scheduler.set_timesteps(args.num_inference_steps, device=DEVICE)
    sigmas = scheduler.sigmas.to(device=DEVICE, dtype=torch.float32)
    init_sigma = float(scheduler.init_noise_sigma)
    timesteps = scheduler.timesteps.to(DEVICE)
    steps = timesteps.shape[0]

    print("Building latent cache...")
    cached = build_cache(
        get_samples(args.cache_size, seed=args.cache_seed),
        vae,
        pipe.text_encoder,
        pipe.tokenizer,
        DEVICE,
        total=args.cache_size,
    )

    negative, _ = encode_prompt(
        NEGATIVE_PROMPT, pipe.text_encoder, pipe.tokenizer, DEVICE
    )
    negative = negative.detach().to(dtype=torch.float16)

    sampler = BucketSampler(cached, args.refs)
    if not sampler:
        raise RuntimeError(
            f"No aspect bucket holds at least {args.refs} samples; raise --cache_size."
        )
    print(f"Cached {len(cached)} samples; {sampler.usable} usable.")

    clip_encoder = CLIPVisionEncoder(model_name=_CLIP_MODEL, detach=False).to(DEVICE)

    optimizer = torch.optim.AdamW(params, lr=args.lr)

    for step in (pbar := tqdm(range(args.steps), desc="chain-rl")):
        optimizer.zero_grad(set_to_none=True)

        items = sampler.draw(args.refs)
        ref_latents = torch.cat([x["latents"] for x in items]).float().to(DEVICE)
        positive = torch.cat([x["text_emb"] for x in items]).to(
            DEVICE, dtype=torch.float16
        )
        batch = ref_latents.shape[0]

        with torch.no_grad():
            ref_feats = [
                f.detach()
                for f in clip_encoder.extract_features(
                    decode_for_clip(vae, ref_latents)
                )
            ]

        neg_batch = negative.expand(batch, -1, -1)
        anchor_base = torch.lerp(neg_batch.float(), positive.float(), args.alpha).to(
            torch.float16
        )

        rms = positive[0].float().norm() / (positive[0].numel() ** 0.5)
        anchor_jittered = (
            anchor_base.float()
            + torch.randn_like(anchor_base.float()) * args.sigma * rms
        ).to(torch.float16)

        def anchor_at(index, _n=neg_batch, _a=anchor_base, _j=anchor_jittered):
            if progress(index, steps) <= args.cutover:
                return _n
            if progress(index, steps) > args.sigma_cutover:
                return _j
            return _a

        def raw_guidance(index):
            if args.alpha == 0.0 or progress(index, steps) <= args.cutover:
                return args.guidance
            return max(
                1.01,
                min(
                    _MAX_RAW_GUIDANCE,
                    (args.guidance - args.alpha) / (1.0 - args.alpha),
                ),
            )

        generators = [
            torch.Generator(device=DEVICE).manual_seed(random.randrange(1 << 30))
            for _ in range(batch)
        ]
        latents = pyramid_latents(
            tuple(ref_latents.shape),
            generators,
            DEVICE,
            torch.float32,
            lf=args.latent_lf,
        )

        if 0 < args.grad_window < steps:
            start = random.randrange(0, steps - args.grad_window + 1)
            grad_steps = set(range(start, start + args.grad_window))
        else:
            grad_steps = set(range(steps))

        gen_latents = rollout_chain(
            pipe,
            init_sigma,
            sigmas,
            timesteps,
            latents,
            positive,
            anchor_at,
            raw_guidance,
            args.strength,
            True,
            grad_steps,
            args.grad_target_rms,
        )

        base_latents = None
        if args.grounding_weight > 0.0:
            with torch.no_grad():
                base_latents = rollout_chain(
                    pipe,
                    init_sigma,
                    sigmas,
                    timesteps,
                    latents,
                    positive,
                    anchor_at,
                    raw_guidance,
                    args.strength,
                    False,
                    set(),
                    args.grad_target_rms,
                )

        reward, perceptual, grounding = chain_reward(
            gen_latents,
            ref_feats,
            base_latents,
            vae,
            clip_encoder,
            args.grounding_weight,
        )

        loss = -reward.mean()

        if not torch.isfinite(loss):
            print(f"\nstep {step}: non-finite loss, skipped")
            del gen_latents, base_latents, reward
            continue

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
        optimizer.step()

        pbar.set_postfix(
            reward=f"{reward.mean().item():.4f}",
            perc=f"{perceptual.mean().item():.4f}",
            gnd=f"{grounding.mean().item():.3f}",
            gnorm=f"{grad_norm.item():.2e}",
            vram=f"{torch.cuda.max_memory_allocated() / 2**30:.1f}G",
        )

        del gen_latents, base_latents, reward, ref_feats, latents

        if step % 20 == 0:
            torch.cuda.empty_cache()

        if args.save_every and (step + 1) % args.save_every == 0:
            snapshot = os.path.join(args.output_dir, f"step{step + 1}")
            for segment in ALL_SEGMENTS:
                save_chain_lora(unet, snapshot, segment)
            print(f"\nSnapshot -> {snapshot}")

    for segment in ALL_SEGMENTS:
        count = save_chain_lora(unet, args.output_dir, segment)
        print(f"Saved {segment}: {count} tensors -> {args.output_dir}")


if __name__ == "__main__":
    main()
