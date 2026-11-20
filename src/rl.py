import argparse
import functools
import json
import os
import random

import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import EulerDiscreteScheduler
from tqdm import tqdm

from .chain import (
    lora_parameters,
    rollout_chain,
    sample_grad_window,
    save_chain_lora,
)
from .config import default_output_dir
from .dataset import DATASET_ID
from .encoding import encode_for_sampling_masked, subset_caption
from .inference import ADAPTER_BASE_PATH, load_pipe
from .model import (
    DEVICE,
    LORA_RANK,
    NUM_INFERENCE_STEPS,
    get_lora_config,
    load_pipeline,
)
from .sampler import (
    ALL_SEGMENTS,
    NEGATIVE_PROMPT,
    adapter_weights,
    anchor_plan,
    lied_timestep,
    prior_latents,
)
from .scorer import ImageScorer

REFERENCE_ADAPTERS = [f"{segment}_ref" for segment in ALL_SEGMENTS]

BUCKETS = [(1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216)]

LAION_QUALITY_FLOOR = 5.0

PRESETS = [
    dict(
        guidance=6.0,
        alpha=0.0,
        sigma=0.0,
        cutover=0.8,
        sigma_cutover=0.8,
        strength=1.0,
        latent_lf=1.0,
        offset=0.0,
        gamma=1.0,
    ),
    dict(
        guidance=7.0,
        alpha=0.4,
        sigma=0.2,
        cutover=0.8,
        sigma_cutover=0.8,
        strength=0.8,
        latent_lf=1.0,
        offset=0.0,
        gamma=1.0,
    ),
    dict(
        guidance=5.0,
        alpha=0.2,
        sigma=0.2,
        cutover=0.4,
        sigma_cutover=0.8,
        strength=1.0,
        latent_lf=1.0,
        offset=0.0,
        gamma=1.0,
    ),
]

SUBJECTS = [
    "an old fisherman",
    "a robot tending a garden",
    "a sleeping dragon",
    "a bowl of ramen",
    "a crowded train platform",
    "two foxes",
    "a mountain shrine",
    "a girl playing the violin",
    "a pirate captain",
    "an astronaut",
    "a street food vendor",
    "a wolf",
    "a tea house",
    "a whale swimming through clouds",
    "a samurai",
    "a bakery window",
    "a grandmother and her grandson",
    "a group of students",
    "a black cat on a fence",
    "an abandoned amusement park",
    "a chef in a busy kitchen",
    "a knight on horseback",
    "a still life of fruit and a vase",
    "a boy flying a kite",
]

SETTINGS = [
    "",
    "",
    "at sunset",
    "in the rain",
    "under the sea",
    "in a neon city at night",
    "in a desert",
    "in a misty forest",
    "inside a library",
    "during a summer festival",
    "in deep winter",
    "in outer space",
]

STYLES = [
    "",
    "",
    "",
    "watercolor",
    "ink sketch",
    "oil painting",
    "pixel art",
    "flat colors",
    "monochrome manga panel with screentone",
    "cel shading",
    "pastel colors",
    "charcoal drawing",
    "ukiyo-e woodblock print",
    "children's book illustration",
    "dramatic chiaroscuro lighting",
    "thick impasto brushstrokes",
]


def random_settings(rng: random.Random) -> dict:
    return dict(
        guidance=rng.uniform(4.0, 8.0),
        alpha=rng.choice([0.0, 0.2, 0.4]),
        sigma=rng.uniform(0.0, 0.3),
        cutover=rng.uniform(0.4, 0.85),
        sigma_cutover=rng.uniform(0.7, 0.9),
        strength=rng.uniform(0.6, 1.2),
        latent_lf=rng.uniform(0.6, 1.3),
        offset=0.0 if rng.random() < 0.7 else rng.uniform(-0.1, 0.1),
        gamma=1.0 if rng.random() < 0.7 else rng.uniform(0.7, 1.5),
    )


def draw_settings(rng: random.Random, preset_prob: float) -> dict:
    if rng.random() < preset_prob:
        return dict(rng.choice(PRESETS))
    return random_settings(rng)


def composed_prompt(rng: random.Random) -> str:
    subject = rng.choice(SUBJECTS)
    setting = rng.choice(SETTINGS)
    style = rng.choice(STYLES)
    text = f"{subject} {setting}".strip()
    return f"{text}, {style}" if style else text


def caption_stream(seed: int, subset_prob: float, subset_min: float):
    rng = random.Random(seed)
    epoch = 0
    while True:
        dataset = load_dataset(
            DATASET_ID, split="train", streaming=True, columns=["text"]
        ).shuffle(seed=seed + epoch, buffer_size=1024)
        for row in dataset:
            text = row.get("text")
            if not text:
                continue
            if rng.random() < subset_prob:
                text = subset_caption(text, rng, subset_min)
            yield text
        epoch += 1


def floor_penalty(value: torch.Tensor, floor: float, temperature: float):
    temperature = max(temperature, 1e-6)
    return F.softplus((floor - value) / temperature) * temperature


def scaled_bucket(bucket: tuple[int, int], scale: float) -> tuple[int, int]:
    width, height = (max(256, int(round(side * scale / 32)) * 32) for side in bucket)
    return width, height


def adapter_exposure(grad_steps: set, strength: float, steps: int) -> list[float]:
    totals = [0.0] * len(ALL_SEGMENTS)
    for index in grad_steps:
        if index == steps - 1:
            continue
        for segment, weight in enumerate(adapter_weights(index, strength)):
            totals[segment] += weight
    return totals


def parse_steps(text: str) -> set[int]:
    return {int(part) for part in text.split(",") if part.strip()}


@torch.no_grad()
def group_norm(tensors: list) -> float | None:
    present = [t for t in tensors if t is not None]
    if not present:
        return None
    return float(torch.stack(torch._foreach_norm(present)).norm())


@torch.no_grad()
def clip_by_exposure(groups: dict, exposure: dict, clip: float) -> tuple[dict, dict]:
    before, after = {}, {}
    for segment, params in groups.items():
        grads = [param.grad for param in params if param.grad is not None]
        norm = group_norm(grads)
        before[segment] = after[segment] = norm
        if norm is None or clip <= 0.0:
            continue
        limit = clip * exposure.get(segment, 0.0)
        if norm > limit:
            torch._foreach_mul_(grads, limit / norm)
            after[segment] = limit
    return before, after


def decode_pixels(vae, latents: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode(latents.to(dtype=vae.dtype) / vae.config.scaling_factor)
    decoded = decoded.sample.float()
    return decoded + (decoded.clamp(-1.0, 1.0) - decoded).detach()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default=ADAPTER_BASE_PATH)
    parser.add_argument("--fresh", type=int, default=0)
    parser.add_argument("--lora_rank", type=int, default=LORA_RANK)
    parser.add_argument("--output_dir", type=str, default=default_output_dir())
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--grad_window", type=int, default=10)
    parser.add_argument("--grad_target_rms", type=float, default=0.0)
    parser.add_argument("--num_inference_steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset_prob", type=float, default=0.5)
    parser.add_argument("--dataset_prompt_prob", type=float, default=0.5)
    parser.add_argument("--caption_subset_prob", type=float, default=0.5)
    parser.add_argument("--caption_subset_min", type=float, default=0.15)
    parser.add_argument(
        "--scorer", type=str, default="laion", choices=["laion", "waifu"]
    )
    parser.add_argument("--scorer_views", type=int, default=3)
    parser.add_argument("--quality_floor", type=float, default=None)
    parser.add_argument("--quality_floor_pct", type=float, default=25.0)
    parser.add_argument("--quality_temp", type=float, default=0.25)
    parser.add_argument("--adherence_weight", type=float, default=15.0)
    parser.add_argument("--adherence_floor", type=float, default=None)
    parser.add_argument("--adherence_floor_pct", type=float, default=30.0)
    parser.add_argument("--adherence_temp", type=float, default=0.02)
    parser.add_argument("--tether_weight", type=float, default=10.0)
    parser.add_argument("--tether_floor", type=float, default=0.85)
    parser.add_argument("--tether_temp", type=float, default=0.02)
    parser.add_argument("--calibration_steps", type=int, default=8)
    parser.add_argument("--unet_checkpoint", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--save_steps", type=str, default="")
    return parser.parse_args()


class QualityRL:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rng = random.Random(args.seed)
        self.captions = caption_stream(
            args.seed, args.caption_subset_prob, args.caption_subset_min
        )

        if args.fresh:
            pipe = load_pipeline()
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            for segment in ALL_SEGMENTS:
                pipe.unet.add_adapter(
                    get_lora_config(args.lora_rank, args.lora_rank),
                    adapter_name=segment,
                )
            self.reference_adapters = None
            print(
                f"Fresh rank-{args.lora_rank} adapters; "
                "the tether reference is the base model"
            )
        else:
            pipe = load_pipe(args.adapter_path)
            self.reference_adapters = REFERENCE_ADAPTERS
            if args.tether_weight > 0.0:
                for segment, name in zip(ALL_SEGMENTS, REFERENCE_ADAPTERS):
                    pipe.load_lora_weights(
                        args.adapter_path,
                        weight_name=f"{segment}.safetensors",
                        adapter_name=name,
                    )
        pipe.set_progress_bar_config(disable=True)
        pipe.unet.requires_grad_(False)
        pipe.vae.requires_grad_(False)
        self.params = lora_parameters(pipe.unet)
        print(f"Trainable LoRA tensors: {len(self.params)}")

        def segment_params(segment: str, kinds=("lora_A", "lora_B")):
            return [
                param
                for name, param in pipe.unet.named_parameters()
                if any(f".{kind}.{segment}.weight" in name for kind in kinds)
            ]

        self.groups = {segment: segment_params(segment) for segment in ALL_SEGMENTS}
        self.b_groups = {
            segment: segment_params(segment, ("lora_B",)) for segment in ALL_SEGMENTS
        }
        self.log_path = ""

        if args.unet_checkpoint:
            pipe.unet.enable_gradient_checkpointing()
            pipe.vae.enable_gradient_checkpointing()

        scheduler = pipe.scheduler
        scheduler.set_timesteps(args.num_inference_steps, device=DEVICE)
        self.sigmas = scheduler.sigmas.to(device=DEVICE, dtype=torch.float32)
        self.init_sigma = float(scheduler.init_noise_sigma)
        self.timesteps = scheduler.timesteps.to(DEVICE)

        self.pipe = pipe
        self.encoders = [pipe.text_encoder, pipe.text_encoder_2]
        self.tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
        self.scorer = ImageScorer(args.scorer, views=args.scorer_views).to(DEVICE)

    def prompts(self, count: int) -> tuple[list[str], list[str]]:
        prompts, sources = [], []
        for _ in range(count):
            if self.rng.random() < self.args.dataset_prompt_prob:
                prompts.append(next(self.captions))
                sources.append("dataset")
            else:
                prompts.append(composed_prompt(self.rng))
                sources.append("composed")
        return prompts, sources

    def rollout(self, prompts: list[str], settings: dict, grad: bool):
        args = self.args
        batch = len(prompts)
        steps = self.timesteps.shape[0]
        width, height = scaled_bucket(self.rng.choice(BUCKETS), args.scale)

        positive, pooled, negative, negative_pooled, chunk_mask = (
            encode_for_sampling_masked(
                prompts, NEGATIVE_PROMPT, self.encoders, self.tokenizers, DEVICE
            )
        )
        jitter_seed = self.rng.randrange(1 << 30)

        def plan(neg, neg_pooled):
            return anchor_plan(
                positive,
                pooled,
                neg.expand(batch, -1, -1),
                neg_pooled.expand(batch, -1),
                steps,
                settings["guidance"],
                settings["alpha"],
                settings["sigma"],
                settings["cutover"],
                settings["sigma_cutover"],
                torch.Generator(device=DEVICE).manual_seed(jitter_seed),
            )

        policy_plan = plan(
            torch.zeros_like(negative), torch.zeros_like(negative_pooled)
        )
        time_ids = torch.tensor(
            [[height, width, 0, 0, height, width]] * (2 * batch),
            device=DEVICE,
            dtype=torch.float16,
        )

        scale = self.pipe.vae_scale_factor
        generators = [
            torch.Generator(device=DEVICE).manual_seed(self.rng.randrange(1 << 30))
            for _ in range(batch)
        ]
        latents = prior_latents(
            (batch, 4, height // scale, width // scale),
            generators,
            DEVICE,
            torch.float32,
            init_sigma=self.init_sigma,
            lf=settings["latent_lf"],
        )

        offset, gamma = settings["offset"], settings["gamma"]
        timestep_at = None
        if offset != 0.0 or gamma != 1.0:
            timestep_at = functools.partial(
                lied_timestep, self.timesteps, offset=offset, gamma=gamma
            )

        grad_steps = (
            sample_grad_window(self.rng, steps - 1, args.grad_window) if grad else set()
        )

        def run(adapters, steps_with_grad, anchor, use_lora=True):
            anchor_at, raw_guidance = anchor
            return rollout_chain(
                self.pipe,
                self.init_sigma,
                self.sigmas,
                self.timesteps,
                latents,
                positive,
                pooled,
                time_ids,
                anchor_at,
                raw_guidance,
                settings["strength"],
                use_lora,
                steps_with_grad,
                args.grad_target_rms,
                adapters,
                timestep_at,
                chunk_mask,
            )

        with torch.set_grad_enabled(grad):
            generated = run(ALL_SEGMENTS, grad_steps, policy_plan)

        reference = None
        if grad and args.tether_weight > 0.0:
            with torch.no_grad():
                reference = run(
                    self.reference_adapters or ALL_SEGMENTS,
                    set(),
                    plan(negative, negative_pooled),
                    use_lora=self.reference_adapters is not None,
                )

        return generated, reference, (width, height), grad_steps

    def score(self, latents, prompts):
        embeds = self.scorer.image_embeds(decode_pixels(self.pipe.vae, latents))
        text = self.scorer.text_embeds(prompts, DEVICE)
        return embeds, self.scorer.quality(embeds), self.scorer.adherence(embeds, text)

    @torch.no_grad()
    def calibrate(self) -> tuple[float, float]:
        args = self.args
        qualities, adherences = [], []
        for _ in tqdm(range(args.calibration_steps), desc="calibrate"):
            prompts, _ = self.prompts(args.batch)
            settings = draw_settings(self.rng, args.preset_prob)
            generated, _, _, _ = self.rollout(prompts, settings, grad=False)
            _, quality, adherence = self.score(generated, prompts)
            qualities.append(quality.float().cpu())
            adherences.append(adherence.float().cpu())

        quality = torch.cat(qualities) if qualities else torch.zeros(1)
        adherence = torch.cat(adherences) if adherences else torch.zeros(1)

        quality_floor = args.quality_floor
        if quality_floor is None:
            quality_floor = (
                LAION_QUALITY_FLOOR
                if args.scorer == "laion"
                else torch.quantile(quality, args.quality_floor_pct / 100).item()
            )
        adherence_floor = args.adherence_floor
        if adherence_floor is None:
            adherence_floor = torch.quantile(
                adherence, args.adherence_floor_pct / 100
            ).item()

        def summary(values):
            q = torch.quantile(values, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
            return " / ".join(f"{v:.3f}" for v in q.tolist())

        print(
            f"Calibration over {quality.numel()} samples (p10/p25/p50/p75/p90):\n"
            f"  quality   {summary(quality)}  floor {quality_floor:.3f} "
            f"({(quality < quality_floor).float().mean().item():.0%} below)\n"
            f"  adherence {summary(adherence)}  floor {adherence_floor:.3f} "
            f"({(adherence < adherence_floor).float().mean().item():.0%} below)"
        )
        self.log(
            calibration=dict(
                quality=quality.tolist(),
                adherence=adherence.tolist(),
                quality_floor=quality_floor,
                adherence_floor=adherence_floor,
            )
        )
        return quality_floor, adherence_floor

    def log(self, **record):
        if not self.log_path:
            return
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def train(self):
        args = self.args
        os.makedirs(args.output_dir, exist_ok=True)
        self.log_path = os.path.join(args.output_dir, "rl_log.jsonl")
        open(self.log_path, "w").close()
        self.log(args=vars(args))
        quality_floor, adherence_floor = self.calibrate()
        optimizer = torch.optim.AdamW(self.params, lr=args.lr)
        save_steps = parse_steps(args.save_steps)
        steps = self.timesteps.shape[0]

        for step in (pbar := tqdm(range(args.steps), desc="rl")):
            optimizer.zero_grad(set_to_none=True)

            prompts, sources = self.prompts(args.batch)
            settings = draw_settings(self.rng, args.preset_prob)
            generated, reference, size, grad_steps = self.rollout(
                prompts, settings, grad=True
            )
            exposure = adapter_exposure(grad_steps, settings["strength"], steps)

            embeds, quality, adherence = self.score(generated, prompts)
            penalty = floor_penalty(quality, quality_floor, args.quality_temp)
            penalty = penalty + args.adherence_weight * floor_penalty(
                adherence, adherence_floor, args.adherence_temp
            )

            identity = torch.ones_like(quality)
            if reference is not None:
                with torch.no_grad():
                    reference_embeds = self.scorer.image_embeds(
                        decode_pixels(self.pipe.vae, reference)
                    )
                identity = self.scorer.identity(embeds, reference_embeds)
                penalty = penalty + args.tether_weight * floor_penalty(
                    identity, args.tether_floor, args.tether_temp
                )

            loss = penalty.mean()
            if not torch.isfinite(loss):
                print(f"\nstep {step}: non-finite loss, skipped")
                self.log(step=step, skipped="non-finite loss")
                del generated, reference, loss
                continue

            loss.backward()
            exposed = dict(zip(ALL_SEGMENTS, exposure))
            segment_grads, clipped_grads = clip_by_exposure(
                self.groups, exposed, args.grad_clip_norm
            )
            grad_norm = sum(n**2 for n in segment_grads.values() if n) ** 0.5
            optimizer.step()

            record = dict(
                step=step,
                loss=loss.item(),
                quality=quality.tolist(),
                adherence=adherence.tolist(),
                identity=identity.tolist(),
                grad_norm=grad_norm,
                segment_grad_norm=segment_grads,
                clipped_grad_norm=clipped_grads,
                exposure=exposed,
                window=[min(grad_steps), max(grad_steps)],
                resolution=list(size),
                settings=settings,
                sources=sources,
                prompts=prompts,
            )
            if step % 10 == 0 or step + 1 == args.steps:
                record["b_norm"] = {
                    segment: group_norm(list(params))
                    for segment, params in self.b_groups.items()
                }
            self.log(**record)

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                q=f"{quality.mean().item():.2f}",
                qmin=f"{quality.min().item():.2f}",
                adh=f"{adherence.mean().item():.3f}",
                ident=f"{identity.mean().item():.3f}",
                gnorm=f"{grad_norm:.2e}",
                win=f"{min(grad_steps)}-{max(grad_steps)}",
                res=f"{size[0]}x{size[1]}",
                vram=f"{torch.cuda.max_memory_allocated() / 2**30:.1f}G",
            )

            del generated, reference, embeds, loss, penalty
            if step % 10 == 0:
                torch.cuda.empty_cache()

            done = step + 1
            if done < args.steps and (
                done in save_steps or (args.save_every and done % args.save_every == 0)
            ):
                self.save(os.path.join(args.output_dir, f"step{done}"))

        self.save(args.output_dir)

    def save(self, path: str):
        for segment in ALL_SEGMENTS:
            count = save_chain_lora(self.pipe.unet, path, segment)
            print(f"Saved {segment}: {count} tensors -> {path}")


def main():
    args = get_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"RL: {vars(args)}")
    QualityRL(args).train()


if __name__ == "__main__":
    main()
