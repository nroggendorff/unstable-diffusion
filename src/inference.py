import argparse
import torch
import numpy as np

from diffusers import StableDiffusionPipeline
from PIL import Image

from .model import EARLY_SEG, MID_SEG, LATE_SEG


MODEL_ID = "glides/counterfeit"
ADAPTER_BASE_PATH = "./creative-lora"

BLEND_SEGMENTS = ["early", "mid", "late"]
ALL_SEGMENTS = ["early", "mid", "late", "final"]

_SEGMENT_CENTERS = [
    EARLY_SEG / 2,
    EARLY_SEG + MID_SEG / 2,
    EARLY_SEG + MID_SEG + LATE_SEG / 2,
]
_BLEND_WINDOW = 8


def _adapter_weights(step_index: int) -> list[float]:
    raw = [
        max(0.0, 1.0 - abs(step_index - center) / _BLEND_WINDOW)
        for center in _SEGMENT_CENTERS
    ]
    total = sum(raw) + 1e-8
    return [w / total for w in raw]


def load_pipe():
    # pyrefly: ignore [missing-attribute]
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to("cuda")

    for segment in ALL_SEGMENTS:
        pipe.load_lora_weights(
            f"{ADAPTER_BASE_PATH}/{segment}",
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name=segment,
        )

    return pipe


def make_segment_callback(use_lora, num_inference_steps):
    last_step = num_inference_steps - 1

    def callback(pipe, step_index, timestep, callback_kwargs):  # noqa: ARG001
        if not use_lora:
            return callback_kwargs
        if step_index == last_step:
            pipe.set_adapters(["final"], adapter_weights=[1.0])
        else:
            pipe.set_adapters(
                BLEND_SEGMENTS, adapter_weights=_adapter_weights(step_index)
            )
        return callback_kwargs

    return callback


def _decode_latents(pipe, latents: torch.Tensor) -> Image.Image:
    latents = latents.to(dtype=pipe.vae.dtype)
    with torch.no_grad():
        decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
    decoded = (decoded.float() / 2 + 0.5).clamp(0, 1)
    arr = (decoded[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def infer_batch(
    pipe,
    prompt,
    use_lora=True,
    num_inference_steps=30,
    guidance_scale=7.0,
    seed=None,
    batch_size=3,
):
    if use_lora:
        pipe.enable_lora()
        pipe.set_adapters(BLEND_SEGMENTS, adapter_weights=_adapter_weights(0))
    else:
        pipe.disable_lora()

    # pyrefly: ignore [unsupported-operation]
    seeds = [seed + i for i in range(batch_size)]
    generators = [torch.Generator(device="cuda").manual_seed(s) for s in seeds]

    images = pipe(
        prompt=[prompt] * batch_size,
        width=1024,
        height=1024,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generators,
        callback_on_step_end=make_segment_callback(use_lora, num_inference_steps),
        callback_on_step_end_tensor_inputs=["latents"],
    ).images

    return images


def infer_with_evolution(
    pipe,
    prompt,
    use_lora=True,
    num_inference_steps=30,
    guidance_scale=7.0,
    seed=None,
):
    if use_lora:
        pipe.enable_lora()
        pipe.set_adapters(BLEND_SEGMENTS, adapter_weights=_adapter_weights(0))
    else:
        pipe.disable_lora()

    captured_latents = []
    segment_cb = make_segment_callback(use_lora, num_inference_steps)

    def callback(p, step_index, timestep, callback_kwargs):
        result = segment_cb(p, step_index, timestep, callback_kwargs)
        if step_index % 2 == 0:
            captured_latents.append(callback_kwargs["latents"][:1].clone())
        return result

    # pyrefly: ignore [bad-argument-type]
    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        prompt=[prompt],
        width=1024,
        height=1024,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )

    final_image = result.images[0]
    evolution_frames = [_decode_latents(pipe, lat) for lat in captured_latents]

    return final_image, evolution_frames


def make_grid(images, rows=2, cols=3):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        grid.paste(img, (col * w, row * h))
    return grid


def make_evolution_grid(
    lora_frames: list[Image.Image], base_frames: list[Image.Image]
) -> Image.Image:
    cols = max(len(lora_frames), len(base_frames))
    w, h = lora_frames[0].size

    def pad_row(frames, target_cols):
        blank = Image.new("RGB", (w, h), color=(0, 0, 0))
        return frames + [blank] * (target_cols - len(frames))

    lora_row = pad_row(lora_frames, cols)
    base_row = pad_row(base_frames, cols)

    grid = Image.new("RGB", size=(cols * w, 2 * h))
    for col, img in enumerate(lora_row):
        grid.paste(img, (col * w, 0))
    for col, img in enumerate(base_row):
        grid.paste(img, (col * w, h))
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="A woman with long, wavy pink hair is shown in profile, "
        "wearing a dark, strapless dress with lace trim around the neckline and armholes. "
        "She has a prominent gold chain around her neck.",
    )
    parser.add_argument("--output", "-o", type=str, default="output.png")
    parser.add_argument("--evolution-output", "-e", type=str, default="evolution.png")
    parser.add_argument("--steps", "-s", type=int, default=30)
    parser.add_argument("--guidance", "-g", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading model...")
    pipe = load_pipe()

    print(f"Generating: {args.prompt}")

    lora_final, lora_frames = infer_with_evolution(
        pipe,
        args.prompt,
        use_lora=True,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )

    base_final, base_frames = infer_with_evolution(
        pipe,
        args.prompt,
        use_lora=False,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )

    lora_rest = infer_batch(
        pipe,
        args.prompt,
        use_lora=True,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed + 1,
        batch_size=2,
    )

    base_rest = infer_batch(
        pipe,
        args.prompt,
        use_lora=False,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed + 1,
        batch_size=2,
    )

    grid = make_grid(
        [lora_final] + lora_rest + [base_final] + base_rest, rows=2, cols=3
    )
    grid.save(args.output)
    print(f"Saved to {args.output}")

    evolution_grid = make_evolution_grid(lora_frames, base_frames)
    evolution_grid.save(args.evolution_output)
    print(f"Evolution saved to {args.evolution_output}")


if __name__ == "__main__":
    main()
