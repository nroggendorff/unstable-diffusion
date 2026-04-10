import argparse
import torch

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
    ).images

    return images


def make_grid(images, rows=2, cols=3):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        grid.paste(img, (col * w, row * h))
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
    parser.add_argument("--steps", "-s", type=int, default=30)
    parser.add_argument("--guidance", "-g", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading model...")
    pipe = load_pipe()

    print(f"Generating: {args.prompt}")
    images = []
    images.extend(
        infer_batch(
            pipe,
            args.prompt,
            use_lora=True,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
        )
    )
    images.extend(
        infer_batch(
            pipe,
            args.prompt,
            use_lora=False,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
        )
    )

    grid = make_grid(images, rows=2, cols=3)
    grid.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
