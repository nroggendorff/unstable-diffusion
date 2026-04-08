import argparse
import torch

from diffusers import StableDiffusionPipeline
from PIL import Image


MODEL_ID = "glides/counterfeit"
ADAPTER_PATH = "./creative-lora"
ADAPTER_NAME = "creative"


def load_pipe():
    # pyrefly: ignore [missing-attribute]
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to("cuda")

    pipe.load_lora_weights(ADAPTER_PATH, adapter_name=ADAPTER_NAME)

    return pipe


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
        pipe.set_adapters([ADAPTER_NAME])
        pipe.enable_lora()
    else:
        pipe.disable_lora()

    # pyrefly: ignore [unsupported-operation]
    seeds = [seed + i for i in range(batch_size)]
    generators = [torch.Generator(device="cuda").manual_seed(s) for s in seeds]

    images = pipe(
        prompt=[prompt] * batch_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generators,
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
    parser.add_argument("prompt", type=str)
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
