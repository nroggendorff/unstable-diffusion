import argparse
import torch

from diffusers import StableDiffusionXLPipeline
from peft import PeftModel


MODEL_ID = "glides/illustriousxl"
ADAPTER_PATH = "./creative-early-step/lora"
OUTPUT_DIR = "./outputs"


def load_pipe():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to("cuda")

    pipe.unet = PeftModel.from_pretrained(pipe.unet, ADAPTER_PATH)
    pipe.unet.merge_and_unload()

    return pipe


def infer(pipe, prompt, num_inference_steps=30, guidance_scale=7.0, seed=None):
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator = None

    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--output", "-o", type=str, default="output.png")
    parser.add_argument("--steps", "-s", type=int, default=30)
    parser.add_argument("--guidance", "-g", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        safe_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
        output_path = f"{OUTPUT_DIR}/{safe_prompt}.png"

    print(f"Loading model from {ADAPTER_PATH}...")
    pipe = load_pipe()

    print(f"Generating: {args.prompt}")
    image = infer(pipe, args.prompt, args.steps, args.guidance, args.seed)

    image.save(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
