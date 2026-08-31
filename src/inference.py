import argparse
import torch
import numpy as np

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image

from .encoding import encode_prompt
from .sampler import ALL_SEGMENTS, adapter_weights

MODEL_ID = "glides/counterfeit"
ADAPTER_BASE_PATH = "./creative-lora"

_NEGATIVE_PROMPT = "watermark, text"


def load_pipe(adapter_path: str = ADAPTER_BASE_PATH):
    # pyrefly: ignore [missing-attribute]
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    for segment in ALL_SEGMENTS:
        pipe.load_lora_weights(
            adapter_path,
            weight_name=f"{segment}.safetensors",
            adapter_name=segment,
        )

    return pipe


def _encode_for_pipe(pipe, prompt):
    device = pipe.text_encoder.device

    text_emb, _ = encode_prompt(prompt, pipe.text_encoder, pipe.tokenizer, device)
    neg_emb, _ = encode_prompt(
        _NEGATIVE_PROMPT, pipe.text_encoder, pipe.tokenizer, device
    )

    return text_emb.to(dtype=torch.float16), neg_emb.to(dtype=torch.float16)


def make_segment_callback(use_lora, num_inference_steps, strength):
    last_step = num_inference_steps - 1

    def callback(pipe, step_index, timestep, callback_kwargs):  # noqa: ARG001
        if not use_lora:
            return callback_kwargs

        if step_index + 1 == last_step:
            pipe.set_adapters(ALL_SEGMENTS, [0.0, 0.0, 0.0])
        else:
            pipe.set_adapters(ALL_SEGMENTS, adapter_weights(step_index + 1, strength))
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
    strength=1.0,
    width=512,
    height=512,
):
    text_emb, neg_emb = _encode_for_pipe(pipe, prompt)

    text_emb = text_emb.expand(batch_size, -1, -1)
    neg_emb = neg_emb.expand(batch_size, -1, -1)

    if use_lora:
        pipe.enable_lora()
        pipe.set_adapters(ALL_SEGMENTS, adapter_weights(0, strength))
    else:
        pipe.disable_lora()

    # pyrefly: ignore [unsupported-operation]
    seeds = [seed + i for i in range(batch_size)]
    generators = [torch.Generator(device="cuda").manual_seed(s) for s in seeds]

    return pipe(
        prompt_embeds=text_emb,
        negative_prompt_embeds=neg_emb,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generators,
        callback_on_step_end=make_segment_callback(
            use_lora, num_inference_steps, strength
        ),
        callback_on_step_end_tensor_inputs=["latents"],
    ).images


def infer_with_evolution(
    pipe,
    prompt,
    use_lora=True,
    num_inference_steps=30,
    guidance_scale=7.0,
    seed=None,
    strength=1.0,
    width=512,
    height=512,
):
    text_emb, neg_emb = _encode_for_pipe(pipe, prompt)

    if use_lora:
        pipe.enable_lora()
        pipe.set_adapters(ALL_SEGMENTS, adapter_weights(0, strength))
    else:
        pipe.disable_lora()

    captured_latents = []
    segment_cb = make_segment_callback(use_lora, num_inference_steps, strength)

    def callback(p, step_index, timestep, callback_kwargs):
        result = segment_cb(p, step_index, timestep, callback_kwargs)
        if step_index % 2 == 0:
            captured_latents.append(callback_kwargs["latents"][:1].clone())
        return result

    # pyrefly: ignore [bad-argument-type]
    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        prompt_embeds=text_emb,
        negative_prompt_embeds=neg_emb,
        width=width,
        height=height,
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


def make_two_row_grid(top: list[Image.Image], bottom: list[Image.Image]) -> Image.Image:
    cols = max(len(top), len(bottom))
    w, h = top[0].size

    def pad_row(frames, target_cols):
        blank = Image.new("RGB", (w, h), color=(0, 0, 0))
        return frames + [blank] * (target_cols - len(frames))

    top_row = pad_row(top, cols)
    bottom_row = pad_row(bottom, cols)

    grid = Image.new("RGB", size=(cols * w, 2 * h))
    for col, img in enumerate(top_row):
        grid.paste(img, (col * w, 0))
    for col, img in enumerate(bottom_row):
        grid.paste(img, (col * w, h))
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="A woman with long, wavy pink hair is shown in profile",
    )
    parser.add_argument("--output", "-o", type=str, default="output.png")
    parser.add_argument("--evolution-output", "-e", type=str, default="evolution.png")
    parser.add_argument("--steps", "-s", type=int, default=30)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--guidance", "-g", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    print("Loading model...")
    pipe = load_pipe()

    print(f"Generating: {args.prompt}")

    shared = dict(
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
    )

    lora_final, lora_frames = infer_with_evolution(
        pipe,
        args.prompt,
        use_lora=True,
        seed=args.seed,
        strength=args.strength,
        **shared,
    )
    base_final, base_frames = infer_with_evolution(
        pipe, args.prompt, use_lora=False, seed=args.seed, **shared
    )
    lora_rest = infer_batch(
        pipe,
        args.prompt,
        use_lora=True,
        seed=args.seed + 1,
        batch_size=2,
        strength=args.strength,
        **shared,
    )
    base_rest = infer_batch(
        pipe, args.prompt, use_lora=False, seed=args.seed + 1, batch_size=2, **shared
    )

    grid = make_grid(
        [lora_final] + lora_rest + [base_final] + base_rest, rows=2, cols=3
    )
    grid.save(args.output)
    print(f"Saved to {args.output}")

    evolution_grid = make_two_row_grid(lora_frames, base_frames)
    evolution_grid.save(args.evolution_output)
    print(f"Evolution saved to {args.evolution_output}")


if __name__ == "__main__":
    main()
