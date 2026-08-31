import argparse
import json
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .config import default_output_dir
from .encoding import encode_prompt
from .inference import ADAPTER_BASE_PATH, load_pipe, make_two_row_grid
from .model import NUM_INFERENCE_STEPS
from .sampler import anchor_generate

CLIP_ID = "openai/clip-vit-base-patch32"

NEGATIVE_PROMPT = "ugly, low quality"

PROMPTS = [
    "a girl",
    "a boy standing",
    "a portrait",
    "two people talking",
    "a girl with short hair looking at the viewer",
    "a boy in a school uniform on a rooftop",
    "a woman with long, wavy pink hair is shown in profile",
    "a swordsman in the rain",
    "a witch reading by candlelight",
    "a cat sitting on a windowsill",
    "a city street at dusk",
    "a quiet room with afternoon light",
    "a girl holding an umbrella in heavy snow",
    "a musician on an empty stage",
    "a fisherman mending nets at dawn",
    "an old woman tending a garden",
    "a knight resting against a broken wall",
    "a child chasing fireflies",
    "a mechanic under a half-built machine",
    "a dancer mid-turn",
    "a scholar surrounded by stacked books",
    "a traveller at a crossroads",
    "a girl asleep on a train",
    "a lighthouse keeper in a storm",
]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default=ADAPTER_BASE_PATH)
    parser.add_argument("--output_dir", type=str, default=default_output_dir())
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--basin_seeds", type=int, default=2)
    parser.add_argument("--seed_base", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--basin_guidance", type=float, default=1.5)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--cutover", type=float, default=0.8)
    parser.add_argument("--sigma_cutover", type=float, default=0.8)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--prompt_limit", type=int, default=0)
    parser.add_argument("--no_contact_sheets", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def embed(clip, processor, images, prompt, device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = processor(
        text=[prompt],
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    output = clip(**inputs)
    return (
        F.normalize(output.image_embeds.float(), dim=-1),
        F.normalize(output.text_embeds.float(), dim=-1)[0],
    )


def angular_spread(features: torch.Tensor) -> float:
    n = features.shape[0]
    if n < 2:
        return 0.0
    sim = features @ features.T
    upper = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
    return (1.0 - sim[upper]).mean().item()


def radius(features: torch.Tensor, basin: torch.Tensor) -> float:
    return (1.0 - features @ basin).mean().item()


def adherence(features: torch.Tensor, text_feature: torch.Tensor) -> float:
    return (features @ text_feature).mean().item()


def summarize(
    per_prompt: list[dict], basin: torch.Tensor, text_features: torch.Tensor
) -> dict:
    spreads, radii, adherences = [], [], []

    for index, entry in enumerate(per_prompt):
        features = entry["features"]
        entry["angular_spread"] = angular_spread(features)
        entry["radius"] = radius(features, basin)
        entry["adherence"] = adherence(features, text_features[index])

        spreads.append(entry["angular_spread"])
        radii.append(entry["radius"])
        adherences.append(entry["adherence"])

    return {
        "angular_spread": sum(spreads) / len(spreads),
        "radius": sum(radii) / len(radii),
        "adherence": sum(adherences) / len(adherences),
        "per_prompt": [
            {
                "prompt": entry["prompt"],
                "angular_spread": entry["angular_spread"],
                "radius": entry["radius"],
                "adherence": entry["adherence"],
            }
            for entry in per_prompt
        ],
    }


def main():
    args = _args()
    device = torch.device("cuda")

    prompts = PROMPTS[: args.prompt_limit] if args.prompt_limit > 0 else PROMPTS

    eval_dir = os.path.join(args.output_dir, "eval")
    sheet_dir = os.path.join(eval_dir, "sheets")
    os.makedirs(sheet_dir, exist_ok=True)

    print("Loading pipeline...")
    pipe = load_pipe(args.adapter_path)

    clip = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_ID)

    encoded = {}
    for prompt in prompts:
        positive, _ = encode_prompt(prompt, pipe.text_encoder, pipe.tokenizer, device)
        negative, _ = encode_prompt(
            NEGATIVE_PROMPT, pipe.text_encoder, pipe.tokenizer, device
        )
        encoded[prompt] = (
            positive.to(dtype=torch.float16),
            negative.to(dtype=torch.float16),
        )

    shared = dict(
        strength=args.strength,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        cutover=args.cutover,
        sigma_cutover=args.sigma_cutover,
    )

    print(f"Basin pass ({args.basin_seeds} seeds x {len(prompts)} prompts)...")
    basin_features = []
    for index, prompt in enumerate(tqdm(prompts, desc="basin")):
        positive, negative = encoded[prompt]
        images = anchor_generate(
            pipe,
            positive,
            negative,
            seeds=[args.seed_base + index * 100 + s for s in range(args.basin_seeds)],
            use_lora=False,
            guidance=args.basin_guidance,
            alpha=0.0,
            sigma=0.0,
            offset=0.0,
            gamma=1.0,
            **shared,
        )
        features, _ = embed(clip, processor, images, prompt, device)
        basin_features.append(features)
        del images, features

    basin = F.normalize(torch.cat(basin_features).mean(0), dim=-1)
    del basin_features

    results: dict[str, list] = {}
    sheets: dict[int, list] = {}
    text_features_by_prompt: dict[int, torch.Tensor] = {}

    for use_lora, label in ((True, "lora"), (False, "base")):
        print(f"\nSampling pass: {label} ({args.seeds} seeds x {len(prompts)} prompts)")
        per_prompt = []

        for index, prompt in enumerate(tqdm(prompts, desc=label)):
            positive, negative = encoded[prompt]
            images = anchor_generate(
                pipe,
                positive,
                negative,
                seeds=[args.seed_base + index * 100 + s for s in range(args.seeds)],
                use_lora=use_lora,
                guidance=args.guidance,
                alpha=args.alpha,
                sigma=args.sigma,
                offset=args.offset,
                gamma=args.gamma,
                **shared,
            )

            features, text_feature = embed(clip, processor, images, prompt, device)
            text_features_by_prompt[index] = text_feature
            per_prompt.append({"prompt": prompt, "features": features})

            if args.no_contact_sheets:
                del images
            elif use_lora:
                sheets[index] = images
            else:
                sheet = make_two_row_grid(sheets.pop(index), images)
                sheet.save(os.path.join(sheet_dir, f"{index:02d}.png"))
                del images, sheet

        results[label] = per_prompt

    text_features = torch.stack(
        [text_features_by_prompt[i] for i in range(len(prompts))]
    )

    metrics = {
        "args": vars(args),
        "prompts": len(prompts),
        "seeds": args.seeds,
        "lora": summarize(results["lora"], basin, text_features),
        "base": summarize(results["base"], basin, text_features),
    }

    metrics_path = os.path.join(eval_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    for label in ("lora", "base"):
        row = metrics[label]
        print(
            f"{label:>5}  angular_spread={row['angular_spread']:.4f}  "
            f"radius={row['radius']:.4f}  adherence={row['adherence']:.4f}"
        )

    print(f"\nWrote {metrics_path}")
    if not args.no_contact_sheets:
        print(f"Contact sheets in {sheet_dir} (top row LoRA, bottom row base).")
    print(
        "None of these numbers say whether a sample is any good. "
        "Angular spread and radius both rise on garbage; adherence is the only "
        "brake. Look at the sheets."
    )


if __name__ == "__main__":
    main()
