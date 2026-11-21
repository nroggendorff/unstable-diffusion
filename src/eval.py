import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .config import default_output_dir
from .encoding import encode_for_sampling
from .inference import ADAPTER_BASE_PATH, load_pipe, make_two_row_grid
from .model import NUM_INFERENCE_STEPS
from .sampler import (
    COND_HI,
    COND_LO,
    NEGATIVE_PROMPT,
    PG_HI,
    PG_LO,
    PG_SPACES,
    anchor_generate,
)

CLIP_ID = "openai/clip-vit-base-patch32"

TEXTUAL_PROMPTS = [
    "an image with large text",
    "a poster with a title",
    "a logo",
    "a video game cover",
    "a magazine cover",
    "a manga page with speech bubbles",
    "a comic strip",
    "a calendar",
    "a birthday card with handwritten text",
    "typography",
]
PLAIN_PROMPTS = [
    "an anime illustration of a character",
    "an anime girl",
    "a drawing of a person",
    "a painting of a landscape",
    "a watercolor portrait",
    "a fantasy illustration",
    "a group of anime characters",
    "a sketch",
]
TEXTUAL_THRESHOLD = 0.3

_UNRECORDED_NEGATIVE = (
    "nsfw, explicit, questionable, sensitive, suggestive, underwear, panties, "
    "lingerie, cleavage, nude, ugly, low quality, lowres, worst quality, "
    "text, watermark, comic"
)

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default=ADAPTER_BASE_PATH)
    parser.add_argument("--output_dir", type=str, default=default_output_dir())
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--basin_seeds", type=int, default=4)
    parser.add_argument("--seed_base", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--basin_guidance", type=float, default=1.5)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=1216)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--cutover", type=float, default=0.8)
    parser.add_argument("--sigma_cutover", type=float, default=0.8)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--latent_lf", type=float, default=1.0)
    parser.add_argument("--cond_sigma", type=float, default=0.0)
    parser.add_argument("--cond_lo", type=float, default=COND_LO)
    parser.add_argument("--cond_hi", type=float, default=COND_HI)
    parser.add_argument("--pg_strength", type=float, default=0.0)
    parser.add_argument("--pg_lo", type=float, default=PG_LO)
    parser.add_argument("--pg_hi", type=float, default=PG_HI)
    parser.add_argument("--pg_space", choices=PG_SPACES, default="latent")
    parser.add_argument("--prompt_limit", type=int, default=0)
    parser.add_argument("--no_contact_sheets", action="store_true")
    parser.add_argument(
        "--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"), default=None
    )
    parser.add_argument("--base_from", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default=NEGATIVE_PROMPT)
    return parser


def _args() -> argparse.Namespace:
    return _parser().parse_args()


def _with_defaults(args: dict) -> dict:
    return {
        **vars(_parser().parse_args([])),
        "negative_prompt": _UNRECORDED_NEGATIVE,
        **args,
    }


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


@torch.no_grad()
def text_probe(clip, processor) -> tuple[torch.Tensor, float]:
    inputs = processor(
        text=TEXTUAL_PROMPTS + PLAIN_PROMPTS,
        images=[Image.new("RGB", (224, 224))],
        return_tensors="pt",
        padding=True,
    ).to(clip.device)
    probe = F.normalize(clip(**inputs).text_embeds.float(), dim=-1)
    return probe, clip.logit_scale.exp().item()


def textual_probability(
    features: torch.Tensor,
    probe: torch.Tensor,
    scale: float,
    textual: int = len(TEXTUAL_PROMPTS),
) -> torch.Tensor:
    probs = (scale * features.float() @ probe.float().to(features.device).T).softmax(-1)
    return probs[..., :textual].sum(-1)


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


def mean_se(values: torch.Tensor) -> tuple[float, float]:
    values = values.flatten().double()
    n = values.numel()
    se = (values.std(unbiased=True) / n**0.5).item() if n > 1 else float("nan")
    return values.mean().item(), se


def paired_deltas(
    candidate: list[torch.Tensor],
    baseline: list[torch.Tensor],
    basin: torch.Tensor,
    text_features: torch.Tensor,
    probe: tuple[torch.Tensor, float] | None = None,
) -> dict:
    fc, fb = torch.stack(candidate).float(), torch.stack(baseline).float()
    text = text_features.float()[:, None, :].to(fc.device)
    basin = basin.float().to(fc.device)
    rows = {
        "angular_spread": torch.tensor(
            [angular_spread(c) - angular_spread(b) for c, b in zip(candidate, baseline)]
        ),
        "radius": (fb @ basin) - (fc @ basin),
        "adherence": (fc * text).sum(-1) - (fb * text).sum(-1),
        "displacement": 1.0 - (fc * fb).sum(-1),
    }
    if probe is not None:
        pc, pb = textual_probability(fc, *probe), textual_probability(fb, *probe)
        plain = (pc <= TEXTUAL_THRESHOLD) & (pb <= TEXTUAL_THRESHOLD)
        rows["textual"] = pc - pb
        rows["radius_plain"] = rows["radius"][plain]
    deltas = {
        name: dict(zip(("delta", "se"), mean_se(values)))
        for name, values in rows.items()
    }
    if probe is not None:
        deltas["radius_plain"]["n"] = int(rows["radius_plain"].numel())
    return deltas


def print_deltas(title: str, deltas: dict) -> None:
    print(f"\n{title} (paired on identical seeds; +- 1 SE):")
    for name, row in deltas.items():
        t = row["delta"] / row["se"] if row["se"] > 0 else float("nan")
        sign = "" if name == "displacement" else "+"
        tail = "" if name == "displacement" else f"  t={t:+.1f}"
        if "n" in row:
            tail += f"  ({row['n']} pairs with neither image textual)"
        print(f"  {name:15s} {row['delta']:{sign}.4f} +- {row['se']:.4f}{tail}")


def black_frames(images) -> int:
    return sum(
        float(np.asarray(image, dtype=np.float32).std()) < 2.0 for image in images
    )


def _features_path(path: str) -> str:
    return path if path.endswith(".pt") else os.path.join(path, "eval", "features.pt")


_UNPAIRED_ARGS = ("adapter_path", "output_dir", "compare", "no_contact_sheets")
_BASE_FREE_ARGS = _UNPAIRED_ARGS + ("base_from", "strength")


def reusable_base(prior: dict, args: dict, prompts: list, seed_list: list) -> dict:
    if prior["prompts"] != list(prompts) or prior["seed_list"] != seed_list:
        raise ValueError("--base_from eval used different prompts or seeds")
    prior_args, args = _with_defaults(prior["args"]), _with_defaults(args)
    differing = sorted(
        k
        for k in set(prior_args) | set(args)
        if k not in _BASE_FREE_ARGS and prior_args.get(k) != args.get(k)
    )
    if differing:
        raise ValueError(f"--base_from eval differs in {', '.join(differing)}")
    return prior


def _base_row(sheet_path: str, size: tuple[int, int], count: int) -> list:
    if not os.path.exists(sheet_path):
        return []
    w, h = size
    with Image.open(sheet_path) as sheet:
        return [sheet.crop((i * w, h, (i + 1) * w, 2 * h)).copy() for i in range(count)]


def textual_summary(
    features: list[torch.Tensor], probe: tuple[torch.Tensor, float]
) -> dict:
    p = textual_probability(torch.cat([f.float() for f in features]), *probe)
    return {
        "textual": p.mean().item(),
        "textual_rate": (p > TEXTUAL_THRESHOLD).float().mean().item(),
    }


def _textual_line(label: str, row: dict) -> str:
    return (
        f"{label:>5}  textual={row['textual']:.3f}  "
        f"{row['textual_rate']:.0%} over {TEXTUAL_THRESHOLD}"
    )


def compare(baseline_path: str, candidate_path: str) -> dict:
    baseline = torch.load(_features_path(baseline_path), weights_only=True)
    candidate = torch.load(_features_path(candidate_path), weights_only=True)
    for key in ("prompts", "seed_list"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Cannot pair: the two evals differ in {key}")
    baseline_args = _with_defaults(baseline["args"])
    candidate_args = _with_defaults(candidate["args"])
    differing = sorted(
        k
        for k in set(baseline_args) | set(candidate_args)
        if k not in _UNPAIRED_ARGS + ("base_from",)
        and baseline_args.get(k) != candidate_args.get(k)
    )
    if differing:
        print(
            "Settings that differ:",
            ", ".join(
                f"{k} {baseline_args.get(k)} -> {candidate_args.get(k)}"
                for k in differing
            ),
        )
    clip = CLIPModel.from_pretrained(CLIP_ID).eval()
    probe = text_probe(clip, CLIPProcessor.from_pretrained(CLIP_ID))
    del clip
    for label, features in (
        ("baseline", baseline["lora"]),
        ("candidate", candidate["lora"]),
    ):
        print(_textual_line(label, textual_summary(features, probe)))
    deltas = paired_deltas(
        candidate["lora"], baseline["lora"], baseline["basin"], baseline["text"], probe
    )
    print_deltas(f"candidate - baseline: {candidate_path} vs {baseline_path}", deltas)
    return deltas


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
    if args.compare:
        compare(*args.compare)
        return
    device = torch.device("cuda")

    prompts = PROMPTS[: args.prompt_limit] if args.prompt_limit > 0 else PROMPTS

    eval_dir = os.path.join(args.output_dir, "eval")
    sheet_dir = os.path.join(eval_dir, "sheets")
    os.makedirs(sheet_dir, exist_ok=True)

    print("Loading pipeline...")
    pipe = load_pipe(args.adapter_path)

    clip = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_ID)
    probe = text_probe(clip, processor)

    encoders = [pipe.text_encoder, pipe.text_encoder_2]
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]

    encoded = {
        prompt: encode_for_sampling(
            prompt, args.negative_prompt, encoders, tokenizers, device
        )
        for prompt in prompts
    }

    shared = dict(
        strength=args.strength,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        cutover=args.cutover,
        sigma_cutover=args.sigma_cutover,
        latent_lf=args.latent_lf,
    )

    black = {"basin": 0, "lora": 0, "base": 0}

    seed_list = [
        [args.seed_base + index * 100 + s for s in range(args.seeds)]
        for index in range(len(prompts))
    ]

    prior = None
    prior_sheets = ""
    if args.base_from:
        prior_path = _features_path(args.base_from)
        prior = reusable_base(
            torch.load(prior_path, weights_only=True), vars(args), prompts, seed_list
        )
        prior_sheets = os.path.join(os.path.dirname(prior_path), "sheets")
        print(f"Reusing the basin and base passes of {prior_path}")
        basin = prior["basin"].to(device)
    else:
        print(f"Basin pass ({args.basin_seeds} seeds x {len(prompts)} prompts)...")
        basin_features = []
        for index, prompt in enumerate(tqdm(prompts, desc="basin")):
            positive, positive_pooled, negative, negative_pooled = encoded[prompt]
            images = anchor_generate(
                pipe,
                positive,
                positive_pooled,
                negative,
                negative_pooled,
                seeds=[
                    args.seed_base + index * 100 + s for s in range(args.basin_seeds)
                ],
                use_lora=False,
                guidance=args.basin_guidance,
                alpha=0.0,
                sigma=0.0,
                offset=0.0,
                gamma=1.0,
                **shared,
            )
            black["basin"] += black_frames(images)
            features, _ = embed(clip, processor, images, prompt, device)
            basin_features.append(features)
            del images, features

        basin = F.normalize(torch.cat(basin_features).mean(0), dim=-1)
        del basin_features

    results: dict[str, list] = {}
    sheets: dict[int, list] = {}
    text_features_by_prompt: dict[int, torch.Tensor] = {}

    passes = (
        [(True, "lora")] if prior is not None else [(True, "lora"), (False, "base")]
    )
    for use_lora, label in passes:
        print(f"\nSampling pass: {label} ({args.seeds} seeds x {len(prompts)} prompts)")
        per_prompt = []

        for index, prompt in enumerate(tqdm(prompts, desc=label)):
            positive, positive_pooled, negative, negative_pooled = encoded[prompt]
            images = anchor_generate(
                pipe,
                positive,
                positive_pooled,
                negative,
                negative_pooled,
                seeds=seed_list[index],
                use_lora=use_lora,
                guidance=args.guidance,
                alpha=args.alpha,
                sigma=args.sigma,
                offset=args.offset,
                gamma=args.gamma,
                cond_sigma=args.cond_sigma,
                cond_lo=args.cond_lo,
                cond_hi=args.cond_hi,
                pg_strength=args.pg_strength,
                pg_lo=args.pg_lo,
                pg_hi=args.pg_hi,
                pg_space=args.pg_space,
                **shared,
            )

            black[label] += black_frames(images)
            features, text_feature = embed(clip, processor, images, prompt, device)
            text_features_by_prompt[index] = text_feature
            per_prompt.append({"prompt": prompt, "features": features})

            if args.no_contact_sheets:
                del images
            elif prior is not None:
                base_row = _base_row(
                    os.path.join(prior_sheets, f"{index:02d}.png"),
                    images[0].size,
                    len(images),
                )
                sheet = make_two_row_grid(images, base_row)
                sheet.save(os.path.join(sheet_dir, f"{index:02d}.png"))
                del images, sheet, base_row
            elif use_lora:
                sheets[index] = images
            else:
                sheet = make_two_row_grid(sheets.pop(index), images)
                sheet.save(os.path.join(sheet_dir, f"{index:02d}.png"))
                del images, sheet

        results[label] = per_prompt

    if prior is not None:
        results["base"] = [
            {"prompt": prompt, "features": features.to(device)}
            for prompt, features in zip(prompts, prior["base"])
        ]

    text_features = torch.stack(
        [text_features_by_prompt[i] for i in range(len(prompts))]
    )

    lora_features = [entry["features"] for entry in results["lora"]]
    base_features = [entry["features"] for entry in results["base"]]
    metrics = {
        "args": vars(args),
        "prompts": len(prompts),
        "seeds": args.seeds,
        "lora": {
            **summarize(results["lora"], basin, text_features),
            **textual_summary(lora_features, probe),
        },
        "base": {
            **summarize(results["base"], basin, text_features),
            **textual_summary(base_features, probe),
        },
        "lora_vs_base": paired_deltas(
            lora_features, base_features, basin, text_features, probe
        ),
        "black_frames": black,
    }

    torch.save(
        {
            "args": vars(args),
            "prompts": list(prompts),
            "seed_list": seed_list,
            "basin": basin.cpu(),
            "text": text_features.cpu(),
            "lora": [f.cpu() for f in lora_features],
            "base": [f.cpu() for f in base_features],
        },
        os.path.join(eval_dir, "features.pt"),
    )

    metrics_path = os.path.join(eval_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    for label in ("lora", "base"):
        row = metrics[label]
        print(
            f"{label:>5}  angular_spread={row['angular_spread']:.4f}  "
            f"radius={row['radius']:.4f}  adherence={row['adherence']:.4f}"
        )
        print(_textual_line("", row))

    print_deltas("lora - base", metrics["lora_vs_base"])
    if any(black.values()):
        print(
            f"\nWARNING: black frames {black}. A black image is a NaN decode and "
            "skews every metric it touches; check the sheets and rerun."
        )

    print(f"\nWrote {metrics_path}")
    if not args.no_contact_sheets:
        print(f"Contact sheets in {sheet_dir} (top row LoRA, bottom row base).")
    print(
        "None of these numbers say whether a sample is any good. "
        "Angular spread and radius both rise on garbage, and radius rises on text, "
        "posters and comic pages (textual, radius_plain); adherence is the only "
        "brake. Look at the sheets."
    )


if __name__ == "__main__":
    main()
