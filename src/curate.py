import argparse
import os
import random

import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .cache import load_cache, save_cache
from .eval import CLIP_ID, TEXTUAL_THRESHOLD, text_probe, textual_probability
from .model import DEVICE, load_vae
from .scorer import ImageScorer

CELL = (320, 448)


def scores_path(cache_file: str) -> str:
    return cache_file + ".scores.pt"


def basin_radius(features: torch.Tensor, basin: torch.Tensor) -> torch.Tensor:
    return 1.0 - features.float() @ basin.float()


def dedup_order(
    features: torch.Tensor,
    order: torch.Tensor,
    threshold: float,
    limit: int | None = None,
    block: int = 1024,
) -> tuple[torch.Tensor, int]:
    if threshold >= 1.0:
        head = order if limit is None else order[:limit]
        return head, head.numel()
    feats = features.float()
    kept: list[int] = []
    kept_feats = feats.new_zeros((0, feats.shape[1]))
    examined = 0
    for start in range(0, order.numel(), block):
        idx = order[start : start + block]
        candidates = feats[idx]
        clash = (candidates @ kept_feats.T > threshold).any(1)
        within = candidates @ candidates.T > threshold
        accepted: list[int] = []
        for j in range(idx.numel()):
            if limit is not None and len(kept) + len(accepted) >= limit:
                break
            examined += 1
            if clash[j] or (accepted and bool(within[j, accepted].any())):
                continue
            accepted.append(j)
        kept.extend(idx[accepted].tolist())
        kept_feats = torch.cat([kept_feats, candidates[accepted]])
        if limit is not None and len(kept) >= limit:
            break
    return torch.tensor(kept, dtype=torch.long), examined


def select(
    radius: torch.Tensor,
    quality: torch.Tensor,
    features: torch.Tensor,
    count: int,
    floor: float,
    threshold: float,
    textual: torch.Tensor | None = None,
    text_max: float = 1.0,
) -> tuple[torch.Tensor, int]:
    keep = quality >= floor
    if textual is not None:
        keep = keep & (textual <= text_max)
    eligible = torch.nonzero(keep).flatten()
    order = eligible[radius[eligible].argsort(descending=True)]
    return dedup_order(features, order, threshold, limit=count)


@torch.no_grad()
def decode(vae, latents: torch.Tensor) -> torch.Tensor:
    scaled = latents.to(DEVICE, dtype=torch.float16) / vae.config.scaling_factor
    return vae.decode(scaled).sample.float().clamp(-1.0, 1.0)


def to_image(pixels: torch.Tensor) -> Image.Image:
    array = ((pixels[0] + 1.0) * 127.5).round().clamp(0, 255).byte()
    return Image.fromarray(array.permute(1, 2, 0).cpu().numpy())


@torch.no_grad()
def score_cache(cache_file: str, vae, clip, processor, laion, batch: int) -> dict:
    cached = load_cache(cache_file)
    clip_features, quality, pending = [], [], []

    def flush():
        inputs = processor(
            text=[""], images=pending, return_tensors="pt", padding=True
        ).to(DEVICE)
        features = clip(**inputs).image_embeds
        clip_features.append(torch.nn.functional.normalize(features.float(), dim=-1))
        pending.clear()

    for item in tqdm(cached, desc=os.path.basename(cache_file)):
        pixels = decode(vae, item["latents"])
        quality.append(laion.quality(laion.image_embeds(pixels)).float())
        pending.append(to_image(pixels))
        if len(pending) == batch:
            flush()
    if pending:
        flush()

    scores = {
        "clip": torch.cat(clip_features).half().cpu(),
        "laion": torch.cat(quality).cpu(),
        "bucket": [item["bucket"] for item in cached],
    }
    torch.save(scores, scores_path(cache_file))
    del cached
    return scores


def contact_sheet(rows: list[list[Image.Image]]) -> Image.Image:
    width, height = CELL
    cols = max(len(row) for row in rows)
    sheet = Image.new("RGB", (cols * width, len(rows) * height))
    for r, row in enumerate(rows):
        for c, image in enumerate(row):
            thumb = ImageOps.pad(image, CELL, color=(0, 0, 0))
            sheet.paste(thumb, (c * width, r * height))
    return sheet


def percentiles(values: torch.Tensor) -> str:
    qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    return " / ".join(f"{v:.3f}" for v in torch.quantile(values.float(), qs).tolist())


def run_score(args):
    vae = load_vae()
    clip = CLIPModel.from_pretrained(CLIP_ID).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_ID)
    laion = ImageScorer("laion").to(DEVICE)
    for cache_file in args.caches:
        if os.path.exists(scores_path(cache_file)):
            print(f"{cache_file}: already scored")
            continue
        scores = score_cache(cache_file, vae, clip, processor, laion, args.batch)
        print(
            f"{cache_file}: {len(scores['laion'])} scored, "
            f"LAION p5/p25/p50/p75/p95 {percentiles(scores['laion'])}"
        )


def run_select(args):
    basin = torch.load(args.basin, weights_only=True)["basin"].float()
    loaded = [torch.load(scores_path(c), weights_only=True) for c in args.caches]
    owner = torch.cat(
        [
            torch.full((len(s["laion"]),), i, dtype=torch.long)
            for i, s in enumerate(loaded)
        ]
    )
    local = torch.cat([torch.arange(len(s["laion"])) for s in loaded])
    features = torch.cat([s["clip"] for s in loaded]).float()
    quality = torch.cat([s["laion"] for s in loaded]).float()
    buckets = [b for s in loaded for b in s["bucket"]]
    radius = basin_radius(features, basin)

    clip = CLIPModel.from_pretrained(CLIP_ID).eval()
    probe = text_probe(clip, CLIPProcessor.from_pretrained(CLIP_ID))
    del clip
    textual = textual_probability(features, *probe)
    plain = textual <= args.text_max

    eligible = (quality >= args.laion_floor) & plain
    chosen, examined = select(
        radius,
        quality,
        features,
        args.count,
        args.laion_floor,
        args.dedup,
        textual,
        args.text_max,
    )
    cutoff = radius[chosen].min()
    far = radius >= cutoff

    print(f"Pool: {len(quality)} samples from {len(args.caches)} caches")
    print(f"  LAION     p5/p25/p50/p75/p95 {percentiles(quality)}")
    print(f"  radius    p5/p25/p50/p75/p95 {percentiles(radius)}")
    print(f"  p_text    p5/p25/p50/p75/p95 {percentiles(textual)}")
    print(
        f"  {int((quality < args.laion_floor).sum())} below LAION {args.laion_floor}, "
        f"{int((~plain).sum())} above p_text {args.text_max}, {int(eligible.sum())} eligible; "
        f"filling the selection skipped {examined - len(chosen)} of {examined} as "
        f"near-duplicates (CLIP cos > {args.dedup}) of a higher-radius pick"
    )
    print(
        f"  at radius >= the cutoff: {int(far.sum())} images, "
        f"{int((far & (quality < args.laion_floor)).sum())} below the floor, "
        f"{int((far & ~plain).sum())} textual"
    )
    print(f"Selected {len(chosen)} (asked {args.count})")
    print(f"  LAION     p5/p25/p50/p75/p95 {percentiles(quality[chosen])}")
    print(f"  radius    p5/p25/p50/p75/p95 {percentiles(radius[chosen])}")
    print(f"  p_text    p5/p25/p50/p75/p95 {percentiles(textual[chosen])}")
    print(
        f"  radius cutoff {cutoff:.3f}; pool mean {radius.mean():.3f}, selected mean {radius[chosen].mean():.3f}"
    )
    pool_counts: dict = {}
    for b in buckets:
        pool_counts[b] = pool_counts.get(b, 0) + 1
    chosen_counts: dict = {}
    for i in chosen.tolist():
        chosen_counts[buckets[i]] = chosen_counts.get(buckets[i], 0) + 1
    print("  bucket      pool  selected")
    for bucket, n in sorted(pool_counts.items(), key=lambda kv: -kv[1]):
        print(
            f"  {bucket[0]:>4}x{bucket[1]:<4} {n:6d} {chosen_counts.get(bucket, 0):8d}"
        )
    for i, cache_file in enumerate(args.caches):
        print(
            f"  from {os.path.basename(cache_file)}: {int((owner[chosen] == i).sum())}"
        )

    if args.dry_run:
        return

    rng = random.Random(0)
    chosen_list = chosen.tolist()
    rejected = sorted(
        set(torch.nonzero(eligible).flatten().tolist()) - set(chosen_list)
    )
    below = torch.nonzero(far & (quality < args.laion_floor)).flatten().tolist()
    wordy = torch.nonzero(far & ~plain).flatten().tolist()
    sheet_rows = {
        "top": chosen_list[:8],
        "selected": rng.sample(chosen_list, min(8, len(chosen_list))),
        "rejected": rng.sample(rejected, min(8, len(rejected))),
        "far_below_floor": rng.sample(below, min(8, len(below))),
        "far_textual": rng.sample(wordy, min(8, len(wordy))),
    }
    wanted = {i for row in sheet_rows.values() for i in row}

    vae = load_vae() if args.sheet else None
    images: dict = {}
    selected_items = []
    chosen_set = set(chosen_list)
    for i, cache_file in enumerate(args.caches):
        cached = load_cache(cache_file)
        for position in torch.nonzero(owner == i).flatten().tolist():
            if position in chosen_set:
                selected_items.append(cached[int(local[position])])
            if vae is not None and position in wanted:
                images[position] = to_image(
                    decode(vae, cached[int(local[position])]["latents"])
                )
        del cached
        print(
            f"Collected {len(selected_items)} items after {os.path.basename(cache_file)}"
        )

    save_cache(selected_items, args.out)
    print(f"Saved curated cache ({len(selected_items)} samples) to {args.out}")
    if args.sheet:
        sheet = contact_sheet([[images[i] for i in row] for row in sheet_rows.values()])
        sheet.save(args.sheet)
        print(f"Wrote {args.sheet}: rows are {', '.join(sheet_rows)}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score")
    score.add_argument("--caches", nargs="+", required=True)
    score.add_argument("--batch", type=int, default=32)

    choose = sub.add_parser("select")
    choose.add_argument("--caches", nargs="+", required=True)
    choose.add_argument("--basin", type=str, required=True)
    choose.add_argument("--count", type=int, default=6000)
    choose.add_argument("--laion_floor", type=float, default=5.0)
    choose.add_argument("--dedup", type=float, default=0.97)
    choose.add_argument("--text_max", type=float, default=TEXTUAL_THRESHOLD)
    choose.add_argument("--out", type=str, default="")
    choose.add_argument("--sheet", type=str, default="")
    choose.add_argument("--dry_run", type=int, default=0)

    args = parser.parse_args()
    if args.command == "score":
        run_score(args)
    else:
        if not args.dry_run and not args.out:
            parser.error("select needs --out unless --dry_run 1")
        run_select(args)


if __name__ == "__main__":
    main()
