import argparse
import json
import os
import random

import torch
from tqdm import tqdm

from .config import default_output_dir
from .model import load_model, DEVICE, SEGMENT_TIMESTEP_RANGES
from .dataset import get_samples, prepare_sample
from .encoding import encode_prompt, subset_caption, CHUNK_TOKENS, MAX_CHUNKS
from .encoder import compute_perceptual_discrepancy, SubjectMaskBuilder
from .encoder import CrossAttentionCapture
from .encoder.mask_builder import normalize_mask
from .scheduler import SpatiallyVaryingDDPMScheduler, compute_spatial_noise_scale
from .scheduler import pyramid_noise
from .cache import blend_alpha, blend_masks, stack_added_cond, BucketSampler
from .train import DISCREPANCY_MAX_T

CONDITIONS = ["full", "subset", "shuffled"]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--mini_batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subset_min", type=float, default=0.15)
    parser.add_argument("--mask_blur_sigma_start", type=float, default=7.0)
    parser.add_argument("--mask_blur_sigma_end", type=float, default=1.0)
    parser.add_argument("--mask_min_value", type=float, default=0.0)
    parser.add_argument("--mask_gain", type=float, default=1.5)
    parser.add_argument("--noise_bg_boost", type=float, default=1.5)
    parser.add_argument("--noise_t_ramp", type=float, default=0.3)
    parser.add_argument("--noise_lf_levels", type=int, default=8)
    parser.add_argument("--noise_lf_decay", type=float, default=0.66)
    parser.add_argument("--loss_bg_weight", type=float, default=0.25)
    parser.add_argument("--output_dir", type=str, default=default_output_dir())
    return parser.parse_args()


def _token_count(tokenizer, text: str) -> int:
    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=CHUNK_TOKENS * MAX_CHUNKS,
    )["input_ids"]
    return len(ids)


def _corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.flatten(1).float()
    b = b.flatten(1).float()
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    num = (a * b).sum(1)
    den = a.norm(dim=1) * b.norm(dim=1)
    return num / den.clamp(min=1e-8)


def build_probe_cache(args, vae, text_encoders, tokenizers) -> list:
    rng = random.Random(args.seed)
    content_cache: dict = {}
    cached = []

    stream = get_samples(args.cache_size, seed=args.seed)
    for sample in tqdm(stream, desc="Caching", total=args.cache_size):
        image, prompt, bucket, time_ids = prepare_sample(sample, DEVICE)
        if image is None or prompt is None:
            continue

        subset = subset_caption(prompt, rng, args.subset_min)

        with torch.no_grad():
            latents = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
            full_emb, full_pooled, full_content = encode_prompt(
                prompt, text_encoders, tokenizers, DEVICE, content_cache
            )
            sub_emb, sub_pooled, sub_content = encode_prompt(
                subset, text_encoders, tokenizers, DEVICE, content_cache
            )

        cached.append(
            {
                "latents": latents.cpu().half(),
                "text_emb": full_emb.cpu().half(),
                "pooled": full_pooled.cpu().half(),
                "time_ids": torch.tensor(time_ids, dtype=torch.float32),
                "token_content_mask": full_content.cpu(),
                "subset_text_emb": sub_emb.cpu().half(),
                "subset_pooled": sub_pooled.cpu().half(),
                "subset_token_content_mask": sub_content.cpu(),
                "full_tokens": _token_count(tokenizers[0], prompt),
                "subset_tokens": _token_count(tokenizers[0], subset),
                "full_words": len(prompt.split()),
                "subset_words": len(subset.split()),
                "bucket": bucket,
            }
        )

    return cached


def condition_mask(
    unet,
    capture,
    scheduler,
    mask_builder,
    latents,
    noise,
    t,
    text_emb,
    added_cond_kwargs,
    content,
    alpha,
    blur_sigma,
):
    spatial_size = (latents.shape[2], latents.shape[3])

    with torch.no_grad():
        uniform_noisy = scheduler.add_noise(latents, noise, t)

        capture.set_context(content, spatial_size)
        with capture:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                base_pred = unet(
                    uniform_noisy.to(dtype=torch.float16),
                    t,
                    encoder_hidden_states=text_emb,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample.float()

            raw = capture.raw_mask(spatial_size)

        attn_mask = normalize_mask(raw, capture.gain)

        denoised = scheduler.predict_x0(uniform_noisy.float(), base_pred, t)
        raw_diff = compute_perceptual_discrepancy([denoised], [latents])

        gate = (t < DISCREPANCY_MAX_T).float().view(-1, 1, 1, 1)
        blended = blend_masks(
            attn_mask, raw_diff, spatial_size, alpha * gate, gain=capture.gain
        )

    mask = mask_builder.build_mask(blended, blur_sigma)
    return raw, mask


def measure(args, unet, vae, text_encoders, tokenizers, scheduler) -> dict:
    cached = build_probe_cache(args, vae, text_encoders, tokenizers)
    sampler = BucketSampler(cached, args.mini_batch_size)
    if not sampler:
        raise RuntimeError(
            "No aspect bucket holds a full mini-batch; raise --cache_size."
        )

    print(f"Cached {len(cached)} samples; {sampler.usable} usable.")

    mask_builder = SubjectMaskBuilder(
        blur_sigma_start=args.mask_blur_sigma_start,
        blur_sigma_end=args.mask_blur_sigma_end,
        min_mask_value=args.mask_min_value,
        gain=args.mask_gain,
    )

    capture = CrossAttentionCapture(unet, gain=args.mask_gain)
    stats: dict = {}

    for segment_name, (t_low, t_high) in SEGMENT_TIMESTEP_RANGES.items():
        acc: dict = {c: {} for c in CONDITIONS}
        acc["pair"] = {}

        for step in tqdm(range(args.batches), desc=f"Probing {segment_name}"):
            items = sampler.draw(args.mini_batch_size)

            latents = torch.cat([x["latents"] for x in items]).float().to(DEVICE)
            batch = latents.shape[0]

            noise = pyramid_noise(
                latents, levels=args.noise_lf_levels, decay=args.noise_lf_decay
            )
            t = torch.randint(
                t_low, t_high + 1, (batch,), device=DEVICE, dtype=torch.long
            )

            alpha = blend_alpha(step, args.batches)
            blur_sigma = mask_builder.blur_sigma_for_step(step, args.batches)

            embeds = {}
            contents = {}
            conds = {}

            pooled, time_ids = stack_added_cond(items, DEVICE)
            subset_pooled = torch.cat([x["subset_pooled"] for x in items]).to(
                DEVICE, dtype=torch.float16
            )

            embeds["full"] = torch.cat([x["text_emb"] for x in items]).to(
                DEVICE, dtype=torch.float16
            )
            contents["full"] = torch.cat([x["token_content_mask"] for x in items]).to(
                DEVICE
            )

            embeds["subset"] = torch.cat([x["subset_text_emb"] for x in items]).to(
                DEVICE, dtype=torch.float16
            )
            contents["subset"] = torch.cat(
                [x["subset_token_content_mask"] for x in items]
            ).to(DEVICE)

            conds["full"] = {"text_embeds": pooled, "time_ids": time_ids}
            conds["subset"] = {"text_embeds": subset_pooled, "time_ids": time_ids}
            conds["shuffled"] = conds["full"]

            embeds["shuffled"] = embeds["full"]
            contents["shuffled"] = torch.stack(
                [
                    row[torch.randperm(row.shape[0], device=DEVICE)]
                    for row in contents["full"]
                ]
            )

            masks = {}

            for condition in CONDITIONS:
                raw, mask = condition_mask(
                    unet,
                    capture,
                    scheduler,
                    mask_builder,
                    latents,
                    noise,
                    t,
                    embeds[condition],
                    conds[condition],
                    contents[condition],
                    alpha,
                    blur_sigma,
                )
                masks[condition] = mask

                flat_raw = raw.flatten(1).float()
                raw_cv = flat_raw.std(1) / flat_raw.mean(1).abs().clamp(min=1e-8)

                weight = args.loss_bg_weight + mask.float().clamp(0.0, 1.0)
                weight = weight / weight.flatten(1).mean(1).clamp(min=1e-8).view(
                    -1, 1, 1, 1
                )

                noise_scale = compute_spatial_noise_scale(
                    mask,
                    t.float() / 1000.0,
                    bg_boost=args.noise_bg_boost,
                    t_ramp=args.noise_t_ramp,
                )

                token_key = "subset_tokens" if condition == "subset" else "full_tokens"
                word_key = "subset_words" if condition == "subset" else "full_words"
                n_tokens = torch.tensor(
                    [float(x[token_key]) for x in items], device=DEVICE
                )
                n_words = torch.tensor(
                    [float(x[word_key]) for x in items], device=DEVICE
                )

                record = {
                    "prompt_words": n_words,
                    "prompt_tokens": n_tokens,
                    "content_tokens": contents[condition].sum(1),
                    "content_frac": contents[condition].sum(1) / n_tokens.clamp(min=1),
                    "raw_cv": raw_cv,
                    "mask_mean": mask.flatten(1).mean(1),
                    "mask_std": mask.flatten(1).std(1),
                    "weight_std": weight.flatten(1).std(1),
                    "noise_scale_std": noise_scale.flatten(1).std(1),
                }

                for key, value in record.items():
                    acc[condition].setdefault(key, []).append(value.detach().cpu())

            pairs = {
                "corr_full_shuffled": _corr(masks["full"], masks["shuffled"]),
                "corr_full_subset": _corr(masks["full"], masks["subset"]),
            }
            for key, value in pairs.items():
                acc["pair"].setdefault(key, []).append(value.detach().cpu())

            del latents, noise, masks, embeds, contents, conds
            if step % 10 == 0:
                torch.cuda.empty_cache()

        stats[segment_name] = {
            group: {
                key: {
                    "mean": float(torch.cat(vals).mean()),
                    "std": float(torch.cat(vals).std()),
                }
                for key, vals in entries.items()
            }
            for group, entries in acc.items()
        }

    return stats


_ROWS = [
    ("prompt_words", "caption words"),
    ("prompt_tokens", "caption tokens"),
    ("content_tokens", "content tokens"),
    ("content_frac", "content fraction"),
    ("raw_cv", "raw attn CV"),
    ("mask_mean", "mask mean"),
    ("mask_std", "mask std"),
    ("weight_std", "loss weight std"),
    ("noise_scale_std", "noise scale std"),
]


def _report(stats: dict) -> None:
    for segment_name, groups in stats.items():
        print(f"\n=== {segment_name} ===")
        header = "metric".ljust(20)
        for condition in CONDITIONS:
            header += condition.rjust(14)
        print(header)

        for key, label in _ROWS:
            line = label.ljust(20)
            for condition in CONDITIONS:
                line += f"{groups[condition][key]['mean']:>14.4f}"
            print(line)

        for key, value in groups["pair"].items():
            print(key.ljust(20) + f"{value['mean']:>14.4f}")


def main() -> None:
    args = _args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    models = load_model()
    # pyrefly: ignore [missing-attribute]
    unet = models["pipe"].unet.eval()
    for param in unet.parameters():
        param.requires_grad_(False)

    scheduler = SpatiallyVaryingDDPMScheduler.from_config(
        # pyrefly: ignore [missing-attribute]
        models["pipe"].scheduler.config
    )

    stats = measure(
        args,
        unet,
        models["vae"],
        models["text_encoders"],
        models["tokenizers"],
        scheduler,
    )

    _report(stats)

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "maskcov.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
