import argparse

_DEFAULT_OUTPUT_DIR = "./creative-lora"


def default_output_dir() -> str:
    return _DEFAULT_OUTPUT_DIR


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    def add(name: str, type_, default):
        parser.add_argument(f"--{name}", type=type_, default=default)

    add("steps", int, 15000)
    add("mini_batch_size", int, 2)
    add("grad_accum_steps", int, 4)
    add("cache_size", int, 4000)
    add("shuffle_buffer", int, 768)
    add("lr", float, 1e-4)
    add("lora_rank", int, 32)
    add("lora_alpha", int, 32)
    add("grad_clip_norm", float, 1.0)

    add("mask_blur_sigma_start", float, 7.0)
    add("mask_blur_sigma_end", float, 1.0)
    add("mask_min_value", float, 0.0)
    add("mask_gain", float, 1.5)

    add("noise_bg_boost", float, 1.5)
    add("noise_t_ramp", float, 0.3)
    add("noise_lf_levels", int, 8)
    add("noise_lf_decay", float, 0.66)

    add("loss_bg_weight", float, 0.25)
    add("snr_gamma", float, 5.0)

    add("cond_dropout_prob", float, 0.1)
    add("cond_partial_prob", float, 0.1)
    add("cond_partial_max", float, 0.6)
    add("embed_jitter_max", float, 0.3)
    add("caption_subset_prob", float, 0.5)
    add("caption_subset_min", float, 0.15)

    add("refresh_cache_per_segment", _bool, True)

    add("rl_steps", int, 0)
    add("rl_lr", float, 3e-6)
    add("rl_grounding_weight", float, 0.0)
    add("rl_refs", int, 1)
    add("rl_group", int, 2)
    add("rl_logprob_subsample", int, 1)

    parser.add_argument("--output_dir", type=str, default=default_output_dir())

    cfg = parser.parse_args()

    if cfg.rl_steps > 0 and cfg.rl_group < 2:
        parser.error(
            "--rl_group must be at least 2; group-relative advantages need a spread."
        )

    cfg.effective_batch_size = cfg.mini_batch_size * cfg.grad_accum_steps
    cfg.train_steps = cfg.steps // cfg.effective_batch_size
    return cfg
