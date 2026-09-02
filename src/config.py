import argparse
import json
import os

_SM_HP_PATH = "/opt/ml/input/config/hyperparameters.json"
_SM_MODEL_DIR = "/opt/ml/model"
_LOCAL_MODEL_DIR = "./creative-lora"


def _sm_defaults() -> dict:
    if not os.path.exists(_SM_HP_PATH):
        return {}
    with open(_SM_HP_PATH) as f:
        return json.load(f)


def default_output_dir() -> str:
    return _SM_MODEL_DIR if os.path.isdir(_SM_MODEL_DIR) else _LOCAL_MODEL_DIR


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def get_config() -> argparse.Namespace:
    sm = _sm_defaults()

    parser = argparse.ArgumentParser()

    def add(name: str, type_, default):
        val = sm.get(name)
        parser.add_argument(
            f"--{name}", type=type_, default=type_(val) if val is not None else default
        )

    add("steps", int, 15000)
    add("mini_batch_size", int, 2)
    add("grad_accum_steps", int, 4)
    add("cache_size", int, 4000)
    add("shuffle_buffer", int, 1000)
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
    add("noise_lf_levels", int, 6)
    add("noise_lf_decay", float, 0.6)

    add("loss_bg_weight", float, 0.25)
    add("snr_gamma", float, 5.0)

    add("cond_dropout_prob", float, 0.1)
    add("cond_partial_prob", float, 0.1)
    add("cond_partial_max", float, 0.6)
    add("embed_jitter_max", float, 0.3)
    add("caption_subset_prob", float, 0.5)
    add("caption_subset_min", float, 0.15)

    add("refresh_cache_per_segment", _bool, True)

    add("rl_steps", int, 300)
    add("rl_lr", float, 3e-6)
    add("rl_grounding_weight", float, 0.4)
    add("rl_refs", int, 2)
    add("rl_group", int, 4)
    add("rl_logprob_subsample", int, 2)

    sm_default_output = default_output_dir()
    sm_output_val = sm.get("output_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=sm_output_val if sm_output_val is not None else sm_default_output,
    )

    cfg = parser.parse_args()

    if cfg.rl_steps > 0 and cfg.rl_group < 2:
        parser.error(
            "--rl_group must be at least 2; group-relative advantages need a spread."
        )

    cfg.effective_batch_size = cfg.mini_batch_size * cfg.grad_accum_steps
    cfg.train_steps = cfg.steps // cfg.effective_batch_size
    return cfg
