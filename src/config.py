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


def get_config() -> argparse.Namespace:
    sm = _sm_defaults()

    parser = argparse.ArgumentParser()

    def add(name: str, type_: type, default):
        val = sm.get(name)
        parser.add_argument(
            f"--{name}", type=type_, default=type_(val) if val is not None else default
        )

    add("steps", int, 15000)
    add("mini_batch_size", int, 2)
    add("grad_accum_steps", int, 4)
    add("lr", float, 1e-4)
    add("lora_rank", int, 32)
    add("lora_alpha", int, 32)
    add("mask_blur_sigma_start", float, 7.0)
    add("mask_blur_sigma_end", float, 1.0)
    add("mask_min_value", float, 0.0)
    add("scheduler_subject_power", float, 0.6)
    add("scheduler_bg_scale", float, 1.0)
    add("scheduler_min_scale", float, 0.0)

    add("rl_steps", int, 300)
    add("rl_lr", float, 5e-6)
    add("rl_grounding_weight", float, 0.4)
    add("rl_diversity_weight", float, 0.1)
    add("rl_baseline_momentum", float, 0.95)

    sm_default_output = (
        _SM_MODEL_DIR if os.path.isdir(_SM_MODEL_DIR) else _LOCAL_MODEL_DIR
    )
    sm_output_val = sm.get("output_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=sm_output_val if sm_output_val is not None else sm_default_output,
    )

    cfg = parser.parse_args()
    cfg.effective_batch_size = cfg.mini_batch_size * cfg.grad_accum_steps
    cfg.train_steps = cfg.steps // cfg.effective_batch_size
    return cfg
