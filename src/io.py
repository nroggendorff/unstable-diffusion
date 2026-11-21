import os
from typing import Any

from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file


def save_lora(model, path, segment):
    state_dict = get_peft_model_state_dict(model)
    converted = {
        k.replace("base_model.model.", "unet."): v for k, v in state_dict.items()
    }

    os.makedirs(path, exist_ok=True)
    save_file(converted, os.path.join(path, f"{segment}.safetensors"))


def load_lora(model, path, segment) -> int:
    state = load_file(os.path.join(path, f"{segment}.safetensors"))
    converted = {
        "base_model.model." + k.removeprefix("unet."): v for k, v in state.items()
    }
    result: Any = set_peft_model_state_dict(model, converted)
    missing = [k for k in result.missing_keys if "lora_" in k]
    if result.unexpected_keys or missing:
        raise RuntimeError(
            f"{segment}.safetensors in {path} does not match the model: "
            f"{len(result.unexpected_keys)} unexpected keys, {len(missing)} LoRA keys "
            f"missing (first: {(result.unexpected_keys + missing)[0]})"
        )
    return len(converted)
