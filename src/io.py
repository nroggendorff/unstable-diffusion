import os

from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file


def save_lora(model, path, segment):
    state_dict = get_peft_model_state_dict(model)
    converted = {
        k.replace("base_model.model.", "unet."): v for k, v in state_dict.items()
    }

    os.makedirs(path, exist_ok=True)
    save_file(converted, os.path.join(path, f"{segment}.safetensors"))
