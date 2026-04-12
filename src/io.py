import os

from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file


def save_lora(model, path):
    state_dict = get_peft_model_state_dict(model)

    converted = {}
    for k, v in state_dict.items():
        k = k.replace("base_model.model.", "unet.")
        k = k.replace(".early", str())
        k = k.replace(".mid", str())
        k = k.replace(".late", str())
        k = k.replace(".final", str())
        converted[k] = v

    os.makedirs(path, exist_ok=True)
    save_file(converted, os.path.join(path, "pytorch_lora_weights.safetensors"))
