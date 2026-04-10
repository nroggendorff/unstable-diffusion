import torch

from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig

MODEL_ID = "glides/counterfeit"
DEVICE = "cuda"
NUM_INFERENCE_STEPS = 30

EARLY_SEG = 10
MID_SEG = 10
LATE_SEG = 10

LR = 1e-4
LORA_RANK = 32
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]


def get_lora_config():
    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        inference_mode=False,
    )


def load_model():
    # pyrefly: ignore [missing-attribute]
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(DEVICE)

    vae = pipe.vae.to(dtype=torch.float16)
    text_encoder = pipe.text_encoder.to(dtype=torch.float16)
    tokenizer = pipe.tokenizer

    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    # pyrefly: ignore [missing-attribute]
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)

    return {
        "pipe": pipe,
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
    }
