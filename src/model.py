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

SEGMENT_TIMESTEP_RANGES = {
    "early": (595, 999),
    "mid": (265, 694),
    "late": (1, 364),
}


def get_lora_config(rank: int = LORA_RANK, alpha: int = LORA_ALPHA):
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=LORA_TARGET_MODULES,
        inference_mode=False,
    )


def load_model():
    # pyrefly: ignore [missing-attribute]
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
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
