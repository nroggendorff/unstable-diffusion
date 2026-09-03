import torch

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from peft import LoraConfig

MODEL_ID = "glides/illustriousxl"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
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


def load_vae(device=DEVICE):
    # pyrefly: ignore [missing-attribute]
    return AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16).to(device)


def load_unet():
    # pyrefly: ignore [missing-attribute]
    return UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=torch.float16
    )


def load_pipeline(device=DEVICE):
    # pyrefly: ignore [missing-attribute]
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        add_watermarker=False,
    ).to(device)

    pipe.vae = load_vae(device)
    return pipe


def load_model():
    pipe = load_pipeline()

    text_encoders = [
        pipe.text_encoder.to(dtype=torch.float16),
        pipe.text_encoder_2.to(dtype=torch.float16),
    ]
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]

    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    # pyrefly: ignore [missing-attribute]
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)

    return {
        "pipe": pipe,
        "vae": pipe.vae,
        "text_encoders": text_encoders,
        "tokenizers": tokenizers,
        "scheduler": scheduler,
        "zero_uncond": bool(getattr(pipe.config, "force_zeros_for_empty_prompt", True)),
    }
