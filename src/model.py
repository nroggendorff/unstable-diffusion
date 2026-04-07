import copy
import torch

from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from peft import get_peft_model, LoraConfig


MODEL_ID = "glides/illustriousxl"
DEVICE = "cuda"
NUM_INFERENCE_STEPS = 30
EARLY_STEPS = 4
LR = 1e-5

LORA_RANK = 4
LORA_ALPHA = 4
LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]


def load_model():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(DEVICE)

    vae = pipe.vae.to(dtype=torch.float16)
    text_encoder = pipe.text_encoder.to(dtype=torch.float16)
    text_encoder_2 = pipe.text_encoder_2.to(dtype=torch.float16)
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2

    unet_base = pipe.unet.eval().to(dtype=torch.float16)
    for param in unet_base.parameters():
        param.requires_grad = False

    unet_for_creative = copy.deepcopy(unet_base)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        inference_mode=False,
    )
    unet_creative = get_peft_model(unet_for_creative, lora_config).train()

    optimizer = torch.optim.AdamW(unet_creative.parameters(), lr=LR)

    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    early_timesteps = scheduler.timesteps[:EARLY_STEPS].to(DEVICE)

    return {
        "pipe": pipe,
        "vae": vae,
        "unet_base": unet_base,
        "unet_creative": unet_creative,
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "early_timesteps": early_timesteps,
    }
