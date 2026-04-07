import copy

import torch

from diffusers import StableDiffusionXLPipeline, DDPMScheduler


MODEL_ID = "glides/illustriousxl"
DEVICE = "cuda"
NUM_INFERENCE_STEPS = 30
EARLY_STEPS = 4
LR = 1e-5


def load_model():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(DEVICE)

    vae = pipe.vae
    unet_base = pipe.unet.eval()
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    unet_creative = copy.deepcopy(unet_base).train()

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
        "tokenizer": tokenizer,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "early_timesteps": early_timesteps,
    }
