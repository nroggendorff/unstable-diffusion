import torch
import torch.nn.functional as F


def apply_prompt_seed_offset(
    encoder_hidden_states: torch.Tensor,
    pooled: torch.Tensor,
    prompt_seed: int,
    strength: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen_seq = torch.Generator(device=encoder_hidden_states.device).manual_seed(
        prompt_seed
    )
    gen_pool = torch.Generator(device=pooled.device).manual_seed(
        prompt_seed ^ 0xABCD1234
    )

    seq_noise = torch.randn(
        encoder_hidden_states.shape,
        generator=gen_seq,
        device=encoder_hidden_states.device,
    ).to(encoder_hidden_states.dtype)
    pool_noise = torch.randn(
        pooled.shape,
        generator=gen_pool,
        device=pooled.device,
    ).to(pooled.dtype)

    seq_scale = (
        encoder_hidden_states.float().norm(dim=-1, keepdim=True).mean() * strength
    )
    pool_scale = pooled.float().norm(dim=-1, keepdim=True).mean() * strength

    return (
        encoder_hidden_states + seq_noise * seq_scale.to(encoder_hidden_states.dtype),
        pooled + pool_noise * pool_scale.to(pooled.dtype),
    )


def encode_prompt(
    prompts, text_encoder, text_encoder_2, tokenizer, tokenizer_2, device
):
    def tokenize(tok, text):
        return tok(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=tok.model_max_length,
            truncation=True,
        ).to(device)

    inputs1 = tokenize(tokenizer, prompts)
    inputs2 = tokenize(tokenizer_2, prompts)

    out1 = text_encoder(input_ids=inputs1.input_ids, output_hidden_states=True)
    hidden1 = out1.hidden_states[-2]

    out2 = text_encoder_2(input_ids=inputs2.input_ids, output_hidden_states=True)
    hidden2 = out2.hidden_states[-2]
    pooled = out2[0]

    encoder_hidden_states = torch.cat([hidden1, hidden2], dim=-1)
    return encoder_hidden_states, pooled, inputs1.attention_mask


def decode_for_clip(
    vae, latents: torch.Tensor, clip_mean: torch.Tensor, clip_std: torch.Tensor
) -> torch.Tensor:
    latents = latents.to(dtype=vae.dtype)

    decoded = vae.decode(latents / vae.config.scaling_factor).sample

    decoded = (decoded.float().clamp(-1, 1) + 1) / 2
    decoded = F.interpolate(
        decoded, size=(224, 224), mode="bilinear", align_corners=False
    )
    return (decoded - clip_mean) / clip_std
