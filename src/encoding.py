import string

import torch
import torch.nn.functional as F

CHUNK_TOKENS = 75
MAX_CHUNKS = 2
SEQUENCE_LENGTH = MAX_CHUNKS * (CHUNK_TOKENS + 2)

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "into",
    "through",
    "during",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "then",
    "once",
    "here",
    "there",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "can",
    "will",
    "just",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "as",
    "while",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "he",
    "she",
    "they",
    "them",
    "his",
    "her",
    "their",
    "who",
    "which",
    "what",
    "where",
    "when",
    "how",
    "why",
    "also",
    "against",
    "between",
}


def clip_normalization(device, dtype=torch.float32):
    mean = torch.tensor(_CLIP_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def rms_scaled_noise(reference: torch.Tensor, sigma: float, generator=None):
    noise = torch.randn(
        reference.shape,
        generator=generator,
        device=reference.device,
        dtype=torch.float32,
    )
    rms = reference[0].float().norm() / (reference[0].numel() ** 0.5)
    return (noise * sigma * rms).to(reference.dtype)


def _token_text(tokenizer, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    if token is None:
        return ""
    return token.replace("</w>", "").strip().lower()


def _is_content(tokenizer, token_id: int, cache: dict) -> bool:
    if token_id not in cache:
        text = _token_text(tokenizer, token_id)
        cache[token_id] = bool(
            text
            and text not in _STOPWORDS
            and not all(ch in string.punctuation for ch in text)
        )
    return cache[token_id]


def encode_prompt(prompts, text_encoder, tokenizer, device, content_cache=None):
    if isinstance(prompts, str):
        prompts = [prompts]
    if content_cache is None:
        content_cache = {}

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

    batch_ids = []
    batch_content = []

    for prompt in prompts:
        ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=CHUNK_TOKENS * MAX_CHUNKS,
        )["input_ids"]

        for chunk in range(MAX_CHUNKS):
            window = ids[chunk * CHUNK_TOKENS : (chunk + 1) * CHUNK_TOKENS]
            padding = [pad] * (CHUNK_TOKENS - len(window))

            batch_ids.append([bos] + window + padding + [eos])
            batch_content.append(
                [0]
                + [int(_is_content(tokenizer, t, content_cache)) for t in window]
                + [0] * len(padding)
                + [0]
            )

    input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
    hidden = text_encoder(input_ids=input_ids).last_hidden_state

    batch = len(prompts)
    embeddings = hidden.reshape(batch, SEQUENCE_LENGTH, hidden.shape[-1])
    content = torch.tensor(batch_content, dtype=torch.float32, device=device).reshape(
        batch, SEQUENCE_LENGTH
    )

    return embeddings, content


def decode_for_clip(vae, latents: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode(latents.to(dtype=vae.dtype) / vae.config.scaling_factor).sample

    decoded = (decoded.float().clamp(-1, 1) + 1) / 2
    decoded = F.interpolate(
        decoded, size=(224, 224), mode="bilinear", align_corners=False
    )

    mean, std = clip_normalization(decoded.device, decoded.dtype)
    return (decoded - mean) / std
