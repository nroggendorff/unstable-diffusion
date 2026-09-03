import random
import re
import string

import torch
import torch.nn.functional as F

CHUNK_TOKENS = 75
MAX_CHUNKS = 2
SEQUENCE_LENGTH = MAX_CHUNKS * (CHUNK_TOKENS + 2)

HIDDEN_SPLIT = 768
POOLED_DIM = 1280

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


_CLAUSE_BOUNDARY = re.compile(r"(?<=[.,;:!?])\s+")


def split_clauses(caption: str) -> list[str]:
    parts = [part.strip() for part in _CLAUSE_BOUNDARY.split(caption.strip())]
    return [part for part in parts if part]


def _trim_dangling(text: str) -> str:
    words = text.split()
    while words:
        bare = words[-1].strip(string.punctuation).lower()
        if bare and bare not in _STOPWORDS:
            break
        words.pop()
    return " ".join(words).strip(string.punctuation + " ")


def subset_caption(caption: str, rng: random.Random, min_keep: float = 0.15) -> str:
    clauses = split_clauses(caption)
    if len(clauses) < 2:
        return caption

    keep = max(1, round(rng.uniform(min_keep, 1.0) * len(clauses)))
    if keep >= len(clauses):
        return caption

    trimmed = _trim_dangling(" ".join(clauses[:keep]))
    return trimmed if trimmed else caption


def clip_normalization(device, dtype=torch.float32):
    mean = torch.tensor(_CLIP_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def rms_scaled_noise(
    reference: torch.Tensor,
    sigma: float,
    generator=None,
    split: int | None = None,
):
    noise = torch.randn(
        reference.shape,
        generator=generator,
        device=reference.device,
        dtype=torch.float32,
    )
    ref = reference[0].float()

    def block_rms(block: torch.Tensor) -> float:
        return block.norm().item() / (block.numel() ** 0.5)

    if split is None or not 0 < split < reference.shape[-1]:
        return (noise * sigma * block_rms(ref)).to(reference.dtype)

    scale = torch.empty(
        reference.shape[-1], device=reference.device, dtype=torch.float32
    )
    scale[:split] = block_rms(ref[..., :split])
    scale[split:] = block_rms(ref[..., split:])
    return (noise * sigma * scale).to(reference.dtype)


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


def _as_list(value) -> list:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def encode_prompt(prompts, text_encoders, tokenizers, device, content_cache=None):
    if isinstance(prompts, str):
        prompts = [prompts]
    if content_cache is None:
        content_cache = {}

    text_encoders = _as_list(text_encoders)
    tokenizers = _as_list(tokenizers)

    primary = tokenizers[0]
    bos = primary.bos_token_id
    eos = primary.eos_token_id

    windows = []
    batch_content = []

    for prompt in prompts:
        ids = primary(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=CHUNK_TOKENS * MAX_CHUNKS,
        )["input_ids"]

        for chunk in range(MAX_CHUNKS):
            window = ids[chunk * CHUNK_TOKENS : (chunk + 1) * CHUNK_TOKENS]
            windows.append(window)
            batch_content.append(
                [0]
                + [int(_is_content(primary, t, content_cache)) for t in window]
                + [0] * (CHUNK_TOKENS - len(window))
                + [0]
            )

    hidden = []
    pooled = None

    for encoder, tokenizer in zip(text_encoders, tokenizers):
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
        input_ids = torch.tensor(
            [[bos] + w + [pad] * (CHUNK_TOKENS - len(w)) + [eos] for w in windows],
            dtype=torch.long,
            device=device,
        )
        output = encoder(input_ids=input_ids, output_hidden_states=True)
        hidden.append(output.hidden_states[-2])
        if getattr(output, "text_embeds", None) is not None:
            pooled = output.text_embeds

    batch = len(prompts)
    embeddings = torch.cat(hidden, dim=-1).reshape(batch, SEQUENCE_LENGTH, -1)
    content = torch.tensor(batch_content, dtype=torch.float32, device=device).reshape(
        batch, SEQUENCE_LENGTH
    )

    if pooled is None:
        pooled = embeddings.new_zeros((batch, POOLED_DIM))
    else:
        pooled = pooled.reshape(batch, MAX_CHUNKS, -1)[:, 0]

    return embeddings, pooled, content


def decode_for_clip(
    vae, latents: torch.Tensor, straight_through: bool = False
) -> torch.Tensor:
    decoded = vae.decode(latents.to(dtype=vae.dtype) / vae.config.scaling_factor).sample

    decoded = decoded.float()
    clamped = decoded.clamp(-1, 1)
    if straight_through:
        clamped = decoded + (clamped - decoded).detach()
    decoded = (clamped + 1) / 2
    decoded = F.interpolate(
        decoded, size=(224, 224), mode="bilinear", align_corners=False
    )

    mean, std = clip_normalization(decoded.device, decoded.dtype)
    return (decoded - mean) / std
