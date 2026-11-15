import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import CLIPModel, CLIPTokenizer

from .encoding import clip_normalization

CLIP_L = "openai/clip-vit-large-patch14"
CLIP_SIZE = 224

SCORERS = {
    "waifu": ("Eugeoter/waifu-scorer-v3", "model.safetensors"),
    "laion": (
        "camenduru/improved-aesthetic-predictor",
        "sac+logos+ava1-l14-linearMSE.pth",
    ),
}


class _WaifuHead(nn.Module):
    def __init__(self, width: int = 768):
        super().__init__()
        dims = [width, 2048, 512, 256, 128]
        dropout = [0.3, 0.3, 0.2, 0.1]
        layers: list[nn.Module] = []
        for index in range(4):
            layers += [
                nn.Linear(dims[index], dims[index + 1]),
                nn.ReLU(),
                nn.BatchNorm1d(dims[index + 1]),
                nn.Dropout(dropout[index]),
            ]
        layers += [nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _LaionHead(nn.Module):
    def __init__(self, width: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def load_head(kind: str) -> nn.Module:
    repo, filename = SCORERS[kind]
    path = hf_hub_download(repo, filename)
    if path.endswith(".safetensors"):
        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    head = _WaifuHead() if kind == "waifu" else _LaionHead()
    head.load_state_dict(state)
    return head.eval()


def square_views(pixels: torch.Tensor, size: int = CLIP_SIZE, views: int = 3):
    height, width = pixels.shape[-2:]
    scale = size / min(height, width)
    new_h = max(size, round(height * scale))
    new_w = max(size, round(width * scale))
    resized = F.interpolate(
        pixels, size=(new_h, new_w), mode="bicubic", align_corners=False, antialias=True
    )

    span = max(new_h, new_w) - size
    count = 1 if span == 0 else max(views, 1)
    offsets = [round(span * i / max(count - 1, 1)) for i in range(count)]

    crops = []
    for offset in offsets:
        if new_h >= new_w:
            crops.append(resized[..., offset : offset + size, :size])
        else:
            crops.append(resized[..., :size, offset : offset + size])
    return torch.stack(crops, dim=1)


class ImageScorer(nn.Module):
    def __init__(self, kind: str = "waifu", views: int = 3):
        super().__init__()
        clip = CLIPModel.from_pretrained(CLIP_L, torch_dtype=torch.float16)
        self.vision = clip.vision_model
        self.visual_projection = clip.visual_projection
        self.text = clip.text_model
        self.text_projection = clip.text_projection
        self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_L)
        self.head = load_head(kind).float()
        self.kind = kind
        self.views = views
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # noqa: ARG002
        return super().train(False)

    def image_embeds(self, pixels: torch.Tensor) -> torch.Tensor:
        batch = pixels.shape[0]
        views = square_views(((pixels.float() + 1.0) / 2.0), views=self.views)
        count = views.shape[1]
        flat = views.flatten(0, 1)
        mean, std = clip_normalization(flat.device, flat.dtype)
        flat = (flat - mean) / std

        with torch.autocast("cuda", dtype=torch.float16, enabled=flat.is_cuda):
            pooled = self.vision(pixel_values=flat.to(self.vision.dtype)).pooler_output
            embeds = self.visual_projection(pooled)

        return F.normalize(embeds.float(), dim=-1).view(batch, count, -1)

    @torch.no_grad()
    def text_embeds(self, prompts: list[str], device) -> torch.Tensor:
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(device)
        pooled = self.text(**tokens).pooler_output
        return F.normalize(self.text_projection(pooled).float(), dim=-1)

    def quality(self, embeds: torch.Tensor) -> torch.Tensor:
        batch, count, width = embeds.shape
        return (
            self.head(embeds.reshape(batch * count, width)).view(batch, count).mean(1)
        )

    def adherence(self, embeds: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return (embeds * text.unsqueeze(1)).sum(-1).mean(1)

    def identity(self, embeds: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return (embeds * reference).sum(-1).mean(1)
