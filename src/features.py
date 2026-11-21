import torch
import torch.nn.functional as F
import torchvision
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoModel

DINO_ID = "facebook/dinov2-base"
VGG_ID = "timm/vgg19.tv_in1k"
SPACES = ("content", "style")
DINO_SIZE = (322, 224)
VGG_SIZE = (608, 416)
VGG_LAYERS = (1, 6, 11, 20, 29)
VGG_BLOCKS = (64, 64, 128, 128, 256, 256, 512, 512, 512, 512)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_vgg_features() -> torch.nn.Sequential:
    layers = list(torchvision.models.vgg19().features.children())
    features = torch.nn.Sequential(*layers[: VGG_LAYERS[-1] + 1])
    path = hf_hub_download(VGG_ID, "model.safetensors")
    state = {}
    with safe_open(path, framework="pt") as weights:
        for key in weights.keys():
            if key.startswith("features."):
                index = int(key.split(".")[1])
                if index <= VGG_LAYERS[-1]:
                    state[key[len("features.") :]] = weights.get_tensor(key)
    features.load_state_dict(state)
    return features


def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        x, size=size, mode="bilinear", antialias=True, align_corners=False
    )


class FeatureSpaces(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dino = AutoModel.from_pretrained(DINO_ID)
        self.vgg = load_vgg_features()
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # noqa: ARG002
        return super().train(False)

    def forward(self, pixels: torch.Tensor) -> dict[str, torch.Tensor]:
        mean = torch.tensor(IMAGENET_MEAN, device=pixels.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=pixels.device).view(1, 3, 1, 1)
        x = (pixels.float() - mean) / std
        content = self.dino(pixel_values=_resize(x, DINO_SIZE)).pooler_output
        h = _resize(x, VGG_SIZE)
        stats = []
        with torch.autocast("cuda", dtype=torch.float16, enabled=h.is_cuda):
            for index, layer in enumerate(self.vgg):
                h = layer(h)
                if index in VGG_LAYERS:
                    flat = h.float().flatten(2)
                    stats += [flat.mean(-1), flat.std(-1)]
        return {
            "content": F.normalize(content.float(), dim=-1),
            "style": torch.cat(stats, -1),
        }


def project(space: str, raw: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if space == "content":
        return F.normalize(raw.float(), dim=-1)
    reference = reference.float()
    z = (raw.float() - reference.mean(0)) / (reference.std(0) + 1e-6)
    blocks = torch.split(z, list(VGG_BLOCKS), dim=-1)
    return F.normalize(torch.cat([b / b.shape[-1] ** 0.5 for b in blocks], -1), dim=-1)
