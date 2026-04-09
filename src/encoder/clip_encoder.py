import torch
import torch.nn as nn
from transformers import CLIPModel


class CLIPVisionEncoder(nn.Module):
    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        feature_layers=None,
        freeze=True,
    ):
        super().__init__()
        if feature_layers is None:
            feature_layers = [2, 4, 6, 8]
        self.feature_layers = feature_layers

        full_clip = CLIPModel.from_pretrained(model_name)
        self.model = full_clip.vision_model
        self.hidden_dim = self.model.config.hidden_size

        self._activations = {}
        for i, block in enumerate(self.model.encoder.layers):
            if i in feature_layers:
                block.register_forward_hook(self._make_hook(i))

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        self.eval()

    def _make_hook(self, layer_idx):
        def hook(module, input, output):  # noqa: ARG001
            self._activations[layer_idx] = output[0].detach()

        return hook

    def extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        self._activations.clear()
        self.model(x)

        features = []
        for idx in self.feature_layers:
            acts = self._activations[idx]
            B, N, C = acts.shape
            acts = acts[:, 1:, :]
            side = int((N - 1) ** 0.5)
            spatial = acts.transpose(-1, -2).reshape(B, C, side, side)
            features.append(spatial)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)[-1]
