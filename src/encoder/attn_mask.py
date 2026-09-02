import torch
import torch.nn.functional as F

from .mask_builder import normalize_mask


def _unwrap_unet(module):
    current = module
    seen = set()

    while True:
        if id(current) in seen:
            break
        seen.add(id(current))

        if hasattr(current, "set_attn_processor") and hasattr(
            current, "attn_processors"
        ):
            return current

        if hasattr(current, "base_model") and hasattr(current.base_model, "model"):
            current = current.base_model.model
            continue

        if hasattr(current, "model"):
            current = current.model
            continue

        break

    return module


def _grid_for(tokens: int, latent_h: int, latent_w: int):
    for factor in (1, 2, 4, 8, 16):
        grid_h, grid_w = latent_h // factor, latent_w // factor
        if grid_h > 0 and grid_w > 0 and grid_h * grid_w == tokens:
            return grid_h, grid_w
    return None


class _CapturingProcessor:
    def __init__(self, capture):
        self.capture = capture

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ):
        kv = encoder_hidden_states
        if kv is None:
            kv = hidden_states

        q = attn.to_q(hidden_states)
        k = attn.to_k(kv)
        v = attn.to_v(kv)

        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)

        scale = q.shape[-1] ** -0.5
        scores = torch.bmm(q, k.transpose(-1, -2)) * scale
        if attention_mask is not None:
            scores = scores + attention_mask

        weights = torch.softmax(scores.float(), dim=-1)

        self.capture.absorb(weights)

        out = torch.bmm(weights.to(q.dtype), v)
        out = attn.batch_to_head_dim(out)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


class CrossAttentionCapture:
    def __init__(self, unet, gain: float = 0.0):
        self.unet = unet
        self.gain = gain
        self._target = _unwrap_unet(unet)
        self._original = {}
        self.store = []
        self._content = None
        self._latent_hw = None

    def set_context(self, token_content_mask: torch.Tensor, latent_hw: tuple):
        self._content = token_content_mask.float()
        self._latent_hw = latent_hw

    def absorb(self, weights: torch.Tensor):
        if self._content is None or self._latent_hw is None:
            return

        content = self._content.to(weights.device)
        batch, tokens = content.shape
        heads_batch, positions, kv_tokens = weights.shape

        if heads_batch % batch != 0 or kv_tokens != tokens:
            return

        grid = _grid_for(positions, *self._latent_hw)
        if grid is None:
            return

        heads = heads_batch // batch
        averaged = weights.view(batch, heads, positions, tokens).mean(1)

        reduced = torch.bmm(averaged, content.unsqueeze(-1)).squeeze(-1)
        self.store.append(reduced.view(batch, 1, grid[0], grid[1]))

    def __enter__(self):
        self.store.clear()

        if not hasattr(self._target, "attn_processors") or not hasattr(
            self._target, "set_attn_processor"
        ):
            raise RuntimeError(
                "Could not find a diffusers UNet with attention processor support."
            )

        self._original = dict(self._target.attn_processors)
        merged = {
            name: _CapturingProcessor(self) if "attn2" in name else proc
            for name, proc in self._original.items()
        }

        replaced = sum(1 for name in self._original if "attn2" in name)
        if replaced == 0:
            raise RuntimeError(
                "No cross-attention processors were found to replace. "
                "Your diffusers UNet naming may differ from this version."
            )

        self._target.set_attn_processor(merged)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._original:
            self._target.set_attn_processor(self._original)

    def raw_mask(self, spatial_size: tuple[int, int]) -> torch.Tensor:
        if not self.store:
            raise RuntimeError(
                "No cross-attention weights were captured. "
                "The attention processor hook did not run."
            )

        resized = [
            F.interpolate(
                m, size=spatial_size, mode="bilinear", align_corners=False
            ).squeeze(1)
            for m in self.store
        ]
        self.store.clear()

        stack = torch.stack(resized, dim=1)

        n = stack.shape[1]
        layer_weights = torch.linspace(0.5, 1.0, n, device=stack.device)
        layer_weights = layer_weights / layer_weights.sum()

        averaged = (stack * layer_weights.view(1, n, 1, 1)).sum(dim=1)

        return averaged.unsqueeze(1)

    def build_mask(self, spatial_size: tuple[int, int]) -> torch.Tensor:
        return normalize_mask(self.raw_mask(spatial_size), self.gain)
