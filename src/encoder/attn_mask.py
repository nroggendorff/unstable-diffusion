import torch
import torch.nn.functional as F


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


class _CapturingProcessor:
    def __init__(self, store):
        self.store = store

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
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
        self.store.append(weights.detach().half().cpu())

        weights_typed = weights.to(q.dtype)
        out = torch.bmm(weights_typed, v)
        out = attn.batch_to_head_dim(out)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


class CrossAttentionCapture:
    def __init__(self, unet):
        self.unet = unet
        self._target = _unwrap_unet(unet)
        self._original = {}
        self.store = []

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
            name: _CapturingProcessor(self.store) if "attn2" in name else proc
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

    def build_mask(
        self,
        token_attention_mask: torch.Tensor,
        spatial_size: tuple[int, int],
    ) -> torch.Tensor:
        if not self.store:
            raise RuntimeError(
                "No cross-attention weights were captured. "
                "The attention processor hook did not run."
            )

        B = token_attention_mask.shape[0]
        accumulated = []

        for weights in self.store:
            weights = weights.float().to(token_attention_mask.device)
            BH, S, T = weights.shape
            if BH % B != 0:
                continue
            H = BH // B
            w = weights.reshape(B, H, S, T).mean(1)

            content = token_attention_mask.float().to(w.device)
            w = (w * content.unsqueeze(1)).sum(-1)

            side = int(S**0.5)
            if side * side != S:
                continue

            w_spatial = w.reshape(B, 1, side, side)
            w_resized = F.interpolate(
                w_spatial,
                size=spatial_size,
                mode="bilinear",
                align_corners=False,
            )

            accumulated.append(w_resized.squeeze(1))

        self.store.clear()

        if not accumulated:
            raise RuntimeError(
                "Cross-attention was captured, but no usable maps were produced."
            )

        stack = torch.stack(accumulated, dim=1)

        n = stack.shape[1]
        layer_weights = torch.linspace(0.5, 1.0, n, device=stack.device)
        layer_weights = layer_weights / layer_weights.sum()

        layer_weights = layer_weights.view(1, n, 1, 1)

        avg = (stack * layer_weights).sum(dim=1)

        spatial = avg.unsqueeze(1)

        mn = spatial.flatten(1).min(1).values.view(B, 1, 1, 1)
        mx = spatial.flatten(1).max(1).values.view(B, 1, 1, 1)
        spatial = (spatial - mn) / (mx - mn + 1e-8)

        return spatial
