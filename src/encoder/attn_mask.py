import torch
import torch.nn.functional as F


class _CapturingProcessor:
    def __init__(self, store):
        self.store = store

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None
    ):
        kv = encoder_hidden_states

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
        weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)

        self.store.append(weights.detach().cpu())

        out = torch.bmm(weights, v)
        out = attn.batch_to_head_dim(out)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


class CrossAttentionCapture:
    def __init__(self, unet):
        self.unet = unet
        self._original = {}
        self.store = []

    def __enter__(self):
        self.store.clear()
        self._original = dict(self.unet.attn_processors)
        merged = {
            name: _CapturingProcessor(self.store) if "attn2" in name else proc
            for name, proc in self._original.items()
        }
        self.unet.set_attn_processor(merged)
        return self

    def __exit__(self, *args):
        self.unet.set_attn_processor(self._original)

    def build_mask(
        self, token_attention_mask: torch.Tensor, spatial_size: tuple
    ) -> torch.Tensor:
        if not self.store:
            raise RuntimeError("No cross-attention weights were captured.")

        B = token_attention_mask.shape[0]
        accumulated = []

        for weights in self.store:
            weights = weights.to(token_attention_mask.device)
            BH, S, T = weights.shape
            if BH % B != 0:
                continue
            H = BH // B
            w = weights.reshape(B, H, S, T).mean(1)
            content = token_attention_mask.float()
            w = (w * content.unsqueeze(1)).sum(-1)
            side = int(w.shape[1] ** 0.5)

            w_spatial = w.reshape(B, 1, side, side)

            w_resized = F.interpolate(
                w_spatial,
                size=spatial_size,
                mode="bilinear",
                align_corners=False,
            )

            accumulated.append(w_resized.squeeze(1))

        if not accumulated:
            raise RuntimeError("No valid cross-attention maps found.")

        avg = torch.stack(accumulated).mean(0)
        spatial = avg.unsqueeze(1)

        mn = spatial.flatten(1).min(1).values.view(B, 1, 1, 1)
        mx = spatial.flatten(1).max(1).values.view(B, 1, 1, 1)
        spatial = (spatial - mn) / (mx - mn + 1e-8)

        return F.interpolate(
            spatial.to(token_attention_mask.device),
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )
