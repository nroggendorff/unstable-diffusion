import torch
import torch.nn.functional as F


LAYER_PATTERNS = [
    ".mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0",
    ".up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0",
    ".up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0",
]


def compute_diversity_loss(
    unet,
    noisy_latents_a,
    noisy_latents_b,
    t,
    text_emb,
    pooled_emb,
    time_ids,
    layer_patterns,
):
    noisy_latents_a = noisy_latents_a.detach().contiguous()
    noisy_latents_b = noisy_latents_b.detach().contiguous()
    t = t.detach()
    text_emb = text_emb.detach().contiguous()
    pooled_emb = pooled_emb.detach()
    time_ids = time_ids.detach()

    added_cond_kwargs = {"text_embeds": pooled_emb, "time_ids": time_ids}
    store_a = {}
    store_b = {}
    hooks = []

    def make_hook(store, pattern):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            flat = (
                out.float().mean(dim=(0, 2, 3))
                if out.ndim == 4
                else out.float().mean(dim=(0, 1))
            )
            store[pattern] = flat

        return hook

    try:
        for name, module in unet.named_modules():
            for p in layer_patterns:
                if name.endswith(p):
                    hooks.append(module.register_forward_hook(make_hook(store_a, p)))

        with torch.no_grad():
            unet(
                noisy_latents_a,
                t,
                encoder_hidden_states=text_emb,
                added_cond_kwargs=added_cond_kwargs,
            )
    finally:
        for hook in hooks:
            hook.remove()
    hooks = []

    try:
        for name, module in unet.named_modules():
            for p in layer_patterns:
                if name.endswith(p):
                    hooks.append(module.register_forward_hook(make_hook(store_b, p)))

        with torch.no_grad():
            unet(
                noisy_latents_b,
                t,
                encoder_hidden_states=text_emb,
                added_cond_kwargs=added_cond_kwargs,
            )
    finally:
        for hook in hooks:
            hook.remove()

    shared = [p for p in layer_patterns if p in store_a and p in store_b]
    if not shared:
        return torch.tensor(0.0, device=noisy_latents_a.device)
    similarity = torch.stack(
        [
            F.cosine_similarity(store_a[p].unsqueeze(0), store_b[p].unsqueeze(0))
            for p in shared
        ]
    ).mean()

    return similarity
