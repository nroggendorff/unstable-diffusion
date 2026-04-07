import torch
import torch.nn.functional as F


LAYER_PATTERNS = [
    ".mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0",
    ".up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0",
    ".up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0",
]


def _matches_any(name, patterns):
    for p in patterns:
        if name.endswith(p):
            return True
    return False


def collect_activations(unet, batch_inputs, layer_patterns):
    activations = {p: [] for p in layer_patterns}
    hooks = []

    def make_hook(pattern):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            flat = (
                out.detach().float().mean(dim=(0, 2, 3))
                if out.ndim == 4
                else out.detach().float().mean(dim=(0, 1))
            )
            activations[pattern].append(flat)

        return hook

    for name, module in unet.named_modules():
        for p in layer_patterns:
            if name.endswith(p):
                hooks.append(module.register_forward_hook(make_hook(p)))

    with torch.no_grad():
        for inp in batch_inputs:
            unet(**inp)

    for hook in hooks:
        hook.remove()

    return activations


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

    for name, module in unet.named_modules():
        for p in layer_patterns:
            if name.endswith(p):
                hooks.append(module.register_forward_hook(make_hook(store_a, p)))

    unet(
        noisy_latents_a,
        t,
        encoder_hidden_states=text_emb,
        added_cond_kwargs=added_cond_kwargs,
    )

    for hook in hooks:
        hook.remove()
    hooks = []

    for name, module in unet.named_modules():
        for p in layer_patterns:
            if name.endswith(p):
                hooks.append(module.register_forward_hook(make_hook(store_b, p)))

    unet(
        noisy_latents_b,
        t,
        encoder_hidden_states=text_emb,
        added_cond_kwargs=added_cond_kwargs,
    )

    for hook in hooks:
        hook.remove()

    shared = [p for p in layer_patterns if p in store_a and p in store_b]
    if not shared:
        return torch.tensor(0.0)
    similarity = torch.stack(
        [
            F.cosine_similarity(store_a[p].unsqueeze(0), store_b[p].unsqueeze(0))
            for p in shared
        ]
    ).mean()

    return similarity
