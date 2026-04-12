import torch
import torch.nn.functional as F


GROUNDING_WEIGHT = 0.4
SUBJECT_DRIFT_WEIGHT = 0.25
DIVERSITY_WEIGHT = 0.1

SUBJECT_DRIFT_TARGET_SIM = 0.1


def compute_loss(
    unet,
    noisy_latents,
    t,
    text_emb,
    added_cond_kwargs,
    mask,
    noise_scale,
    alphas_cumprod,
    t_normalized,
    clean_latents,
    base_pred,
    uniform_noisy,
    grounding_weight=GROUNDING_WEIGHT,
    subject_drift_weight=SUBJECT_DRIFT_WEIGHT,
    diversity_weight=DIVERSITY_WEIGHT,
):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        pred = unet(
            noisy_latents,
            t,
            encoder_hidden_states=text_emb,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

    pred_f = pred.float()
    clean_f = clean_latents.float().to(pred_f.device)
    base_f = base_pred.float().to(pred_f.device)
    mask_f = mask.float().to(pred_f.device).clamp(0.0, 1.0)

    a = alphas_cumprod[t.long()].view(-1, 1, 1, 1).sqrt()
    b = (1 - alphas_cumprod[t.long()]).view(-1, 1, 1, 1).sqrt()
    ns = noise_scale.float().to(pred_f.device) if noise_scale is not None else 1.0

    pred_x0 = (noisy_latents.float() - b * ns * pred_f) / a
    base_x0 = (uniform_noisy.float().to(pred_f.device) - b * base_f) / a
    del base_f

    t_scalar = float(t_normalized.mean().item())
    gated = 0.15 <= t_scalar <= 0.85

    lora_delta = ((pred_x0 - base_x0) * mask_f).flatten(1)
    target_delta = ((clean_f - base_x0) * mask_f).flatten(1)
    grounding_loss = (
        1.0 - F.cosine_similarity(lora_delta, target_delta, dim=1, eps=1e-6)
    ).mean()
    del lora_delta, target_delta, clean_f

    pred_subj = (pred_x0 * mask_f).flatten(1)
    if gated and subject_drift_weight > 0:
        base_subj = (base_x0 * mask_f).flatten(1)
        subj_sim = F.cosine_similarity(pred_subj, base_subj, dim=1, eps=1e-6)
        subject_drift_loss = F.relu(subj_sim - SUBJECT_DRIFT_TARGET_SIM).mean()
        del base_subj
    else:
        subject_drift_loss = pred_f.new_tensor(0.0)

    del base_x0, mask_f

    if pred_f.shape[0] > 1 and diversity_weight > 0:
        pred_subj_norm = F.normalize(pred_subj, dim=1)
        sim_matrix = pred_subj_norm @ pred_subj_norm.T
        n = pred_subj_norm.shape[0]
        pair_mask = torch.triu(
            torch.ones(n, n, dtype=torch.bool, device=pred_subj_norm.device),
            diagonal=1,
        )
        batch_diversity_loss = sim_matrix[pair_mask].mean()
    else:
        batch_diversity_loss = pred_f.new_tensor(0.0)

    del pred_subj

    loss = (
        grounding_weight * grounding_loss
        + subject_drift_weight * subject_drift_loss
        + diversity_weight * batch_diversity_loss
    )
    return (
        loss,
        grounding_loss.detach(),
        subject_drift_loss.detach(),
        batch_diversity_loss.detach(),
    )
