import torch
import torch.nn.functional as F


def compute_perceptual_discrepancy(
    pred_features: list[torch.Tensor],
    target_features: list[torch.Tensor],
) -> torch.Tensor:
    if len(pred_features) == 0:
        raise ValueError("pred_features is empty")

    diffs = []

    for p_feat, t_feat in zip(pred_features, target_features):
        if t_feat.shape[2:] != p_feat.shape[2:]:
            t_feat = F.interpolate(
                t_feat, size=p_feat.shape[2:], mode="bilinear", align_corners=False
            )

        diff = (p_feat - t_feat).pow(2)
        diff_per_loc = diff.sum(dim=1, keepdim=True)
        diffs.append(diff_per_loc)

    total_diff = torch.stack(diffs, dim=0).mean(dim=0)
    return torch.sqrt(total_diff)
