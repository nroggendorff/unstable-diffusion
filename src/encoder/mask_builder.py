import torch
import torch.nn.functional as F


def percentile_normalize(
    x: torch.Tensor, low: float = 0.02, high: float = 0.98, eps: float = 1e-8
) -> torch.Tensor:
    flat = x.flatten(1).float()
    lo = torch.quantile(flat, low, dim=1).view(-1, 1, 1, 1)
    hi = torch.quantile(flat, high, dim=1).view(-1, 1, 1, 1)
    return ((x.float() - lo) / (hi - lo + eps)).clamp(0.0, 1.0)


def relative_normalize(x: torch.Tensor, gain: float, eps: float = 1e-8) -> torch.Tensor:
    flat = x.flatten(1).float()
    center = flat.median(dim=1).values.view(-1, 1, 1, 1)
    relative = (x.float() - center) / (center.abs() + eps)
    return (0.5 + gain * relative).clamp(0.0, 1.0)


def normalize_mask(x: torch.Tensor, gain: float) -> torch.Tensor:
    if gain <= 0.0:
        return percentile_normalize(x)
    return relative_normalize(x, gain)


def _gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    ax = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def _gaussian_blur(img: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    kernel = _gaussian_kernel(kernel_size, sigma).to(img.device, img.dtype)
    half = kernel_size // 2
    padding = min(half, min(img.shape[-2], img.shape[-1]) - 1)
    padded = F.pad(img, (padding, padding, padding, padding), mode="replicate")
    return F.conv2d(padded, kernel, padding=half - padding, groups=img.shape[1])


class SubjectMaskBuilder:
    def __init__(
        self,
        blur_sigma_start=5.0,
        blur_sigma_end=1.0,
        min_mask_value=0.1,
        gain=0.0,
    ):
        self.blur_sigma_start = blur_sigma_start
        self.blur_sigma_end = blur_sigma_end
        self.min_mask_value = min_mask_value
        self.gain = gain

    def blur_sigma_for_step(self, step: int, total_steps: int) -> float:
        frac = step / max(total_steps - 1, 1)
        return self.blur_sigma_start + frac * (
            self.blur_sigma_end - self.blur_sigma_start
        )

    def build_mask(
        self, raw_discrepancy: torch.Tensor, blur_sigma: float
    ) -> torch.Tensor:
        kernel_size = int(6 * blur_sigma + 1)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        blurred = _gaussian_blur(raw_discrepancy, kernel_size, blur_sigma)
        if self.gain > 0.0:
            return blurred.clamp(self.min_mask_value, 1.0)
        return torch.clamp(percentile_normalize(blurred), min=self.min_mask_value)
