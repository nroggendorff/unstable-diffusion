import torch
import torch.nn.functional as F


def _gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    ax = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def _gaussian_blur(img: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    kernel = _gaussian_kernel(kernel_size, sigma).to(img.device, img.dtype)
    padding = kernel_size // 2
    return F.conv2d(img, kernel, padding=padding, groups=img.shape[1])


class SubjectMaskBuilder:
    def __init__(
        self,
        blur_sigma_start=5.0,
        blur_sigma_end=1.0,
        min_mask_value=0.1,
    ):
        self.blur_sigma_start = blur_sigma_start
        self.blur_sigma_end = blur_sigma_end
        self.min_mask_value = min_mask_value

    def blur_sigma_for_step(self, step: int, total_steps: int) -> float:
        frac = step / max(total_steps - 1, 1)
        return self.blur_sigma_start + frac * (
            self.blur_sigma_end - self.blur_sigma_start
        )

    def build_mask(
        self, raw_discrepancy: torch.Tensor, blur_sigma: float
    ) -> torch.Tensor:
        kernel_size = int(4 * blur_sigma + 1)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        blurred = _gaussian_blur(raw_discrepancy, kernel_size, blur_sigma)
        bmin = blurred.amin(dim=(2, 3), keepdim=True)
        bmax = blurred.amax(dim=(2, 3), keepdim=True)
        normalized = (blurred - bmin) / (bmax - bmin + 1e-8)
        return torch.clamp(normalized, min=self.min_mask_value)
