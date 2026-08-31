import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler


# pyrefly: ignore [invalid-inheritance]
class SpatiallyVaryingDDPMScheduler(DDPMScheduler):
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
        noise_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        alphas_cumprod = self.alphas_cumprod.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        if noise_scale is not None:
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod * noise_scale

        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    def predict_x0(self, x_t, noise_pred, t, noise_scale=None):
        alphas = self.alphas_cumprod.to(x_t.device)

        a = alphas[t.long()].view(-1, 1, 1, 1).sqrt()
        b = (1 - alphas[t.long()]).view(-1, 1, 1, 1).sqrt()

        if noise_scale is not None:
            noise_pred = noise_pred * noise_scale

        return (x_t - b * noise_pred) / a


def pyramid_noise(
    latents: torch.Tensor,
    levels: int = 4,
    decay: float = 0.35,
    eps: float = 1e-8,
) -> torch.Tensor:
    noise = torch.randn(latents.shape, device=latents.device, dtype=torch.float32)

    batch, channels, height, width = latents.shape
    for level in range(1, max(levels, 1)):
        stride = 2**level
        low_h, low_w = height // stride, width // stride
        if low_h < 1 or low_w < 1:
            break

        low = torch.randn(
            (batch, channels, low_h, low_w),
            device=latents.device,
            dtype=torch.float32,
        )
        noise = noise + F.interpolate(
            low, size=(height, width), mode="bilinear", align_corners=False
        ) * (decay**level)

    rms = noise.pow(2).flatten(1).mean(1).sqrt().view(-1, 1, 1, 1)
    return (noise / (rms + eps)).to(latents.dtype)


def compute_spatial_noise_scale(
    mask: torch.Tensor,
    t_normalized: torch.Tensor,
    bg_boost: float = 1.5,
    t_ramp: float = 0.3,
    eps: float = 1e-8,
) -> torch.Tensor:
    t = t_normalized.view(-1, 1, 1, 1)

    scale = 1.0 + (1.0 - mask) * (bg_boost - 1.0)

    envelope = (t / t_ramp).clamp(0.0, 1.0)
    scale = 1.0 + envelope * (scale - 1.0)

    rms = scale.pow(2).flatten(1).mean(1).sqrt().view(-1, 1, 1, 1)
    return scale / (rms + eps)
