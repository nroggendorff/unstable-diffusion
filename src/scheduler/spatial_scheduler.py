import torch
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

    def get_x0_target(self, x_t, noise, t, noise_scale=None):
        alphas = self.alphas_cumprod.to(x_t.device)

        a = alphas[t.long()].view(-1, 1, 1, 1).sqrt()
        b = (1 - alphas[t.long()]).view(-1, 1, 1, 1).sqrt()

        if noise_scale is not None:
            noise = noise * noise_scale

        return (x_t - b * noise) / a

    def predict_noise_from_x0(self, x_t, x0, t, noise_scale=None):
        alphas = self.alphas_cumprod.to(x_t.device)

        a = alphas[t.long()].view(-1, 1, 1, 1).sqrt()
        b = (1 - alphas[t.long()]).view(-1, 1, 1, 1).sqrt()

        if noise_scale is not None:
            b = b * noise_scale

        return (x_t - a * x0) / (b + 1e-6)


def compute_spatial_noise_scale(
    mask: torch.Tensor,
    t_normalized: torch.Tensor,
    subject_power: float = 0.6,
    bg_scale: float = 0.75,
    min_scale: float = 0.0,
) -> torch.Tensor:
    t = t_normalized.view(-1, 1, 1, 1)
    sigma_subject = min_scale + (1.0 - min_scale) * t**subject_power
    sigma_background = min_scale + (bg_scale - min_scale) * t
    spatial = mask * sigma_subject + (1 - mask) * sigma_background

    uniform = min_scale + (1.0 - min_scale) * t
    envelope = 4.0 * t * (1.0 - t)

    return envelope * spatial + (1.0 - envelope) * uniform
