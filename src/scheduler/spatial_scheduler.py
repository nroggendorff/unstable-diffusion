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


def compute_spatial_noise_scale(
    mask: torch.Tensor,
    t_normalized: torch.Tensor,
    subject_power: float = 1.5,
    bg_scale: float = 0.4,
) -> torch.Tensor:
    t = t_normalized.view(-1, 1, 1, 1)
    sigma_subject = t**subject_power
    sigma_background = bg_scale * t
    return mask * sigma_subject + (1 - mask) * sigma_background
