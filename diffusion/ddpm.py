import torch
import torch.nn.functional as F

from typing import Optional, Tuple

from scheduler import Scheduler
from .utils import extract


class Diffusion:
    def __init__(
        self,
        scheduler: Scheduler,
        image_size: int,
    ) -> None:
        self.scheduler = scheduler
        self.image_size = image_size
        self.betas = self.scheduler.get_schedule()
        self.alpha = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.alpha_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1],
            (1, 0),
            value=1.
        )

        self.posterior_mean_coef1 = self.betas \
            * torch.sqrt(self.alpha_cumprod_prev) \
            / (1. - self.alphas_cumprod)

        self.posterior_mean_coef2 = torch.sqrt(self.alphas) \
            * (1. - self.alpha_cumprod_prev) \
            / (1. - self.alphas_cumprod)

    def get_variance(self, time: int, image_shape: Tuple[int]):
        if time == 0:
            return 0

        variance = extract(self.betas, time, image_shape) \
            * (1. - extract(self.alpha_cumprod_prev, time, image_shape)) \
            / (1. - extract(self.alpha_cumprod, time, image_shape))

        variance = variance.clip(1e-20)
        return variance

    def q_sample(
        self,
        image: torch.Tensor,
        time: int,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(image)

        alpha_cumprod_t = extract(self.alpha_cumprod, time, image.shape)
        sqrt_alpha_t = torch.sqrt(alpha_cumprod_t)
        sqrt_alpha_m1_t = torch.sqrt(1 - alpha_cumprod_t)
        return sqrt_alpha_t * image + sqrt_alpha_m1_t * noise, noise

    def q_posterior(
        self,
        image_s: torch.Tensor,
        image_t: torch.Tensor,
        time: int
    ):
        term_s = extract(self.posterior_mean_coef1, time, image_s.shape)
        term_t = extract(self.posterior_mean_coef1, time, image_t.shape)
        return term_s * image_s + term_t * image_t

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            image = torch.randn((n, 3, self.img_size, self.img_size))
            for i in reversed(range(1, self.noise_steps)):
                time = torch.tensor([i])
                predicted_noise = model(image, time)

                # alpha = self.alpha[t][:, None, None, None]
                # alpha_hat = self.alpha_hat[t][:, None, None, None]
                # beta = self.beta[t][:, None, None, None]

                alpha_t = extract(self.alpha, time, image.shape)
                alpha_cumprod_t = extract(
                    self.alpha_cumprod,
                    time,
                    image.shape
                )

                mean = q_posterior()
                noise = torch.randn_like(image)
                varience = self.get_variance(time, image.shape)

                x = 1 / torch.sqrt(alpha_t) * (image - ((1 - alpha_t) / (torch.sqrt(1 - alpha_cumprod_t))) * predicted_noise) + torch.sqrt(varience) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
