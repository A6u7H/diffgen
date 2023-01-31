import torch
import torch.nn as nn

from typing import Optional

from scheduler import Scheduler
from .utils import extract


class Diffusion:
    def __init__(
        self,
        scheduler: Scheduler,
        image_size: int,
    ) -> None:
        # super(Diffusion, self).__init__()
        self.scheduler = scheduler
        self.image_size = image_size

        self.beta = self.scheduler.get_schedule()
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

    def noise_image(
        self,
        image: torch.Tensor,
        time: int,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(image)

        alpha_cumprod_t = extract(self.alpha_cumprod, time)
        sqrt_alpha_t = torch.sqrt(alpha_cumprod_t)
        sqrt_alpha_m1_t = torch.sqrt(1 - alpha_cumprod_t)
        return sqrt_alpha_t * image + sqrt_alpha_m1_t * noise, noise
