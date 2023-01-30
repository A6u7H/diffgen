import torch
import torch.nn as nn

from scheduler import Scheduler


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
        time: int
    ) -> torch.Tensor:
        sqrt_alpha = torch.sqrt(self.alpha_cumprod[time])[:, None, None, None]
        sqrt_alpha_m1 = torch.sqrt(1 - self.alpha_cumprod[time])[:, None, None, None]
        noise = torch.randn_like(image)
        return sqrt_alpha * image + sqrt_alpha_m1 * noise, noise
