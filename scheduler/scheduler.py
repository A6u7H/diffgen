import torch
import math


class Scheduler:
    def get_schedule(self):
        raise NotImplementedError()


class LinearScheduler(Scheduler):
    def __init__(
        self,
        beta_start: int,
        beta_end: int,
        timesteps: int
    ) -> None:
        super(LinearScheduler, self).__init__()
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps

    def get_schedule(self):
        return torch.linspace(
            self.beta_start,
            self.beta_end,
            self.timesteps,
            dtype=torch.float64
        )


class CosineScheduler(Scheduler):
    def __init__(
        self,
        timesteps: int,
        s: int = 0.008
    ) -> None:
        super(CosineScheduler, self).__init__()
        self.timesteps = timesteps
        self.steps = timesteps + 1
        self.s = s

    def get_schedule(self):
        t = torch.linspace(
            0,
            self.timesteps,
            self.steps,
            dtype=torch.float64
        )
        alphas_cumprod = torch.cos(
            (t / self.timesteps + self.s) / (1 + self.s) * math.pi * 0.5
        ) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
