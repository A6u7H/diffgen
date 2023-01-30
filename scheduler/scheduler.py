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
        timestamps: int
    ) -> None:
        super(LinearScheduler, self).__init__()
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timestamps = timestamps

    def get_schedule(self):
        return torch.linspace(
            self.beta_start,
            self.beta_end,
            self.timestamps,
            dtype=torch.float64
        )

class CosineScheduler(Scheduler):
    def __init__(
        self,
        timestamps: int,
        s: int = 0.008
    ) -> None:
        super(CosineScheduler, self).__init__()
        self.timestamps = timestamps
        self.steps = timestamps + 1
        self.s = s

    def get_schedule(self):
        t = torch.linspace(
            0,
            self.timestamps,
            self.steps,
            dtype=torch.float64
        )
        alphas_cumprod = torch.cos(
            (t / self.timestamps + self.s) / (1 + self.s) * math.pi * 0.5
        ) ** 2
        alphas_cumprod /= alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
