import torch
import numpy as np
import torch.nn.functional as F

from typing import Optional, Tuple, Callable
from collections import namedtuple

from scheduler import Scheduler
from .utils import extract


Prediction = namedtuple('Prediction', ['pred_noise', 'pred_x_start'])


class Diffusion:
    def __init__(
        self,
        scheduler: Scheduler,
        image_size: int,
        objective: str
    ) -> None:
        self.scheduler = scheduler
        self.image_size = image_size
        self.objective = objective

        self.betas = self.scheduler.get_schedule()
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1],
            (1, 0),
            value=1.
        )

        self.posterior_mean_coef1 = self.betas \
            * torch.sqrt(self.alphas_cumprod_prev) \
            / (1. - self.alphas_cumprod)

        self.posterior_mean_coef2 = torch.sqrt(self.alphas) \
            * (1. - self.alphas_cumprod_prev) \
            / (1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(
            1. / self.alphas_cumprod
        )
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1. / self.alphas_cumprod - 1
        )

    def get_variance(self, time: int, image_shape: Tuple[int]):
        """
        explanation: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice
        """
        variance = extract(self.betas, time, image_shape) \
            * (1. - extract(self.alphas_cumprod_prev, time, image_shape)) \
            / (1. - extract(self.alphas_cumprod, time, image_shape))

        variance = variance.clip(1e-20)
        return variance

    def predict_start_from_noise(
        self,
        x_time: torch.Tensor,
        noise: torch.Tensor,
        time: torch.Tensor
    ):
        """
        x_s = 1 / sqrt(alpha_t') * (x_t - sqrt(1 - alpha_t') * noise)
        explanation: https://theaisummer.com/diffusion-models/?fbclid=IwAR1BIeNHqa3NtC8SL0sKXHATHklJYphNH-8IGNoO3xZhSKM_GYcvrrQgB0o
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, time, x_time.shape) * x_time -
            extract(self.sqrt_recipm1_alphas_cumprod, time, x_time.shape) * noise
        )

    def predict_noise_from_start(
        self,
        x_start: torch.Tensor,
        x_time: torch.Tensor,
        time: torch.Tensor
    ):
        """
        noise = 1 / sqrt(alpha_t') * (x_t - sqrt(1 - alpha_t') * noise)
        explanation: https://theaisummer.com/diffusion-models/?fbclid=IwAR1BIeNHqa3NtC8SL0sKXHATHklJYphNH-8IGNoO3xZhSKM_GYcvrrQgB0o
        """
        return (
            (extract(self.sqrt_recip_alphas_cumprod, time, x_time.shape) * x_time - x_start) /
            extract(self.sqrt_recipm1_alphas_cumprod, time, x_time.shape)
        )

    def q_posterior(
        self,
        x_start: torch.Tensor,
        x_time: torch.Tensor,
        time: torch.Tensor
    ):
        term_s = extract(self.posterior_mean_coef1, time, x_start.shape)
        term_t = extract(self.posterior_mean_coef2, time, x_time.shape)
        return term_s * x_start + term_t * x_time

    def model_predictions(
        self,
        model: Callable,
        image: torch.Tensor,
        time: torch.Tensor,
    ):
        model_output = model(image, time)
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(image, pred_noise, time)
        elif self.objective == 'pred_xs':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(image, x_start, time)

        return Prediction(pred_noise, x_start)

    def q_sample(
        self,
        image: torch.Tensor,
        time: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(image)

        alphas_cumprod_t = extract(self.alphas_cumprod, time, image.shape)
        sqrt_alphas_t = torch.sqrt(alphas_cumprod_t)
        sqrt_alphas_m1_t = torch.sqrt(1 - alphas_cumprod_t)
        return sqrt_alphas_t * image + sqrt_alphas_m1_t * noise, noise

    def p_mean_variance(
        self,
        model: Callable,
        image: torch.Tensor,
        time: torch.Tensor,
    ):
        preds = self.model_predictions(model, image, time)
        x_start = preds.pred_x_start

        model_mean = self.q_posterior(
            x_start=x_start,
            x_time=image,
            time=time
        )

        model_variance = self.get_variance(time, image.shape)
        return model_mean, model_variance

    def p_sample(
        self,
        model: Callable,
        image: torch.Tensor,
        time: torch.Tensor
    ):
        model_mean, model_variance = self.p_mean_variance(
            model,
            image,
            time
        )

        image_size = image.shape[1:]
        noise = torch.stack([
            torch.randn(image_size) if t > 0 else torch.zeros(image_size)
            for t in time
        ])

        pred_prev_img = model_mean + (model_variance ** 0.5) * noise  # (0.5 * model_log_variance).exp() * noise
        return pred_prev_img

    def p_sample_loop(
        self,
        model: Callable,
        shape: Tuple[int],
        num_timesteps: int,
        device: str = "cuda",
        return_all_timesteps: bool = False
    ):
        sample = torch.randn(shape).to(device)
        timesteps = list(range(num_timesteps))[::-1]

        for time in timesteps:
            time_tensor = torch.from_numpy(np.repeat(time, shape[0])).long()
            sample = self.p_sample(model, sample, time_tensor)
        return sample
