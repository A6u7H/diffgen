import torch
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
        if time == 0:
            return 0

        variance = extract(self.betas, time, image_shape) \
            * (1. - extract(self.alpha_cumprod_prev, time, image_shape)) \
            / (1. - extract(self.alpha_cumprod, time, image_shape))

        variance = variance.clip(1e-20)
        return variance

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        noise: torch.Tensor,
        time: torch.Tensor
    ):
        """
        x_s = 1 / sqrt(alpha_t') * (x_t - sqrt(1 - alpha_t') * noise)
        explanation: https://theaisummer.com/diffusion-models/?fbclid=IwAR1BIeNHqa3NtC8SL0sKXHATHklJYphNH-8IGNoO3xZhSKM_GYcvrrQgB0o
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, time, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, time, x_t.shape) * noise
        )

    def predict_noise_from_start(
        self,
        x_s: torch.Tensor,
        x_t: torch.Tensor,
        time: torch.Tensor
    ):
        """
        noise = 1 / sqrt(alpha_t') * (x_t - sqrt(1 - alpha_t') * noise)
        explanation: https://theaisummer.com/diffusion-models/?fbclid=IwAR1BIeNHqa3NtC8SL0sKXHATHklJYphNH-8IGNoO3xZhSKM_GYcvrrQgB0o
        """
        return (
            (extract(self.sqrt_recip_alphas_cumprod, time, x_t.shape) * x_t - x_s) /
            extract(self.sqrt_recipm1_alphas_cumprod, time, x_t.shape)
        )

    def q_posterior(
        self,
        x_s: torch.Tensor,
        x_t: torch.Tensor,
        time: torch.Tensor
    ):
        term_s = extract(self.posterior_mean_coef1, time, x_s.shape)
        term_t = extract(self.posterior_mean_coef1, time, x_t.shape)
        return term_s * x_s + term_t * x_t

    def model_predictions(
        self,
        model: Callable,
        image: torch.Tensor,
        time: torch.Tensor,
    ):
        model_output = model(image, time)
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(image, time, pred_noise)
        elif self.objective == 'pred_xs':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(image, time, x_start)

        return Prediction(pred_noise, x_start)

    def q_sample(
        self,
        image: torch.Tensor,
        time: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(image)

        alpha_cumprod_t = extract(self.alpha_cumprod, time, image.shape)
        sqrt_alpha_t = torch.sqrt(alpha_cumprod_t)
        sqrt_alpha_m1_t = torch.sqrt(1 - alpha_cumprod_t)
        return sqrt_alpha_t * image + sqrt_alpha_m1_t * noise, noise

    def p_mean_variance(
        self,
        model: Callable,
        image: torch.Tensor,
        time: torch.Tensor,
    ):
        preds = self.model_predictions(model, image, time)
        x_start = preds.pred_x_start

        model_mean = self.q_posterior(
            image_s=x_start,
            image_t=image,
            time=time
        )

        model_variance = self.get_variance(time)
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
        noise = torch.randn_like(image) if time > 0 else 0
        pred_prev_img = model_mean + (model_variance ** 0.5) * noise  # (0.5 * model_log_variance).exp() * noise
        return pred_prev_img

    def p_sample_loop(
        self,
        model: Callable,
        shape: Tuple[int],
        device: str = "cuda"
    ):
        image = torch.randn(shape, device=device)
        image_list = [image]

        for time in range(self.num_timesteps, -1, -1):
            img = self.p_sample(model, image, time)
            image_list.append(img)

        result = torch.stack(image_list, dim=1)
        return result
