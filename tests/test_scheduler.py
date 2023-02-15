import unittest
import torch
import numpy as np
import cv2

from torchvision.utils import make_grid
from dataset import ImageTransform, ImageReverseTransform
from PIL import Image

from scheduler import LinearScheduler, CosineScheduler
from diffusion import Diffusion


class TestDataset(unittest.TestCase):
    def test_linearschedule(self):
        image = Image.open("tests/images/1.jpg")
        transpose = ImageTransform((256, 256))
        reverse_transpose = ImageReverseTransform()
        image_size = 256
        objective = "pred_xs"

        beta_start = 1e-5
        beta_end = 1e-2
        timesteps = 100

        scheduler = LinearScheduler(beta_start, beta_end, timesteps)
        diffusion_model = Diffusion(scheduler, image_size, objective)

        image_tensor = transpose(image).unsqueeze(0)
        indcies = np.arange(0, 100, 10)
        noise_image = torch.stack([
            diffusion_model.q_sample(
                image_tensor,
                torch.tensor(i)
            )[0]
            for i in indcies
        ])

        alls_stages = make_grid(noise_image[:, 0], nrow=10)
        alls_stages_image = reverse_transpose(alls_stages)[:, :, ::-1]
        cv2.imwrite("tests/convert/1_convert_linear.jpg", alls_stages_image)

    def test_linearschedule_step_by_step(self):
        """
        Why the amount of nosie is different from `test_linearschedule`???
        In same settings
        """
        image = Image.open("tests/images/1.jpg")
        transpose = ImageTransform((256, 256))
        reverse_transpose = ImageReverseTransform()

        beta_start = 1e-4
        beta_end = 1e-2
        timesteps = 100

        scheduler = LinearScheduler(beta_start, beta_end, timesteps)
        image_tensor = transpose(image).unsqueeze(0)
        betas = scheduler.get_schedule()

        x_start = image_tensor
        x_seq = [x_start]
        for i in range(timesteps):
            x_seq.append(
                (torch.sqrt(1 - betas[-i]) * x_seq[-1]) +
                (betas[-i] * torch.randn_like(x_start))
            )

        indcies = np.arange(0, 100, 10)
        x_seq = torch.stack(x_seq)[indcies]

        alls_stages = make_grid(x_seq[:, 0], nrow=10)
        alls_stages = reverse_transpose(alls_stages)[:, :, ::-1]

        cv2.imwrite(
            "tests/convert/1_convert_linear_step_by_step.jpg",
            alls_stages
        )

    def test_cosinechedule(self):
        image = Image.open("tests/images/1.jpg")
        transpose = ImageTransform((256, 256))
        reverse_transpose = ImageReverseTransform()
        timesteps = 100
        image_size = 256
        objective = "pred_xs"

        scheduler = CosineScheduler(timesteps)
        diffusion_model = Diffusion(scheduler, image_size, objective)

        image_tensor = transpose(image).unsqueeze(0)
        indcies = np.arange(0, 100, 10)
        noise_images = torch.stack([
            diffusion_model.q_sample(
                image_tensor,
                torch.tensor(i)
            )[0]
            for i in indcies
        ])

        alls_stages = make_grid(noise_images[:, 0], nrow=10)
        alls_stages_image = reverse_transpose(alls_stages)[:, :, ::-1]
        cv2.imwrite("tests/convert/1_convert_cosine.jpg", alls_stages_image)


if __name__ == '__main__':
    unittest.main()
