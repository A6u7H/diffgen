import unittest
import numpy as np
import torch
import cv2
import torchvision.transforms as T

from torchvision.utils import make_grid
from PIL import Image

from scheduler import LinearScheduler, CosineScheduler
from diffusion import Diffusion


class TestDataset(unittest.TestCase):
    def test_linearschedule(self):
        image = Image.open("tests/images/1.jpg")
        transpose = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 2) - 1)
        ])

        reverse_transpose = T.Compose([
            T.Lambda(lambda x: (x + 1) * 0.5),
            T.Lambda(lambda x: x.permute(1, 2, 0)),
            T.Lambda(lambda x: (x * 255)),
            T.Lambda(lambda x: x.numpy().astype(np.uint8)),
            # T.ToPILImage()
        ])

        beta_start = 0.0001
        beta_end = 0.02
        timesteps = 100

        scheduler = LinearScheduler(beta_start, beta_end, timesteps)
        diffusion_model = Diffusion(scheduler, 256)

        image_tensor = transpose(image)
        noise_image, noise = diffusion_model.noise_image(
            image_tensor,
            torch.tensor([5, 10, 55, 90])
        )

        alls_stages = make_grid(noise_image)
        alls_stages_image = reverse_transpose(alls_stages)[:, :, ::-1]
        cv2.imwrite("tests/convert/1_convert_linear.jpg", alls_stages_image)


    def test_cosinechedule(self):
        image = Image.open("tests/images/1.jpg")
        transpose = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 2) - 1)
        ])

        reverse_transpose = T.Compose([
            T.Lambda(lambda x: (x + 1) * 0.5),
            T.Lambda(lambda x: x.permute(1, 2, 0)),
            T.Lambda(lambda x: (x * 255)),
            T.Lambda(lambda x: x.numpy().astype(np.uint8)),
            # T.ToPILImage()
        ])

        beta_start = 0.0001
        beta_end = 0.02
        timesteps = 100

        scheduler = CosineScheduler(timesteps)
        diffusion_model = Diffusion(scheduler, 256)

        image_tensor = transpose(image)
        noise_image, noise = diffusion_model.noise_image(
            image_tensor,
            torch.tensor([5, 10, 55, 90])
        )

        alls_stages = make_grid(noise_image)
        alls_stages_image = reverse_transpose(alls_stages)[:, :, ::-1]
        cv2.imwrite("tests/convert/1_convert_cosine.jpg", alls_stages_image)


if __name__ == '__main__':
    unittest.main()
