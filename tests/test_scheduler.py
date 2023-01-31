import unittest
import numpy as np
import torch
import cv2
import torchvision.transforms as T

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
        transpose = ImageTransform((256, 256))
        reverse_transpose = ImageReverseTransform()
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
