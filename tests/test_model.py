import unittest
import torch
import numpy as np
import cv2

from torchvision.utils import make_grid
from dataset import ImageTransform, ImageReverseTransform
from PIL import Image

from scheduler import LinearScheduler, CosineScheduler
from diffusion import Diffusion
from models import LinNet


class TestModel(unittest.TestCase):
    def test_linmodels(self):
        image_size = 2
        objective = "pred_xs"

        beta_start = 1e-5
        beta_end = 1e-2
        timesteps = 100

        model = LinNet()
        scheduler = LinearScheduler(beta_start, beta_end, timesteps)
        diffusion_model = Diffusion(scheduler, image_size, objective)

        image_tensor = torch.randn(8, 2)
        time = torch.ones(8, dtype=torch.int64) * 5
        noise = torch.randn(8, 2)
        output = model(image_tensor, time)

        self.assertEqual(image_tensor.shape, output.shape)

        output, noise = diffusion_model.q_sample(image_tensor, time, noise)
        self.assertEqual(image_tensor.shape, output.shape)

        output = diffusion_model.p_sample(model, image_tensor, time)
        self.assertEqual(image_tensor.shape, output.shape)

        output = diffusion_model.p_sample_loop(
            model,
            image_tensor.shape,
            timesteps,
            device="cpu"
        )
        self.assertEqual((8, timesteps + 1, 2), output.shape)


if __name__ == '__main__':
    unittest.main()
