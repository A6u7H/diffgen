import torchvision.transforms as T
import numpy as np

from torch import Tensor
from PIL import Image
from typing import Tuple


class ImageTransform:
    def __init__(self, size: Tuple[int, int]) -> None:
        self.transpose = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 2) - 1)  # replace with Normilize()
        ])

    def __call__(self, image: Image) -> Tensor:
        return self.transpose(image)


class ImageReverseTransform:
    def __init__(self) -> None:
        self.transpose = T.Compose([
            T.Lambda(lambda x: (x + 1) * 0.5),
            T.Lambda(lambda x: x.permute(1, 2, 0)),
            T.Lambda(lambda x: (x * 255)),
            T.Lambda(lambda x: x.numpy().astype(np.uint8)),
        ])

    def __call__(self, image: Image) -> Tensor:
        return self.transpose(image)
