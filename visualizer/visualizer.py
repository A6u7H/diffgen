from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from torch import Tensor


class Visualizer:
    @staticmethod
    def plot_scatters(images: Tensor, n_rows: int, n_cols: int):
        count, *_ = images.shape
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 4))
        axs = axs.flatten()
        for i in range(count):
            axs[i].set_title(f"step: {i* 5}")
            axs[i].axis("off")
            axs[i].scatter(images[i, 0, :, 0], images[i, 0, :, 1])
        return fig

    @staticmethod
    def plot_images(images: Tensor, n_rows: int):
        return make_grid(images, n_rows)
