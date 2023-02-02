import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torch import Tensor


class DinoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        num_points: int
    ) -> None:
        super(DinoDataset, self).__init__()
        data = pd.read_csv(data_root, sep="\t")
        data = data[data["dataset"] == "dino"]

        rng = np.random.default_rng(42)
        ix = rng.integers(0, len(data), num_points)
        x = data["x"].iloc[ix].tolist()
        x = np.array(x) + rng.normal(size=len(x)) * 0.15

        y = data["y"].iloc[ix].tolist()
        y = np.array(y) + rng.normal(size=len(x)) * 0.15

        x = (x/54 - 1) * 4
        y = (y/48 - 1) * 4

        self.points = np.stack((x, y), axis=1)

    def __getitem__(self, idx) -> Tensor:
        return self.points[idx]

    def __len__(self):
        return len(self.points)
