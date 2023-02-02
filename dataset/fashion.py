import torchvision

from torch.utils.data import Dataset
from typing import Callable


class FashionDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        train: bool,
        download: bool,
        transform: Callable
    ) -> None:
        super(FashionDataset, self).__init__()
        self.trainset = torchvision.datasets.FashionMNIST(
            root=data_root,
            train=train,
            download=download,
            transform=transform
        )

    def __getitem__(self, idx):
        return self.trainset[idx]

    def __len__(self):
        return len(self.trainset)
