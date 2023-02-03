import torch
import pytorch_lightning as pl

from torch import Tensor
from typing import Dict

from diffusion import Diffusion
from models import UNet
from scheduler import LinearScheduler


class PointSolver(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = self.configure_model()
        self.loss = self.configure_loss()

    def forward(self, x):
        return self.model(x)

    def configure_loss(self):
        return torch.nn.MSELoss()

    def configure_model(self):
        self.model = UNet(1, 1, 256, "cuda")
        scheduler = LinearScheduler(
            beta_start=1e-5,
            beta_end=2e-2,
            timesteps=200
        )
        self.diffuse_model = Diffusion(scheduler, 32)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=1e-3
        )
        return {
           'optimizer': optimizer
        }

    def loss_fn(
        self,
        pred: Tensor,
        target: Tensor,
        features: Tensor
    ) -> Tensor:
        return self.loss(pred, target, features)

    def metric_fn(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.metric(pred, target)

    def training_step(self, batch: Tensor, batch_idx: int):
        raise NotImplementedError

    def validation_step(self, batch: Tensor, batch_idx: int):
        raise NotImplementedError

    def test_step(self, batch: Tensor, batch_idx: int):
        raise NotImplementedError

    def training_epoch_end(self, outputs) -> None:
        raise NotImplementedError

    def validation_epoch_end(self, outputs) -> None:
        raise NotImplementedError

    def test_epoch_end(self, outputs) -> None:
        raise NotImplementedError

    def log_artifact(self, artifact_path: str):
        self.logger.experiment.log_artifact(self.logger.run_id, artifact_path)
