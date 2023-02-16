import os
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from argparse import Namespace
from tqdm import tqdm
from torch.utils.data import DataLoader

from diffusion import Diffusion
from models import LinNet
from scheduler import LinearScheduler
from dataset import DinoDataset
from aim import Run, Image

logger = logging.getLogger(__name__)

config = Namespace(
    experiment_name="default",
    train_batch_size=32,
    eval_batch_size=1000,
    num_epochs=250,
    learning_rate=1e-3,
    num_timesteps=250,
    beta_start=1e-4,
    beta_end=2e-2,
    beta_schedule="linear",
    objective="pred_xs",
    embedding_size=128,
    hidden_size=128,
    hidden_layers=3,
    save_images_step=1,
    dataset_path="/home/dkrivenkov/program/Diffgen/data/The Datasaurus Dozen/DatasaurusDozen.tsv",
    num_points=8000
)

dataset = DinoDataset(config.dataset_path, config.num_points)
dataloader = DataLoader(
    dataset,
    batch_size=config.train_batch_size,
    shuffle=True,
    drop_last=True
)

model = LinNet(
    hidden_size=config.hidden_size,
    hidden_layers=config.hidden_layers,
    emb_size=config.embedding_size
)

noise_scheduler = LinearScheduler(
    beta_start=config.beta_start,
    beta_end=config.beta_end,
    timesteps=config.num_timesteps
)

diffusion_model = Diffusion(
    noise_scheduler,
    2,
    config.objective,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
)


global_step = 0
frames = []
losses = []

run = Run()

logger.info("Training model...")
for epoch in range(config.num_epochs):
    model.train()
    with tqdm(dataloader) as pbar:
        pbar.set_description(f"Epoch {epoch}")
        for images in pbar:
            timesteps = torch.randint(
                0, diffusion_model.scheduler.timesteps, (images.shape[0],)
            ).long()

            noisy_image, noise = diffusion_model.q_sample(images, timesteps)
            noise_pred = model(noisy_image, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            run.track(
                loss.item(),
                name='loss',
                epoch=epoch,
                context={"subset": "train"}
            )
            loss.backward(loss)

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            logs = {"loss": loss.detach().item(), "step": global_step}

            losses.append(loss.detach().item())
            pbar.set_postfix(**logs)
            global_step += 1

        # if (
        #     epoch % config.save_images_step == 0 or
        #     epoch == config.num_epochs - 1
        # ):
        #     model.eval()
        #     sample = torch.randn(config.eval_batch_size, 2)
        #     sample = diffusion_model.p_sample_loop(
        #         model,
        #         images.shape,
        #         diffusion_model.scheduler.timesteps,
        #         device="cpu",
        #         return_all_timesteps=True
        #     )
        #     frames.append(sample.detach().numpy())

logger.info("Saving images...")
imgdir = "experiments/images"
os.makedirs(imgdir, exist_ok=True)
with torch.no_grad():
    model.eval()
    samples = diffusion_model.p_sample_loop(
        model,
        (1000, 2),
        diffusion_model.scheduler.timesteps,
        device="cpu",
        return_all_timesteps=False
    )

fig, axs = plt.subplots(10, 25, figsize=(50, 50))
axs = axs.flatten()

for i in range(diffusion_model.scheduler.timesteps):
    axs[i].scatter(samples[:, i, 0], samples[:, i, 1])

path = f"{imgdir}/result.png"
fig.savefig(path, )
aim_image = Image(
    path,
    format='png',
    optimize=True,
    quality=90
)
run.track(aim_image, name='images')
plt.clf()

path = f"{imgdir}/original.png"
plt.scatter(dataset[:, 0], dataset[:, 1])
plt.savefig(path)

aim_image = Image(
    path,
    format='png',
    optimize=True,
    quality=90
)
run.track(aim_image, name='original')
plt.clf()

path = f"{imgdir}/last.png"
plt.scatter(samples[:, -1, 0], samples[:, -1, 1])
plt.savefig(path)

aim_image = Image(
    path,
    format='png',
    optimize=True,
    quality=90
)
run.track(aim_image, name='last')
