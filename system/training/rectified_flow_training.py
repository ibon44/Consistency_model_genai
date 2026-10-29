#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training/rectified_flow_training.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Trains a VelocityNet on MNIST under the Rectified Flow process, as
    the second candidate teacher for consistency distillation (mirrors
    `training/ve_training.py` for the VE diffusion teacher).

License:
    Proprietary. All rights reserved.
"""

import argparse
import copy
import os
import sys
from dataclasses import dataclass
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.optim as optim
from flow_matching.rectified_flow import RectifiedFlow
from models.velocity_unet import VelocityNet
from torch.utils.data import DataLoader
from utils import get_device, get_logger, set_seed

from data import get_dataloader

# ============================== CONFIG ==============================
DIGIT: Optional[int] = None

BATCH_SIZE = 64
N_EPOCHS = 40
LEARNING_RATE = 1.0e-3
SIGMA_NOISE = 1.0  # Rectified Flow noise-endpoint standard deviation
SEED = 42
LOG_EVERY_N_BATCHES = 100
EMA_DECAY = 0.999  # beta for exponential moving average of model parameters
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints_rf"
# ======================================================================


@dataclass
class TrainConfig:
    digit: Optional[int] = DIGIT
    batch_size: int = BATCH_SIZE
    n_epochs: int = N_EPOCHS
    lr: float = LEARNING_RATE
    sigma_noise: float = SIGMA_NOISE
    seed: int = SEED
    log_every: int = LOG_EVERY_N_BATCHES
    data_dir: str = DATA_DIR
    checkpoint_dir: str = CHECKPOINT_DIR
    ema_decay: float = EMA_DECAY

    @property
    def run_name(self) -> str:
        return (
            f"mnist_digit{self.digit}_rf" if self.digit is not None else "mnist_all_rf"
        )

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.run_name}.pth")

    @property
    def ema_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.run_name}_ema.pth")


def parse_args() -> TrainConfig:
    """Parses CLI overrides for the CONFIG block above."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--digit",
        type=int,
        default=DIGIT,
        help="Train on a single MNIST digit (0-9). Omit for the whole dataset.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--sigma-noise", type=float, default=SIGMA_NOISE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    args = parser.parse_args()
    return TrainConfig(
        digit=args.digit,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        lr=args.lr,
        sigma_noise=args.sigma_noise,
        seed=args.seed,
        ema_decay=args.ema_decay,
    )


def build_dataloader(config: TrainConfig) -> DataLoader:
    """Loads MNIST, optionally filtered down to a single digit."""
    return get_dataloader(
        batch_size=config.batch_size,
        digit=config.digit,
        train=True,
        data_dir=config.data_dir,
    )


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, decay: float
) -> None:
    """Updates the EMA model parameters with the current model parameters.
    theta_ema = decay * theta_ema + (1 - decay) * theta"""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.mul_(decay).add_(param, alpha=1.0 - decay)


def train(config: TrainConfig) -> None:
    logger = get_logger(name=config.run_name)
    set_seed(config.seed)
    device = get_device()

    loader = build_dataloader(config)
    logger.info(
        "Training run '%s' | digit=%s | images=%d | device=%s",
        config.run_name,
        config.digit,
        len(loader.dataset),
        device,
    )

    process = RectifiedFlow(sigma_noise=config.sigma_noise)
    model = VelocityNet().to(device)

    ema_model = copy.deepcopy(model)
    ema_model.eval()  # EMA model is used for evaluation, not training
    ema_model.requires_grad_(False)  # Freeze EMA model parameters

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.n_epochs):
        model.train()
        running_loss = 0.0

        for i, (x, _) in enumerate(loader):
            x = x.to(device)

            optimizer.zero_grad()
            loss = process.loss_function(model, x)
            loss.backward()
            optimizer.step()
            update_ema(ema_model, model, config.ema_decay)

            running_loss += loss.item()

            if i % config.log_every == 0:
                logger.info(
                    "[%d/%d] batch %d - loss: %.4f",
                    epoch + 1,
                    config.n_epochs,
                    i,
                    loss.item(),
                )

        logger.info(
            "Epoch %d - avg loss: %.4f",
            epoch + 1,
            running_loss / len(loader),
        )

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), config.checkpoint_path)
    logger.info("Model saved to %s", config.checkpoint_path)
    torch.save(ema_model.state_dict(), config.ema_checkpoint_path)
    logger.info("EMA model saved to %s", config.ema_checkpoint_path)


if __name__ == "__main__":
    train(parse_args())
