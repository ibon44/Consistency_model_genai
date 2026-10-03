#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training/ve_training.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Trains a ScoreNet on MNIST under the Variance Exploding (VE)
    diffusion process. Edit the CONFIG block below to switch between
    training on the whole dataset or a single digit -- everything else
    (dataloader filtering, checkpoint naming, logging) adapts
    automatically.

License:
    Proprietary. All rights reserved.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import get_dataloader
from diffusion.ve import VarianceExplodingDiffusion
from models.score_unet import ScoreNet
from utils import get_device, get_logger, set_seed


# ============================== CONFIG ==============================
# Set DIGIT to an int in [0, 9] to train on a single MNIST digit, or to
# None to train on the whole dataset. Every other setting below adapts
# to this choice automatically (dataset filtering, checkpoint name).
DIGIT: Optional[int] = None

BATCH_SIZE = 64
N_EPOCHS = 40
LEARNING_RATE = 1.0e-3
SIGMA = 25.0            # VE diffusion process base standard deviation
SEED = 42
LOG_EVERY_N_BATCHES = 100
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
# ======================================================================


@dataclass
class TrainConfig:
    digit: Optional[int] = DIGIT
    batch_size: int = BATCH_SIZE
    n_epochs: int = N_EPOCHS
    lr: float = LEARNING_RATE
    sigma: float = SIGMA
    seed: int = SEED
    log_every: int = LOG_EVERY_N_BATCHES
    data_dir: str = DATA_DIR
    checkpoint_dir: str = CHECKPOINT_DIR

    @property
    def run_name(self) -> str:
        return f"mnist_digit{self.digit}_ve" if self.digit is not None else "mnist_all_ve"

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.run_name}.pth")


def parse_args() -> TrainConfig:
    """Parses CLI overrides for the CONFIG block above.

    All flags are optional; omitting a flag keeps the value hardcoded
    at the top of this file.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--digit", type=int, default=DIGIT,
        help="Train on a single MNIST digit (0-9). Omit for the whole dataset."
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    return TrainConfig(
        digit=args.digit,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        lr=args.lr,
        sigma=args.sigma,
        seed=args.seed,
    )


def build_dataloader(config: TrainConfig) -> DataLoader:
    """Loads MNIST, optionally filtered down to a single digit."""
    return get_dataloader(
        batch_size=config.batch_size,
        digit=config.digit,
        train=True,
        data_dir=config.data_dir,
    )


def train(config: TrainConfig) -> None:
    logger = get_logger(name=config.run_name)
    set_seed(config.seed)
    device = get_device()

    loader = build_dataloader(config)
    logger.info(
        "Training run '%s' | digit=%s | images=%d | device=%s",
        config.run_name, config.digit, len(loader.dataset), device,
    )

    diffusion_process = VarianceExplodingDiffusion(sigma=config.sigma)
    model = ScoreNet(marginal_prob_std=diffusion_process.sigma_t).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.n_epochs):
        model.train()
        running_loss = 0.0

        for i, (x, _) in enumerate(loader):
            x = x.to(device)

            optimizer.zero_grad()
            loss = diffusion_process.loss_function(model, x)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % config.log_every == 0:
                logger.info(
                    "[%d/%d] batch %d - loss: %.4f",
                    epoch + 1, config.n_epochs, i, loss.item(),
                )

        logger.info(
            "Epoch %d - avg loss: %.4f",
            epoch + 1, running_loss / len(loader),
        )

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), config.checkpoint_path)
    logger.info("Model saved to %s", config.checkpoint_path)


if __name__ == "__main__":
    train(parse_args())
