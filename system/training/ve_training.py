#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training/ve_training.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Trains a ScoreNet on MNIST under the Variance Exploding (VE)
    diffusion process. Edit the CONFIG block below to switch between
    training on the whole dataset or a single digit -- everything else
    (dataloader filtering, checkpoint naming, logging) adapts
    automatically.

    Trains with cosine LR decay and keeps an EMA copy of the weights; the
    saved checkpoint is the EMA model (checkpoints/mnist_all_ve_ema.pth),
    which gives cleaner samples than the raw weights. --ema-rate 0 turns
    EMA off and saves to mnist_all_ve.pth instead.

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
from diffusion.ve import VarianceExplodingDiffusion
from models.score_unet import ScoreNet
from torch.utils.data import DataLoader
from utils import get_device, get_logger, set_seed

from data import get_dataloader

# ============================== CONFIG ==============================
# Set DIGIT to an int in [0, 9] to train on a single MNIST digit, or to
# None to train on the whole dataset. Every other setting below adapts
# to this choice automatically (dataset filtering, checkpoint name).
DIGIT: Optional[int] = None

BATCH_SIZE = 64
N_EPOCHS = 60
LEARNING_RATE = 1.0e-3
SIGMA = 25.0  # VE diffusion process base standard deviation
SEED = 42
EMA_RATE = 0.999  # EMA of the weights, used for the saved checkpoint (0 disables)
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
    ema_rate: float = EMA_RATE
    log_every: int = LOG_EVERY_N_BATCHES
    data_dir: str = DATA_DIR
    checkpoint_dir: str = CHECKPOINT_DIR

    @property
    def run_name(self) -> str:
        base = (
            f"mnist_digit{self.digit}_ve" if self.digit is not None else "mnist_all_ve"
        )
        return f"{base}_ema" if self.ema_rate > 0 else base

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
        "--digit",
        type=int,
        default=DIGIT,
        help="Train on a single MNIST digit (0-9). Omit for the whole dataset.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--ema-rate",
        type=float,
        default=EMA_RATE,
        help="EMA decay of the weights saved to the checkpoint; 0 disables "
        "EMA (and the '_ema' suffix on the checkpoint name).",
    )
    args = parser.parse_args()
    return TrainConfig(
        digit=args.digit,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        lr=args.lr,
        sigma=args.sigma,
        seed=args.seed,
        ema_rate=args.ema_rate,
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
        config.run_name,
        config.digit,
        len(loader.dataset),
        device,
    )

    diffusion_process = VarianceExplodingDiffusion(sigma=config.sigma)
    model = ScoreNet(marginal_prob_std=diffusion_process.sigma_t).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Cosine decay to 0 and an EMA copy of the weights: the saved model
    # is the EMA one. A noisy constant-lr snapshot gives visibly worse
    # samples, especially with the deterministic probability-flow ODE.
    total_steps = config.n_epochs * len(loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    ema_model = copy.deepcopy(model) if config.ema_rate > 0 else model
    step = 0

    for epoch in range(config.n_epochs):
        model.train()
        running_loss = 0.0

        for i, (x, _) in enumerate(loader):
            x = x.to(device)

            optimizer.zero_grad()
            loss = diffusion_process.loss_function(model, x)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if config.ema_rate > 0:
                step += 1
                rate = min(config.ema_rate, (1 + step) / (10 + step))
                with torch.no_grad():
                    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                        p_ema.mul_(rate).add_(p, alpha=1.0 - rate)

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
    torch.save(ema_model.state_dict(), config.checkpoint_path)
    logger.info("Model saved to %s", config.checkpoint_path)


if __name__ == "__main__":
    train(parse_args())
