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

    TODO: fill in the training loop (mirrors `ve_training.py::train`,
    swapping `VarianceExplodingDiffusion`+`ScoreNet` for
    `RectifiedFlow`+`VelocityNet`).

License:
    Proprietary. All rights reserved.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader

from data import get_dataloader
from flow_matching.rectified_flow import RectifiedFlow
from models.velocity_unet import VelocityNet
from utils import get_device, get_logger, set_seed


# ============================== CONFIG ==============================
DIGIT: Optional[int] = None

BATCH_SIZE = 64
N_EPOCHS = 25
LEARNING_RATE = 1.0e-3
SIGMA_DATA = 1.0        # Rectified Flow noise-endpoint standard deviation
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
    sigma_data: float = SIGMA_DATA
    seed: int = SEED
    log_every: int = LOG_EVERY_N_BATCHES
    data_dir: str = DATA_DIR
    checkpoint_dir: str = CHECKPOINT_DIR

    @property
    def run_name(self) -> str:
        return f"mnist_digit{self.digit}_rf" if self.digit is not None else "mnist_all_rf"

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.run_name}.pth")


def parse_args() -> TrainConfig:
    """Parses CLI overrides for the CONFIG block above."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--digit", type=int, default=DIGIT,
        help="Train on a single MNIST digit (0-9). Omit for the whole dataset."
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--sigma-data", type=float, default=SIGMA_DATA)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    return TrainConfig(
        digit=args.digit,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        lr=args.lr,
        sigma_data=args.sigma_data,
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

    process = RectifiedFlow(sigma_data=config.sigma_data)
    model = VelocityNet().to(device)

    raise NotImplementedError(
        "TODO: optimizer + training loop, mirroring "
        "training/ve_training.py::train with RectifiedFlow.loss_function"
    )


if __name__ == "__main__":
    train(parse_args())
