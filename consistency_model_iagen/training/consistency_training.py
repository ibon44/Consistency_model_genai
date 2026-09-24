#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training/consistency_training.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Extension #1 from the project proposal: trains a consistency model
    directly from data (consistency training, CT), without distilling
    from a pretrained teacher (see `consistency/ct_loss.py`). Only
    attempted once the essential objectives (teacher training +
    distillation + comparison) are complete.

    TODO: fill in the training loop, analogous to
    `training/consistency_distillation_training.py` but using
    `consistency.ct_loss.consistency_training_loss` and no teacher
    checkpoint/adapter.

License:
    Proprietary. All rights reserved.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Literal, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader

from data import get_dataloader
from utils import get_device, get_logger, set_seed


# ============================== CONFIG ==============================
PROCESS: Literal["ve", "rf"] = "ve"
DIGIT: Optional[int] = None

BATCH_SIZE = 64
N_ITERATIONS = 20_000
LEARNING_RATE = 4.0e-4
SEED = 42
LOG_EVERY_N_BATCHES = 100
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
# ======================================================================


@dataclass
class TrainConfig:
    process: Literal["ve", "rf"] = PROCESS
    digit: Optional[int] = DIGIT
    batch_size: int = BATCH_SIZE
    n_iterations: int = N_ITERATIONS
    lr: float = LEARNING_RATE
    seed: int = SEED
    log_every: int = LOG_EVERY_N_BATCHES
    data_dir: str = DATA_DIR
    checkpoint_dir: str = CHECKPOINT_DIR

    @property
    def run_name(self) -> str:
        base = f"mnist_digit{self.digit}" if self.digit is not None else "mnist_all"
        return f"{base}_cm_{self.process}_ct"

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.run_name}.pth")


def parse_args() -> TrainConfig:
    """Parses CLI overrides for the CONFIG block above."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process", choices=["ve", "rf"], default=PROCESS)
    parser.add_argument("--digit", type=int, default=DIGIT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    return TrainConfig(
        process=args.process,
        digit=args.digit,
        batch_size=args.batch_size,
        n_iterations=args.iterations,
        lr=args.lr,
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
        "Consistency training '%s' | digit=%s | images=%d | device=%s",
        config.run_name, config.digit, len(loader.dataset), device,
    )

    raise NotImplementedError(
        "TODO: build online + EMA ConsistencyWrapper models and run "
        "the consistency_training_loss loop (no teacher needed)."
    )


if __name__ == "__main__":
    train(parse_args())
