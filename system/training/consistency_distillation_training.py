#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training/consistency_distillation_training.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Distills a trained teacher (VE diffusion or Rectified Flow, chosen
    with --teacher) into a consistency model, following Song et al.,
    2023, Algorithm 2 (see `consistency/distillation.py`). Produces
    either checkpoints/mnist_all_cm_ve.pth or
    checkpoints/mnist_all_cm_rf.pth depending on --teacher.

    TODO: fill in the training loop:
      1. Load the corresponding teacher checkpoint (mnist_all_ve.pth /
         mnist_all_rf.pth from `training/ve_training.py` /
         `training/rectified_flow_training.py`).
      2. Build the matching TeacherAdapter (consistency/teachers.py).
      3. Build the online + EMA ConsistencyWrapper models (both
         initialized from the same backbone weights).
      4. Loop: consistency_distillation_loss + optimizer step + EMA
         update of the target model.

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
TEACHER: Literal["ve", "rf"] = "ve"
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
    teacher: Literal["ve", "rf"] = TEACHER
    digit: Optional[int] = DIGIT
    batch_size: int = BATCH_SIZE
    n_iterations: int = N_ITERATIONS
    lr: float = LEARNING_RATE
    seed: int = SEED
    log_every: int = LOG_EVERY_N_BATCHES
    data_dir: str = DATA_DIR
    checkpoint_dir: str = CHECKPOINT_DIR

    @property
    def teacher_run_name(self) -> str:
        base = f"mnist_digit{self.digit}" if self.digit is not None else "mnist_all"
        return f"{base}_{self.teacher}"

    @property
    def teacher_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.teacher_run_name}.pth")

    @property
    def run_name(self) -> str:
        base = f"mnist_digit{self.digit}" if self.digit is not None else "mnist_all"
        return f"{base}_cm_{self.teacher}"

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.run_name}.pth")


def parse_args() -> TrainConfig:
    """Parses CLI overrides for the CONFIG block above."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher", choices=["ve", "rf"], default=TEACHER,
        help="Which pretrained teacher to distill from."
    )
    parser.add_argument("--digit", type=int, default=DIGIT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    return TrainConfig(
        teacher=args.teacher,
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
        "Distilling '%s' -> '%s' | digit=%s | images=%d | device=%s",
        config.teacher_checkpoint_path, config.run_name,
        config.digit, len(loader.dataset), device,
    )

    raise NotImplementedError(
        "TODO: load teacher checkpoint, build TeacherAdapter "
        "(consistency/teachers.py) matching config.teacher, build "
        "online + EMA ConsistencyWrapper models, and run the "
        "consistency_distillation_loss training loop."
    )


if __name__ == "__main__":
    train(parse_args())
