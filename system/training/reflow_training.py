#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training/reflow_training.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Reflow (Liu et al., 2022, Sec. 2.3): turns the trained 1-Rectified
    Flow teacher into a 2-Rectified Flow with straighter trajectories,
    which is a better teacher for consistency distillation (fewer
    errors in 1-step generation, see the paper's Sec. 5).

    1. Pair generation: draws noise x_1 ~ N(0, sigma_noise^2 I) and
       integrates the 1-RF ODE from t=1 to t=0 (Euler, PAIR_STEPS
       steps) to get the image x_0 it maps to. The (x_0, x_1) pairs are
       saved to <checkpoint_dir>/<pairs_name>.pt, so this (slow) step
       runs only once; later runs reuse the file.
    2. Reflow training: trains a VelocityNet with the usual RF loss
       (`flow_matching/rectified_flow.py::loss_function`) but on those
       pairs instead of independently drawn noise, i.e. on the coupling
       the 1-RF induces. Its paths cross less, so the learned velocity
       field is closer to straight and few-step sampling improves.

    Training starts from the 1-RF weights (as in the official Rectified
    Flow code) rather than from scratch, so it converges in a few
    epochs. Like `rectified_flow_training.py`, it saves both the raw
    weights (<run_name>.pth) and an EMA copy (<run_name>_ema.pth).

    The 2-RF is then distilled into a consistency model with:
        python system/training/consistency_distillation_training.py \\
            --teacher rf --teacher-checkpoint ./checkpoints_rf/mnist_all_2rf_ema.pth \\
            --run-suffix _reflow

License:
    Proprietary. All rights reserved.
"""

import argparse
import copy
import os
import sys
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from flow_matching.rectified_flow import RectifiedFlow
from models.velocity_unet import VelocityNet
from training.rectified_flow_training import update_ema
from utils import get_device, get_logger, set_seed


# ============================== CONFIG ==============================
# 1-RF teacher whose coupling is reflowed (EMA weights, the better one).
TEACHER_CHECKPOINT = "./checkpoints_rf/mnist_all_rf_ema.pth"
# Weights the 2-RF starts from (None -> train from scratch).
INIT_CHECKPOINT = "./checkpoints_rf/mnist_all_rf_ema.pth"
N_PAIRS = 60000  # same size as the MNIST training set
PAIR_STEPS = 100  # Euler steps used to solve the 1-RF ODE for each pair
PAIR_BATCH_SIZE = 500

BATCH_SIZE = 64
N_EPOCHS = 10
LEARNING_RATE = 2.0e-4  # lower than 1-RF's 1e-3: fine-tuning from it
SIGMA_NOISE = 1.0  # must match the 1-RF teacher's sigma_noise
EMA_DECAY = 0.999
SEED = 42
LOG_EVERY_N_BATCHES = 100
CHECKPOINT_DIR = "./checkpoints_rf"
PAIRS_NAME = "reflow_pairs"
RUN_NAME = "mnist_all_2rf"
# ======================================================================


@dataclass
class TrainConfig:
    teacher_checkpoint: str = TEACHER_CHECKPOINT
    init_checkpoint: str = INIT_CHECKPOINT
    n_pairs: int = N_PAIRS
    pair_steps: int = PAIR_STEPS
    pair_batch_size: int = PAIR_BATCH_SIZE
    batch_size: int = BATCH_SIZE
    n_epochs: int = N_EPOCHS
    lr: float = LEARNING_RATE
    sigma_noise: float = SIGMA_NOISE
    ema_decay: float = EMA_DECAY
    seed: int = SEED
    log_every: int = LOG_EVERY_N_BATCHES
    checkpoint_dir: str = CHECKPOINT_DIR
    pairs_name: str = PAIRS_NAME
    run_name: str = RUN_NAME

    @property
    def pairs_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.pairs_name}.pt")

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.run_name}.pth")

    @property
    def ema_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.run_name}_ema.pth")


def parse_args() -> TrainConfig:
    """Parses CLI overrides for the CONFIG block above."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", default=TEACHER_CHECKPOINT)
    parser.add_argument(
        "--init-checkpoint", default=INIT_CHECKPOINT,
        help="Initial weights for the 2-RF; pass '' to train from scratch."
    )
    parser.add_argument("--n-pairs", type=int, default=N_PAIRS)
    parser.add_argument("--pair-steps", type=int, default=PAIR_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--sigma-noise", type=float, default=SIGMA_NOISE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--pairs-name", default=PAIRS_NAME,
        help="Pairs file name (without .pt) inside the checkpoint dir."
    )
    parser.add_argument("--run-name", default=RUN_NAME)
    args = parser.parse_args()
    return TrainConfig(
        teacher_checkpoint=args.teacher_checkpoint,
        init_checkpoint=args.init_checkpoint,
        n_pairs=args.n_pairs,
        pair_steps=args.pair_steps,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        lr=args.lr,
        sigma_noise=args.sigma_noise,
        seed=args.seed,
        pairs_name=args.pairs_name,
        run_name=args.run_name,
    )


@torch.no_grad()
def generate_pairs(
    config: TrainConfig, process: RectifiedFlow, device: torch.device, logger
) -> None:
    """Solves the 1-RF ODE from fresh noise and saves the (x_0, x_1) pairs.

    Same Euler integration as `sampling/rectified_flow_sampler.py`, but
    x_0 is kept raw (in [-1, 1], not clamped nor rescaled to [0, 1]),
    since it is a training target, not an image to display. Both
    tensors are stored as float16 to halve the file size (~190 MB for
    60k pairs).
    """
    teacher = VelocityNet().to(device)
    teacher.load_state_dict(torch.load(config.teacher_checkpoint, map_location=device))
    teacher.eval()

    t = torch.linspace(process.T, 0.0, config.pair_steps + 1, device=device)
    dt = t[1:] - t[:-1]

    xs_0, xs_1 = [], []
    for start in range(0, config.n_pairs, config.pair_batch_size):
        n = min(config.pair_batch_size, config.n_pairs - start)
        x_1 = process.sample_prior((n, 1, 28, 28), device=device)
        x = x_1.clone()
        for i in range(config.pair_steps):
            x = x + dt[i] * teacher(x, t[i].expand(n))
        xs_0.append(x.half().cpu())
        xs_1.append(x_1.half().cpu())
        logger.info("Pairs generated: %d/%d", start + n, config.n_pairs)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save({"x_0": torch.cat(xs_0), "x_1": torch.cat(xs_1)}, config.pairs_path)
    logger.info("Pairs saved to %s", config.pairs_path)


def train(config: TrainConfig) -> None:
    logger = get_logger(name=config.run_name)
    set_seed(config.seed)
    device = get_device()
    process = RectifiedFlow(sigma_noise=config.sigma_noise)

    if os.path.exists(config.pairs_path):
        logger.info("Reusing pairs from %s", config.pairs_path)
    else:
        logger.info(
            "Generating %d pairs with %s (%d Euler steps) | device=%s",
            config.n_pairs, config.teacher_checkpoint, config.pair_steps, device,
        )
        generate_pairs(config, process, device, logger)

    pairs = torch.load(config.pairs_path)
    loader = DataLoader(
        TensorDataset(pairs["x_0"], pairs["x_1"]),
        batch_size=config.batch_size,
        shuffle=True,
    )

    model = VelocityNet().to(device)
    if config.init_checkpoint:
        model.load_state_dict(torch.load(config.init_checkpoint, map_location=device))
    logger.info(
        "Reflow training '%s' | pairs=%d | init=%s | device=%s",
        config.run_name, len(loader.dataset), config.init_checkpoint or None, device,
    )

    ema_model = copy.deepcopy(model)
    ema_model.eval()
    ema_model.requires_grad_(False)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.n_epochs):
        model.train()
        running_loss = 0.0

        for i, (x_0, x_1) in enumerate(loader):
            x_0 = x_0.float().to(device)
            x_1 = x_1.float().to(device)

            optimizer.zero_grad()
            loss = process.loss_function(model, x_0, x_1=x_1)
            loss.backward()
            optimizer.step()
            update_ema(ema_model, model, config.ema_decay)

            running_loss += loss.item()

            if i % config.log_every == 0:
                logger.info(
                    "[%d/%d] batch %d - loss: %.4f",
                    epoch + 1, config.n_epochs, i, loss.item(),
                )

        logger.info(
            "Epoch %d - avg loss: %.4f", epoch + 1, running_loss / len(loader)
        )

        # Saved every epoch, so an interrupted run still leaves a usable 2-RF.
        torch.save(model.state_dict(), config.checkpoint_path)
        torch.save(ema_model.state_dict(), config.ema_checkpoint_path)

    logger.info("Model saved to %s", config.checkpoint_path)
    logger.info("EMA model saved to %s", config.ema_checkpoint_path)


if __name__ == "__main__":
    train(parse_args())
