#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training/consistency_training_v2.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Second version of `consistency_training.py` (CT v1, which stays as it
    was): trains a consistency model directly from data, closer to the
    CIFAR-10 setup of Song et al. (2023) and with the budget of CD. Produces
    checkpoints/mnist_all_cm_ve_ct_v2.pth, so it never overwrites the CT v1
    checkpoint (a separate EMA of the online
    network kept only for sampling, as in Song et al., 2023, App. C; same
    format as the distillation checkpoints, so every sampling/comparison
    script loads it unchanged). The learning rate is constant (the paper
    uses no decay).

    Compute control: the N(k)/mu(k) curriculum only works if it runs to
    completion (N must reach its final value, or multistep sampling
    degrades), so stopping a run early is not an option. Instead,
    --budget-minutes M benchmarks the speed of this machine for a few
    iterations and sets the total iteration count so the whole run
    (curriculum included) fits in about M minutes. Without it, the
    run uses --iterations. Progress is checkpointed every
    --checkpoint-every iterations and on Ctrl+C, and can be continued with
    --resume (the iteration count is stored in the checkpoint, so the
    curriculum stays consistent).

    CT needs far more iterations than distillation to reach the same
    quality (there is no teacher telling it where each noisy sample
    should go), so expect noticeably worse samples than the CD model for
    the same budget.

License:
    Proprietary. All rights reserved.
"""

import argparse
import copy
import os
import sys
import time
from dataclasses import dataclass
from typing import Literal, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from consistency.ct_loss import consistency_training_loss
from consistency.distances import pseudo_huber_distance, squared_l2_distance
from consistency.schedules import DiscretizationSchedule, EMADecaySchedule
from data import get_dataloader
from diffusion.ve import VarianceExplodingDiffusion
from models.consistency_wrapper import ConsistencyWrapper
from models.score_unet import ScoreNet
from utils import get_device, get_logger, set_seed


# ============================== CONFIG ==============================
DIGIT: Optional[int] = None

BATCH_SIZE = 64
N_ITERATIONS = 40_000    # same as the CD runs
BUDGET_MINUTES: Optional[float] = None   # overrides N_ITERATIONS when set
LEARNING_RATE = 4.0e-4
MU0 = 0.9                # target EMA base decay, CIFAR-10 value of the paper
SAMPLE_EMA_DECAY = 0.9995  # paper: 0.9999 over 800k iterations; shorter run here
SIGMA = 25.0             # VE base sigma, same as the teacher / CD runs
SIGMA_DATA = 0.5         # std of MNIST in [-1, 1], used by ConsistencyWrapper
EPS = 1.0e-5
DISTANCE: Literal["huber", "l2"] = "huber"
DISTANCE_FNS = {"huber": pseudo_huber_distance, "l2": squared_l2_distance}
SEED = 42
LOG_EVERY_N_BATCHES = 100
CHECKPOINT_EVERY = 1000
BENCHMARK_ITERATIONS = 10
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
# ======================================================================


@dataclass
class TrainConfig:
    digit: Optional[int] = DIGIT
    batch_size: int = BATCH_SIZE
    n_iterations: int = N_ITERATIONS
    budget_minutes: Optional[float] = BUDGET_MINUTES
    lr: float = LEARNING_RATE
    mu0: float = MU0
    sample_ema_decay: float = SAMPLE_EMA_DECAY
    sigma: float = SIGMA
    sigma_data: float = SIGMA_DATA
    eps: float = EPS
    distance: Literal["huber", "l2"] = DISTANCE
    seed: int = SEED
    log_every: int = LOG_EVERY_N_BATCHES
    checkpoint_every: int = CHECKPOINT_EVERY
    resume: bool = False
    tag: Optional[str] = None
    data_dir: str = DATA_DIR
    checkpoint_dir: str = CHECKPOINT_DIR

    @property
    def run_name(self) -> str:
        base = f"mnist_digit{self.digit}" if self.digit is not None else "mnist_all"
        name = f"{base}_cm_ve_ct_v2"
        return f"{name}_{self.tag}" if self.tag else name

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.run_name}.pth")

    @property
    def resume_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.run_name}_resume.pth")


def parse_args() -> TrainConfig:
    """Parses CLI overrides for the CONFIG block above."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digit", type=int, default=DIGIT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS)
    parser.add_argument(
        "--budget-minutes", type=float, default=BUDGET_MINUTES,
        help="Size the run (iteration count) to take about this many "
             "minutes on this machine; overrides --iterations."
    )
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--mu0", type=float, default=MU0)
    parser.add_argument("--sample-ema-decay", type=float, default=SAMPLE_EMA_DECAY)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--sigma-data", type=float, default=SIGMA_DATA)
    parser.add_argument("--distance", choices=["huber", "l2"], default=DISTANCE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from <checkpoint-dir>/<run_name>_resume.pth if it exists."
    )
    parser.add_argument(
        "--tag", default=None,
        help="Optional suffix for the run name (e.g. --tag test)."
    )
    args = parser.parse_args()
    return TrainConfig(
        digit=args.digit,
        batch_size=args.batch_size,
        n_iterations=args.iterations,
        budget_minutes=args.budget_minutes,
        lr=args.lr,
        mu0=args.mu0,
        sample_ema_decay=args.sample_ema_decay,
        sigma=args.sigma,
        sigma_data=args.sigma_data,
        distance=args.distance,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        tag=args.tag,
    )


def build_dataloader(config: TrainConfig) -> DataLoader:
    """Loads MNIST, optionally filtered down to a single digit."""
    return get_dataloader(
        batch_size=config.batch_size,
        digit=config.digit,
        train=True,
        data_dir=config.data_dir,
    )


def build_consistency_models(config: TrainConfig, process, device: torch.device):
    """Builds the online consistency model and its frozen EMA target copy.

    Both wrap a fresh, randomly initialised raw ScoreNet backbone: with
    no teacher there is nothing to initialise from.
    """
    cm = ConsistencyWrapper(
        backbone=ScoreNet(marginal_prob_std=None),
        sigma_data=config.sigma_data,
        eps=config.eps,
        sigma_t=process.sigma_t,
    ).to(device)
    cm_ema = copy.deepcopy(cm)
    for p in cm_ema.parameters():
        p.requires_grad_(False)
    return cm, cm_ema


def update_ema(cm_ema: torch.nn.Module, cm: torch.nn.Module, mu: float) -> None:
    """In-place EMA update: theta- <- mu*theta- + (1-mu)*theta."""
    with torch.no_grad():
        for p_ema, p in zip(cm_ema.parameters(), cm.parameters()):
            p_ema.mul_(mu).add_(p, alpha=1.0 - mu)


def atomic_save(obj, path: str) -> None:
    """Saves through a temporary file, so an interrupted save keeps the old one."""
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def save_training_state(
    config: TrainConfig, cm, cm_ema, cm_sample_ema, optimizer,
    iteration: int, n_iterations: int,
) -> None:
    """Saves the sampling checkpoint and everything --resume needs.

    Args:
        iteration: Last fully completed iteration.
    """
    atomic_save(cm_sample_ema.state_dict(), config.checkpoint_path)
    atomic_save(
        {
            "cm": cm.state_dict(),
            "cm_ema": cm_ema.state_dict(),
            "cm_sample_ema": cm_sample_ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "n_iterations": n_iterations,
        },
        config.resume_checkpoint_path,
    )


def seconds_per_iteration(
    cm, cm_ema, process, loader, config: TrainConfig, device: torch.device
) -> float:
    """Times a few forward+backward passes, without touching any state.

    No optimizer step and no EMA update is made, so the models are left
    exactly as they were. Uses the largest N (most expensive grid
    construction) so the estimate is conservative.
    """
    schedule = DiscretizationSchedule(K=1)  # N(k=K) = s1 + 1, the maximum
    distance_fn = DISTANCE_FNS[config.distance]
    data_iter = iter(loader)
    cm.train()
    start = None
    for i in range(BENCHMARK_ITERATIONS + 2):
        if i == 2:  # first two passes are warm-up
            start = time.perf_counter()
        x, _ = next(data_iter)
        loss = consistency_training_loss(
            cm, cm_ema, process, x.to(device), schedule, 1, distance_fn, config.eps
        )
        loss.backward()
        cm.zero_grad(set_to_none=True)
    return (time.perf_counter() - start) / BENCHMARK_ITERATIONS


def train(config: TrainConfig) -> None:
    logger = get_logger(name=config.run_name)
    set_seed(config.seed)
    device = get_device()

    loader = build_dataloader(config)
    process = VarianceExplodingDiffusion(sigma=config.sigma)
    cm, cm_ema = build_consistency_models(config, process, device)
    cm_sample_ema = copy.deepcopy(cm_ema)
    optimizer = optim.Adam(cm.parameters(), lr=config.lr)
    distance_fn = DISTANCE_FNS[config.distance]

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    if not config.resume and os.path.exists(config.resume_checkpoint_path):
        # A fresh run would overwrite that progress at its first checkpoint.
        raise SystemExit(
            f"{config.resume_checkpoint_path} already exists: pass --resume to "
            "continue that run, --tag NAME to start a separate one, or delete "
            "it to start over."
        )

    start_iteration = 0
    n_iterations = config.n_iterations
    resumed = config.resume and os.path.exists(config.resume_checkpoint_path)
    if resumed:
        checkpoint = torch.load(config.resume_checkpoint_path, map_location=device)
        cm.load_state_dict(checkpoint["cm"])
        cm_ema.load_state_dict(checkpoint["cm_ema"])
        cm_sample_ema.load_state_dict(checkpoint["cm_sample_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = checkpoint["iteration"] + 1
        n_iterations = checkpoint["n_iterations"]
        logger.info(
            "Resumed from %s at iteration %d/%d",
            config.resume_checkpoint_path, start_iteration, n_iterations,
        )
    else:
        if config.resume:
            logger.warning(
                "No checkpoint at %s: starting from scratch",
                config.resume_checkpoint_path,
            )
        if config.budget_minutes is not None:
            sec_per_it = seconds_per_iteration(cm, cm_ema, process, loader, config, device)
            n_iterations = max(1000, int(config.budget_minutes * 60 / sec_per_it))
            logger.info(
                "Budget %.0f min at %.3f s/iter -> %d iterations",
                config.budget_minutes, sec_per_it, n_iterations,
            )

    logger.info(
        "Consistency training '%s' | digit=%s | images=%d | iterations=%d | "
        "distance=%s | device=%s",
        config.run_name, config.digit, len(loader.dataset), n_iterations,
        config.distance, device,
    )

    discretization_schedule = DiscretizationSchedule(K=n_iterations)
    ema_schedule = EMADecaySchedule(discretization_schedule, mu0=config.mu0)

    data_iter = iter(loader)
    cm.train()
    running_loss = 0.0
    run_start = time.perf_counter()
    last_completed = start_iteration - 1
    try:
        for iteration in range(start_iteration, n_iterations):
            try:
                x, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, _ = next(data_iter)
            x = x.to(device)

            optimizer.zero_grad()
            loss = consistency_training_loss(
                cm, cm_ema, process, x, discretization_schedule, iteration,
                distance_fn, config.eps,
            )
            loss.backward()
            optimizer.step()

            mu = ema_schedule(iteration)
            update_ema(cm_ema, cm, mu)
            update_ema(cm_sample_ema, cm, config.sample_ema_decay)
            last_completed = iteration

            running_loss += loss.item()

            if (iteration + 1) % config.log_every == 0:
                done = iteration + 1 - start_iteration
                eta_min = (time.perf_counter() - run_start) / done * (n_iterations - iteration - 1) / 60
                logger.info(
                    "[%d/%d] N=%d mu=%.5f - avg loss (last %d): %.4f | ETA %.0f min",
                    iteration + 1, n_iterations,
                    discretization_schedule(iteration), mu,
                    config.log_every, running_loss / config.log_every, eta_min,
                )
                running_loss = 0.0

            if iteration > 0 and iteration % config.checkpoint_every == 0:
                save_training_state(
                    config, cm, cm_ema, cm_sample_ema, optimizer, iteration, n_iterations
                )
                logger.info("Checkpoint saved at iteration %d", iteration)
    except KeyboardInterrupt:
        save_training_state(
            config, cm, cm_ema, cm_sample_ema, optimizer, last_completed, n_iterations
        )
        logger.info(
            "Interrupted: saved at iteration %d/%d, continue with --resume",
            last_completed + 1, n_iterations,
        )
        return

    save_training_state(
        config, cm, cm_ema, cm_sample_ema, optimizer, n_iterations - 1, n_iterations
    )
    logger.info("Consistency model (sampling EMA) saved to %s", config.checkpoint_path)


if __name__ == "__main__":
    train(parse_args())
