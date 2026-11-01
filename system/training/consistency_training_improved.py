#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training/consistency_training_improved.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Teacher-free consistency training with the techniques of Song &
    Dhariwal, 2024 (see `consistency/ict_loss.py`): no EMA target
    network, exponential N(k) curriculum, lognormal noise-level sampling
    and 1/(sigma_{n+1}-sigma_n) loss weighting. It is a separate script
    from `consistency_training.py` and writes to different files, so the
    original CT run is left untouched:

        checkpoints/mnist_all_cm_ve_ict[_<tag>].pth         (sampling EMA)
        checkpoints/mnist_all_cm_ve_ict[_<tag>]_resume.pth  (resume state)

    The saved model is a slow EMA of the online weights (--sample-ema),
    used only for sampling; the loss itself has no EMA. Same format as
    the other consistency checkpoints, so `consistency_ct_sample_and_plot.py`
    (pass the path as its first argument) and `evaluation/compare.py`
    load it unchanged.

    Compute control, resume and interruption handling work as in
    `consistency_training.py`: --budget-minutes sizes the run so the
    whole curriculum fits in about that time, progress is checkpointed
    every --checkpoint-every iterations, Ctrl+C saves the state, and
    --resume continues. A run refuses to start over an existing resume
    file unless --resume is given (use --tag for a separate run).

License:
    Proprietary. All rights reserved.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim

from consistency.ict_loss import (
    ImprovedDiscretizationSchedule,
    improved_consistency_training_loss,
)
from consistency_training import (
    atomic_save,
    build_consistency_models,
    build_dataloader,
    update_ema,
)
from diffusion.ve import VarianceExplodingDiffusion
from utils import get_device, get_logger, set_seed


# ============================== CONFIG ==============================
DIGIT: Optional[int] = None

BATCH_SIZE = 64
N_ITERATIONS = 20_000
BUDGET_MINUTES: Optional[float] = None   # overrides N_ITERATIONS when set
LEARNING_RATE = 4.0e-4
SIGMA = 25.0             # VE base sigma, same as the teacher / other runs
SIGMA_DATA = 0.5
EPS = 1.0e-5
S0 = 10                  # N(k) starts at s0 + 1 ...
S1 = 160                 # ... and ends at s1 + 1 (doubling each stage)
P_MEAN = -1.1            # lognormal noise-level sampling (log sigma)
P_STD = 2.0
SAMPLE_EMA_RATE = 0.999  # EMA of the online weights, used for sampling only
GRAD_CLIP = 0.0          # max grad norm; 0 disables clipping
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
    constant_lr: bool = False
    sigma: float = SIGMA
    sigma_data: float = SIGMA_DATA
    eps: float = EPS
    s0: int = S0
    s1: int = S1
    p_mean: float = P_MEAN
    p_std: float = P_STD
    sample_ema: float = SAMPLE_EMA_RATE
    grad_clip: float = GRAD_CLIP
    seed: int = SEED
    log_every: int = LOG_EVERY_N_BATCHES
    checkpoint_every: int = CHECKPOINT_EVERY
    resume: bool = False
    init_from: Optional[str] = None
    tag: Optional[str] = None
    data_dir: str = DATA_DIR
    checkpoint_dir: str = CHECKPOINT_DIR

    @property
    def run_name(self) -> str:
        base = f"mnist_digit{self.digit}" if self.digit is not None else "mnist_all"
        name = f"{base}_cm_ve_ict"
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
    parser.add_argument(
        "--constant-lr", action="store_true",
        help="Keep the learning rate constant instead of the cosine decay."
    )
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--sigma-data", type=float, default=SIGMA_DATA)
    parser.add_argument("--s0", type=int, default=S0)
    parser.add_argument("--s1", type=int, default=S1)
    parser.add_argument("--p-mean", type=float, default=P_MEAN)
    parser.add_argument("--p-std", type=float, default=P_STD)
    parser.add_argument("--sample-ema", type=float, default=SAMPLE_EMA_RATE)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoints/<run_name>_resume.pth if it exists."
    )
    parser.add_argument(
        "--init-from", default=None,
        help="Start from the weights of this consistency-model checkpoint "
             "(e.g. checkpoints/mnist_all_cm_ve_ict.pth) instead of a random "
             "init: the optimizer and the curriculum restart, so combine it "
             "with --s0/--s1 and a lower --lr to keep refining a finished run."
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
        constant_lr=args.constant_lr,
        sigma=args.sigma,
        sigma_data=args.sigma_data,
        s0=args.s0,
        s1=args.s1,
        p_mean=args.p_mean,
        p_std=args.p_std,
        sample_ema=args.sample_ema,
        grad_clip=args.grad_clip,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        init_from=args.init_from,
        tag=args.tag,
    )


def save_checkpoint(
    config: TrainConfig, cm, cm_sample, optimizer, iteration: int, n_iterations: int
) -> None:
    """Saves the sampling checkpoint (EMA) and the resume state.

    `iteration` is the index of the last completed iteration.
    """
    atomic_save(cm_sample.state_dict(), config.checkpoint_path)
    atomic_save(
        {
            "cm": cm.state_dict(),
            "cm_sample": cm_sample.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "n_iterations": n_iterations,
            "s0": config.s0,
            "s1": config.s1,
            "constant_lr": config.constant_lr,
        },
        config.resume_checkpoint_path,
    )


def seconds_per_iteration(cm, process, loader, config: TrainConfig, device) -> float:
    """Times a few forward+backward passes, without touching any state.

    No optimizer step and no EMA update is made. Uses the last curriculum
    stage (largest N) so the estimate is conservative.
    """
    schedule = ImprovedDiscretizationSchedule(K=1000, s0=config.s0, s1=config.s1)
    data_iter = iter(loader)
    cm.train()
    start = None
    for i in range(BENCHMARK_ITERATIONS + 2):
        if i == 2:  # first two passes are warm-up
            start = time.perf_counter()
        x, _ = next(data_iter)
        loss = improved_consistency_training_loss(
            cm, process, x.to(device), schedule, 999, config.eps,
            config.p_mean, config.p_std,
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
    cm, cm_sample = build_consistency_models(config, process, device)
    optimizer = optim.Adam(cm.parameters(), lr=config.lr)

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    resumed = config.resume and os.path.exists(config.resume_checkpoint_path)
    if not config.resume and os.path.exists(config.resume_checkpoint_path):
        # A fresh run would overwrite that progress at its first checkpoint.
        raise SystemExit(
            f"{config.resume_checkpoint_path} already exists: pass --resume to "
            "continue that run, --tag NAME to start a separate one, or delete "
            "it to start over."
        )
    if config.resume and not resumed:
        logger.warning(
            "--resume given but %s does not exist: starting from scratch",
            config.resume_checkpoint_path,
        )

    start_iteration = 0
    n_iterations = config.n_iterations
    if resumed:
        checkpoint = torch.load(config.resume_checkpoint_path, map_location=device)
        cm.load_state_dict(checkpoint["cm"])
        cm_sample.load_state_dict(checkpoint["cm_sample"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = checkpoint["iteration"] + 1
        n_iterations = checkpoint["n_iterations"]
        # The curriculum must be the one the run started with, whatever
        # --s0/--s1 say now.
        config.s0 = checkpoint.get("s0", config.s0)
        config.s1 = checkpoint.get("s1", config.s1)
        config.constant_lr = checkpoint.get("constant_lr", config.constant_lr)
        logger.info(
            "Resumed from %s at iteration %d/%d",
            config.resume_checkpoint_path, start_iteration, n_iterations,
        )
    elif config.init_from is not None:
        weights = torch.load(config.init_from, map_location=device)
        cm.load_state_dict(weights)
        cm_sample.load_state_dict(weights)
        logger.info("Initialised weights from %s", config.init_from)
    if not resumed and config.budget_minutes is not None:
        sec_per_it = seconds_per_iteration(cm, process, loader, config, device)
        n_iterations = max(1000, int(config.budget_minutes * 60 / sec_per_it))
        logger.info(
            "Budget %.0f min at %.3f s/iter -> %d iterations",
            config.budget_minutes, sec_per_it, n_iterations,
        )

    logger.info(
        "Improved consistency training '%s' | digit=%s | images=%d | "
        "iterations=%d | N: %d -> %d | device=%s",
        config.run_name, config.digit, len(loader.dataset), n_iterations,
        config.s0 + 1, config.s1 + 1, device,
    )

    discretization_schedule = ImprovedDiscretizationSchedule(
        K=n_iterations, s0=config.s0, s1=config.s1
    )
    scheduler = None
    if not config.constant_lr:
        for group in optimizer.param_groups:
            # The loaded optimizer carries the already-decayed LR; the cosine
            # schedule is recursive, so fast-forwarding from it below would
            # decay twice. Restart from the base LR.
            group["initial_lr"] = group.get("initial_lr", config.lr)
            group["lr"] = group["initial_lr"]
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_iterations, last_epoch=-1
        )
        for _ in range(start_iteration):  # fast-forward the LR schedule on resume
            scheduler.step()

    data_iter = iter(loader)
    cm.train()
    running_loss = 0.0
    run_start = time.perf_counter()
    last_done = start_iteration - 1  # last fully completed iteration
    try:
        for iteration in range(start_iteration, n_iterations):
            try:
                x, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, _ = next(data_iter)
            x = x.to(device)

            optimizer.zero_grad()
            loss = improved_consistency_training_loss(
                cm, process, x, discretization_schedule, iteration, config.eps,
                config.p_mean, config.p_std,
            )
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(cm.parameters(), config.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # EMA with warmup: a plain slow EMA would keep the random init
            # for thousands of steps.
            rate = min(config.sample_ema, (1 + iteration) / (10 + iteration))
            update_ema(cm_sample, cm, rate)
            last_done = iteration

            running_loss += loss.item()

            if (iteration + 1) % config.log_every == 0:
                done = iteration + 1 - start_iteration
                eta_min = (time.perf_counter() - run_start) / done * (n_iterations - iteration - 1) / 60
                logger.info(
                    "[%d/%d] N=%d - avg loss (last %d): %.4f | ETA %.0f min",
                    iteration + 1, n_iterations,
                    discretization_schedule(iteration),
                    config.log_every, running_loss / config.log_every, eta_min,
                )
                running_loss = 0.0

            if iteration > 0 and iteration % config.checkpoint_every == 0:
                save_checkpoint(config, cm, cm_sample, optimizer, iteration, n_iterations)
                logger.info("Checkpoint saved at iteration %d", iteration)
    except KeyboardInterrupt:
        if last_done >= start_iteration:
            save_checkpoint(config, cm, cm_sample, optimizer, last_done, n_iterations)
        logger.warning(
            "Interrupted at iteration %d/%d: state saved, continue with --resume",
            last_done + 1, n_iterations,
        )
        return

    # Final save also refreshes the resume file, so its iteration counter
    # reads n_iterations-1 (done) instead of the last periodic checkpoint.
    save_checkpoint(config, cm, cm_sample, optimizer, n_iterations - 1, n_iterations)
    logger.info(
        "Training finished (%d/%d). Consistency model (sampling EMA) saved to %s",
        n_iterations, n_iterations, config.checkpoint_path,
    )


if __name__ == "__main__":
    train(parse_args())
