#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training/consistency_distillation_training.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Distills a trained teacher (VE diffusion or Rectified Flow, chosen
    with --teacher) into a consistency model, following Song et al.,
    2023, Algorithm 2 (see `consistency/distillation.py`). Produces
    either checkpoints/mnist_all_cm_ve.pth or
    checkpoints/mnist_all_cm_rf.pth depending on --teacher (the same
    folder, see CHECKPOINT_DIR, is used both to read the teacher weights
    and to save the consistency model).

    Unlike `ve_training.py`, this trains for a fixed number of
    iterations rather than epochs (N(k)/mu(k) in `consistency/
    schedules.py` are schedules over iteration count k, matching the
    paper).

    Default recipe (--schedule fixed): a fixed number of teacher
    discretization points N (Karras grid in sigma) with a Heun teacher
    step, and a fixed target-EMA rate mu. N must be large (150): the
    network only ever sees the N grid times, and with a small N (the
    paper's 18) this small UNet does not generalize to the off-grid
    times that multistep sampling re-noises to, producing garbage for
    >= 2 steps (1 step, always at t=T, looks fine).
    --schedule curriculum uses the CT-style N(k)/mu(k) curriculum
    instead (kept for comparison).

    Three copies of the model are kept:
      - cm (f_theta): the online model, updated by Adam every step.
      - cm_target (f_{theta-}): the EMA *target network* used inside
        the loss, with rate mu.
      - cm_sample: a separate, slower EMA (rate SAMPLE_EMA_RATE) used
        only for sampling, as in the reference implementation; this is
        the checkpoint that gets saved as <run_name>.pth.

    A run of this length (thousands of iterations, potentially hours on
    CPU) can get interrupted, so progress is checkpointed periodically,
    not only at the very end:
      - <checkpoint_dir>/<run_name>.pth is overwritten every
        CHECKPOINT_EVERY iterations with the current sampling EMA, so
        there's always a usable (if not fully converged) model to
        sample from without waiting for the full run to finish.
      - <checkpoint_dir>/<run_name>_resume.pth additionally stores the
        online/target models, optimizer state, iteration count and RNG
        state, so the run itself can be continued via --resume.

License:
    Proprietary. All rights reserved.
"""

import argparse
import copy
import itertools
import os
import random
import sys
from dataclasses import dataclass
from typing import Iterator, Literal, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.optim as optim
from consistency.distances import pseudo_huber_distance, squared_l2_distance
from consistency.distillation import consistency_distillation_loss
from consistency.schedules import (
    ConstantSchedule,
    DiscretizationSchedule,
    EMADecaySchedule,
)
from consistency.teachers import RectifiedFlowTeacherAdapter, VETeacherAdapter
from diffusion.ve import VarianceExplodingDiffusion
from flow_matching.rectified_flow import RectifiedFlow
from models.consistency_wrapper import ConsistencyWrapper
from models.score_unet import ScoreNet
from models.velocity_unet import VelocityNet
from torch.utils.data import DataLoader
from utils import get_device, get_logger, set_seed

from data import get_dataloader

# ============================== CONFIG ==============================
TEACHER: Literal["ve", "rf"] = "ve"
DIGIT: Optional[int] = None

BATCH_SIZE = 64
N_ITERATIONS = 40000
LEARNING_RATE = 4.0e-4
SIGMA = 25.0  # must match the VE teacher's own training sigma
SIGMA_NOISE = 1.0  # must match the RF teacher's own training sigma_noise
SIGMA_DATA = (
    0.5  # value of Song et al., used by ConsistencyWrapper (real MNIST std: ~0.62)
)
EPS = 1.0e-5  # boundary time, shared by teacher adapter + wrapper
SEED = 42
LOG_EVERY_N_BATCHES = 100
CHECKPOINT_EVERY = 1000
DATA_DIR = "./data"
# Checkpoint folder: it holds both the teacher weights that are read and
# the consistency model (+ resume state) that is saved.
CHECKPOINT_DIR = "./checkpoints"
# Teacher weights to distill from. None -> <dir>/<teacher_run_name>.pth
# for VE, and the EMA weights <dir>/<teacher_run_name>_ema.pth for RF
# (rectified_flow_training.py saves both; the EMA copy is the better
# teacher), with <dir> = CHECKPOINT_DIR.
# --teacher-tag ema picks <dir>/<teacher_run_name>_ema.pth for either.
TEACHER_CHECKPOINT: Optional[str] = None
# Distance d(x,y) for the distillation loss ("huber" is the stabler,
# outlier-robust choice from Song & Dhariwal, 2024; "l2" is the
# original Consistency Models paper's choice, kept for comparison).
DISTANCE: Literal["huber", "l2"] = "huber"
DISTANCE_FNS = {"huber": pseudo_huber_distance, "l2": squared_l2_distance}
# "fixed": paper's CD recipe (constant N and mu). "curriculum": CT-style
# N(k)/mu(k) from consistency/schedules.py.
SCHEDULE: Literal["fixed", "curriculum"] = "fixed"
N_SCALES = 150  # dense grid: the net must see (almost) every t, see docstring
TARGET_EMA_RATE = 0.95  # fixed mu for the target network
SAMPLE_EMA_RATE = 0.9999  # separate EMA of the online model, used for sampling
# ======================================================================


@dataclass
class TrainConfig:
    teacher: Literal["ve", "rf"] = TEACHER
    digit: Optional[int] = DIGIT
    batch_size: int = BATCH_SIZE
    n_iterations: int = N_ITERATIONS
    lr: float = LEARNING_RATE
    sigma: float = SIGMA
    sigma_noise: float = SIGMA_NOISE
    sigma_data: float = SIGMA_DATA
    eps: float = EPS
    seed: int = SEED
    log_every: int = LOG_EVERY_N_BATCHES
    checkpoint_every: int = CHECKPOINT_EVERY
    resume: bool = False
    data_dir: str = DATA_DIR
    checkpoint_dir: str = CHECKPOINT_DIR
    distance: Literal["huber", "l2"] = DISTANCE
    schedule: Literal["fixed", "curriculum"] = SCHEDULE
    n_scales: int = N_SCALES
    target_ema_rate: float = TARGET_EMA_RATE
    sample_ema_rate: float = SAMPLE_EMA_RATE
    tag: Optional[str] = None
    teacher_tag: Optional[str] = None
    teacher_checkpoint: Optional[str] = TEACHER_CHECKPOINT

    @property
    def teacher_run_name(self) -> str:
        base = f"mnist_digit{self.digit}" if self.digit is not None else "mnist_all"
        name = f"{base}_{self.teacher}"
        return f"{name}_{self.teacher_tag}" if self.teacher_tag else name

    @property
    def teacher_checkpoint_path(self) -> str:
        if self.teacher_checkpoint is not None:
            return self.teacher_checkpoint
        suffix = "_ema" if self.teacher == "rf" and not self.teacher_tag else ""
        return os.path.join(self.checkpoint_dir, f"{self.teacher_run_name}{suffix}.pth")

    @property
    def run_name(self) -> str:
        base = f"mnist_digit{self.digit}" if self.digit is not None else "mnist_all"
        name = f"{base}_cm_{self.teacher}"
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
    parser.add_argument(
        "--teacher",
        choices=["ve", "rf"],
        default=TEACHER,
        help="Which pretrained teacher to distill from.",
    )
    parser.add_argument("--digit", type=int, default=DIGIT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--sigma-noise", type=float, default=SIGMA_NOISE)
    parser.add_argument("--sigma-data", type=float, default=SIGMA_DATA)
    parser.add_argument(
        "--teacher-checkpoint",
        default=TEACHER_CHECKPOINT,
        help="Path to the teacher weights (default: see TEACHER_CHECKPOINT).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <checkpoint_dir>/<run_name>_resume.pth if it exists.",
    )
    parser.add_argument(
        "--distance",
        choices=["huber", "l2"],
        default=DISTANCE,
        help="Distance d(x,y) for the distillation loss (see consistency/distances.py).",
    )
    parser.add_argument(
        "--schedule",
        choices=["fixed", "curriculum"],
        default=SCHEDULE,
        help="fixed N/mu (paper's CD recipe) or the CT-style N(k)/mu(k) curriculum.",
    )
    parser.add_argument(
        "--n-scales",
        type=int,
        default=N_SCALES,
        help="Number of teacher discretization points N for --schedule fixed.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional suffix for the run name, e.g. --tag noema saves "
        "<checkpoint_dir>/mnist_all_cm_rf_noema.pth.",
    )
    parser.add_argument(
        "--teacher-tag",
        default=None,
        help="Suffix of the teacher checkpoint, e.g. --teacher-tag ema loads "
        "checkpoints/mnist_all_ve_ema.pth.",
    )
    args = parser.parse_args()
    return TrainConfig(
        teacher=args.teacher,
        digit=args.digit,
        batch_size=args.batch_size,
        n_iterations=args.iterations,
        lr=args.lr,
        sigma=args.sigma,
        sigma_noise=args.sigma_noise,
        sigma_data=args.sigma_data,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        distance=args.distance,
        schedule=args.schedule,
        n_scales=args.n_scales,
        tag=args.tag,
        teacher_tag=args.teacher_tag,
        teacher_checkpoint=args.teacher_checkpoint,
    )


def build_dataloader(config: TrainConfig) -> DataLoader:
    """Loads MNIST, optionally filtered down to a single digit."""
    return get_dataloader(
        batch_size=config.batch_size,
        digit=config.digit,
        train=True,
        data_dir=config.data_dir,
    )


def build_teacher(config: TrainConfig, device: torch.device):
    """Loads the pretrained teacher and wraps it in a TeacherAdapter.

    Returns:
        Tuple of (teacher_adapter, process) -- `process` is returned
        separately because the online/EMA ConsistencyWrapper also need
        its `sigma_t` to compute c_skip/c_out.
    """
    if config.teacher == "ve":
        process = VarianceExplodingDiffusion(sigma=config.sigma)
        backbone = ScoreNet(marginal_prob_std=process.sigma_t).to(device)
        backbone.load_state_dict(
            torch.load(config.teacher_checkpoint_path, map_location=device)
        )
        adapter_cls = VETeacherAdapter
    else:
        process = RectifiedFlow(sigma_noise=config.sigma_noise, eps=config.eps)
        backbone = VelocityNet().to(device)
        backbone.load_state_dict(
            torch.load(config.teacher_checkpoint_path, map_location=device)
        )
        adapter_cls = RectifiedFlowTeacherAdapter

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    teacher = adapter_cls(process=process, model=backbone, eps=config.eps)
    return teacher, process


def build_consistency_models(config: TrainConfig, process, device: torch.device):
    """Builds the online consistency model and its two frozen EMA copies.

    All wrap a *raw* backbone (no marginal_prob_std scaling -- that's
    what `ConsistencyWrapper`'s own c_skip/c_out replace) of the same
    architecture family as the teacher's, but with fresh random
    weights: the consistency model is trained from scratch, it does
    not start from the teacher's weights.

    Returns:
        Tuple (cm, cm_target, cm_sample).
    """
    if config.teacher == "ve":
        online_backbone = ScoreNet(marginal_prob_std=None)
    else:
        online_backbone = VelocityNet()

    cm = ConsistencyWrapper(
        backbone=online_backbone,
        sigma_data=config.sigma_data,
        eps=config.eps,
        sigma_t=process.sigma_t,
    ).to(device)

    cm_target = copy.deepcopy(cm)
    cm_sample = copy.deepcopy(cm)
    for p in itertools.chain(cm_target.parameters(), cm_sample.parameters()):
        p.requires_grad_(False)

    return cm, cm_target, cm_sample


def update_ema(cm_ema: torch.nn.Module, cm: torch.nn.Module, mu: float) -> None:
    """In-place EMA update: theta- <- mu*theta- + (1-mu)*theta."""
    with torch.no_grad():
        for p_ema, p in zip(cm_ema.parameters(), cm.parameters()):
            p_ema.mul_(mu).add_(p, alpha=1.0 - mu)


def infinite_batches(
    loader: DataLoader, seed: int, start_iteration: int = 0
) -> Iterator:
    """Endless, reproducible stream of batches, resumable mid-epoch.

    Each epoch is shuffled with its own generator seeded with
    `seed + epoch`, so the batch seen at iteration k depends only on
    (seed, k) -- not on how many times the run has been stopped and
    resumed. On resume, the batches of the current epoch that were
    already consumed before `start_iteration` are skipped.
    """
    batches_per_epoch = len(loader)
    epoch, skip = divmod(start_iteration, batches_per_epoch)
    while True:
        loader.sampler.generator = torch.Generator().manual_seed(seed + epoch)
        yield from itertools.islice(iter(loader), skip, None)
        epoch += 1
        skip = 0


def get_rng_state() -> dict:
    """Snapshot of every global RNG, stored in the resume checkpoint."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    """Restores the global RNGs saved by `get_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train(config: TrainConfig) -> None:
    logger = get_logger(name=config.run_name)
    set_seed(config.seed)
    device = get_device()

    loader = build_dataloader(config)
    logger.info(
        "Distilling '%s' -> '%s' | digit=%s | images=%d | device=%s",
        config.teacher_checkpoint_path,
        config.run_name,
        config.digit,
        len(loader.dataset),
        device,
    )

    teacher, process = build_teacher(config, device)
    cm, cm_target, cm_sample = build_consistency_models(config, process, device)
    optimizer = optim.Adam(cm.parameters(), lr=config.lr)
    distance_fn = DISTANCE_FNS[config.distance]

    if config.schedule == "fixed":
        discretization_schedule = ConstantSchedule(config.n_scales)
        ema_schedule = ConstantSchedule(config.target_ema_rate)
    else:
        discretization_schedule = DiscretizationSchedule(K=config.n_iterations)
        ema_schedule = EMADecaySchedule(discretization_schedule)
    logger.info(
        "distance=%s | schedule=%s (N=%s) | sample EMA=%s",
        config.distance,
        config.schedule,
        config.n_scales if config.schedule == "fixed" else "curriculum",
        config.sample_ema_rate,
    )

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    start_iteration = 0
    if config.resume and os.path.exists(config.resume_checkpoint_path):
        # weights_only=False: the RNG state holds numpy/python objects,
        # not just tensors (it's our own file, so it's safe to unpickle).
        checkpoint = torch.load(
            config.resume_checkpoint_path, map_location=device, weights_only=False
        )
        cm.load_state_dict(checkpoint["cm"])
        cm_target.load_state_dict(checkpoint["cm_target"])
        cm_sample.load_state_dict(checkpoint["cm_sample"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = checkpoint["iteration"] + 1
        # Older resume checkpoints don't have it; resuming from them
        # still works, just not bit-for-bit reproducibly.
        if "rng_state" in checkpoint:
            set_rng_state(checkpoint["rng_state"])
        logger.info(
            "Resumed from %s at iteration %d",
            config.resume_checkpoint_path,
            start_iteration,
        )
    elif config.resume:
        logger.warning(
            "--resume given but %s not found, starting from scratch",
            config.resume_checkpoint_path,
        )

    # Infinite iterator over the dataloader: unlike ve_training.py this
    # loops over a fixed iteration count, not epochs, so an epoch
    # boundary just means reshuffling and continuing. It's seeded per
    # epoch so that resuming picks up the exact same batch order.
    data_iter = infinite_batches(loader, config.seed, start_iteration)

    cm.train()
    running_loss = 0.0
    window_size = 0
    for iteration in range(start_iteration, config.n_iterations):
        x, _ = next(data_iter)
        x = x.to(device)

        optimizer.zero_grad()
        loss = consistency_distillation_loss(
            cm,
            cm_target,
            teacher,
            x,
            discretization_schedule,
            iteration,
            distance_fn=distance_fn,
        )
        loss.backward()
        optimizer.step()

        mu = ema_schedule(iteration)
        update_ema(cm_target, cm, mu)
        # Warm-up so early checkpoints aren't dominated by the random
        # init (a plain 0.9999 EMA needs ~10k steps to forget it).
        update_ema(
            cm_sample,
            cm,
            min(config.sample_ema_rate, (1 + iteration) / (10 + iteration)),
        )

        running_loss += loss.item()
        window_size += 1

        if (iteration + 1) % config.log_every == 0:
            logger.info(
                "[%d/%d] N=%d mu=%.5f - avg loss (last %d): %.4f",
                iteration + 1,
                config.n_iterations,
                discretization_schedule(iteration),
                mu,
                window_size,
                running_loss / window_size,
            )
            running_loss = 0.0
            window_size = 0

        if iteration > 0 and iteration % config.checkpoint_every == 0:
            # Usable snapshot -- always the latest one for sampling,
            # even if the run gets interrupted before finishing.
            torch.save(cm_sample.state_dict(), config.checkpoint_path)
            # Full resume state -- lets the *run itself* be continued.
            torch.save(
                {
                    "cm": cm.state_dict(),
                    "cm_target": cm_target.state_dict(),
                    "cm_sample": cm_sample.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iteration": iteration,
                    "rng_state": get_rng_state(),
                },
                config.resume_checkpoint_path,
            )
            logger.info("Checkpoint saved at iteration %d", iteration)

    if window_size > 0:
        logger.info(
            "Final avg loss (last %d iters window): %.4f",
            window_size,
            running_loss / window_size,
        )

    torch.save(cm_sample.state_dict(), config.checkpoint_path)
    logger.info("Consistency model (sampling EMA) saved to %s", config.checkpoint_path)


if __name__ == "__main__":
    train(parse_args())
