#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/consistency_sample_and_plot_RF.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Rectified Flow counterpart of `tests/consistency_sample_and_plot.py`:
    loads a consistency-model checkpoint distilled from the RF teacher
    (`training/consistency_distillation_training.py --teacher rf`) and
    visualizes samples at several step counts side by side (Algorithm
    1, `sampling/consistency_sampler.py`) -- the core speed/quality
    comparison the project measures (objectives 3-4). A last column
    shows the RF teacher itself (Euler, TEACHER_STEPS steps), so bad
    samples can be attributed either to the teacher or to the
    distillation. All columns
    are sampled from the SAME initial noise per row, so differences
    across columns come only from the number of steps. Edit the CONFIG
    block below to point at a checkpoint trained on the whole dataset
    or a single digit.

License:
    Proprietary. All rights reserved.
"""

import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Optional

import matplotlib.pyplot as plt
import torch

from flow_matching import RectifiedFlow
from models.consistency_wrapper import ConsistencyWrapper
from models.velocity_unet import VelocityNet
from sampling.consistency_sampler import ConsistencySampler
from sampling.rectified_flow_sampler import RectifiedFlowSampler
from utils import get_device


# ============================== CONFIG ==============================
CHECKPOINT_PATH = "./checkpoints/mnist_all_cm_rf.pth"
# Teacher the consistency model was distilled from (EMA weights, as in
# consistency_distillation_training.py), shown as a reference column.
TEACHER_CHECKPOINT_PATH = "./checkpoints/mnist_all_rf_ema.pth"
TEACHER_STEPS = 100  # Euler steps for the teacher reference column
N_IMAGES = 15
STEP_COUNTS = [1, 2, 4, 8, 16]  # number of consistency steps to compare
SIGMA_NOISE = 1.0  # must match the RF teacher/consistency training sigma_noise
SIGMA_DATA = 0.5  # must match consistency_distillation_training.py
EPS = 1.0e-5
# Largest re-noising time for multistep sampling. With RF's forward
# x_tau = (1 - tau) * x + tau * x_1, re-noising close to T keeps almost
# nothing of the 1-step prediction (tau=0.94 keeps 6%), so the digit
# itself gets resampled and columns stop being comparable. Capping it
# keeps the digit picked by the first f_theta call and lets extra steps
# only refine it.
TAU_MAX = 0.5
SEED = 42
# The figure is saved here as cm_rf_steps_<ddmmHHMM>.png (day, month,
# hour, minute), so successive runs don't overwrite each other.
FIGURES_DIR = "./figures"
# ======================================================================


def build_intermediate_times(
    n_steps: int, eps: float, tau_max: float
) -> Optional[List[float]]:
    """Builds the decreasing intermediate time points for n_steps calls.

    n_steps total network evaluations need (n_steps - 1) re-noising
    points in between, spaced uniformly in [tau_max, eps). Unlike the VE
    script, this does not reuse the training Karras grid
    (`consistency/schedules.py::karras_time_steps`): with RF's
    sigma(t) = t that grid piles up next to eps, where x_t is already
    almost clean and re-noising changes nothing. The upper end is
    capped at tau_max (see TAU_MAX) instead of starting next to T, so
    the first re-noising keeps enough of the 1-step prediction for
    every column to show the same digit.

    Args:
        n_steps: Total number of consistency-model evaluations wanted
            (1 = pure 1-step generation).
        eps: Lower time bound.
        tau_max: First (largest) re-noising time, in (eps, T).

    Returns:
        None for n_steps=1 (1-step sampling), otherwise a list of
        (n_steps - 1) decreasing time points in [tau_max, eps).
    """
    if n_steps <= 1:
        return None
    # Drop eps: it is where f_theta(x, eps) = x, so it is not a useful
    # re-noising point.
    return torch.linspace(tau_max, eps, n_steps)[:-1].tolist()


def plot_step_comparison(
    images: torch.Tensor, column_titles: List[str], n_images: int, figsize
) -> None:
    """Plots one row per sample, one column per step count.

    Args:
        images: Tensor (N, C, H, W, n_columns), values in [-1, 1].
        column_titles: Title for each column (e.g. ["1-step", ...]).
        n_images: Number of rows to plot.
        figsize: Matplotlib figure size.
    """
    n_columns = len(column_titles)
    _, axes = plt.subplots(n_images, n_columns, figsize=figsize)

    for i in range(n_images):
        for j, title in enumerate(column_titles):
            img = images[i, 0, :, :, j].cpu().numpy()
            ax = axes[i][j] if n_images > 1 else axes[j]
            ax.imshow((img + 1) / 2, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_title(title)

    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%d%m%H%M")
    path = os.path.join(FIGURES_DIR, f"cm_rf_steps_{stamp}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {path}")

    plt.show()


def main() -> None:
    device = get_device()
    torch.manual_seed(SEED)

    process = RectifiedFlow(sigma_noise=SIGMA_NOISE, eps=EPS)
    backbone = VelocityNet()
    cm = ConsistencyWrapper(
        backbone=backbone,
        sigma_data=SIGMA_DATA,
        eps=EPS,
        sigma_t=process.sigma_t,
    ).to(device)
    cm.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    cm.eval()

    sampler = ConsistencySampler(process=process, eps=EPS)
    shape = (N_IMAGES, 1, 28, 28)

    # Same initial noise for every column, so step-count differences
    # come only from the number of steps, not from a different draw.
    x_T = process.sample_prior(shape, device=device)

    columns = []
    for n_steps in STEP_COUNTS:
        intermediate_times = build_intermediate_times(n_steps, EPS, TAU_MAX)
        x = sampler.sample(
            shape,
            cm,
            device=device,
            intermediate_times=intermediate_times,
            x_T=x_T,
        )
        columns.append(x)

    # Teacher reference from the same x_T. RectifiedFlowSampler returns
    # [0, 1], so map it back to [-1, 1] like the consistency columns.
    teacher = VelocityNet().to(device)
    teacher.load_state_dict(
        torch.load(TEACHER_CHECKPOINT_PATH, map_location=device)
    )
    teacher.eval()
    teacher_sampler = RectifiedFlowSampler(process, n_steps=TEACHER_STEPS)
    _, x_teacher = teacher_sampler.sample(shape, teacher, device=device, x_T=x_T)
    columns.append(x_teacher * 2 - 1)

    # Stack into (N, C, H, W, n_columns), the layout plot_image_evolution
    # expects (it was written for a single trajectory's intermediate
    # steps, but works just as well for independent step-count columns).
    images = torch.stack(columns, dim=-1)

    plot_step_comparison(
        images=images,
        column_titles=[f"{n}-step" for n in STEP_COUNTS]
        + [f"teacher ({TEACHER_STEPS})"],
        n_images=N_IMAGES,
        figsize=((len(STEP_COUNTS) + 1) * 3, N_IMAGES * 1.5),
    )


if __name__ == "__main__":
    main()
