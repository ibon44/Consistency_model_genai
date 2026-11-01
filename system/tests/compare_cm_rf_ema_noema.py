#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/compare_cm_rf_ema_noema.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Visual inspection of two consistency models distilled from the
    Rectified Flow teacher: one from its EMA weights
    (mnist_all_rf_ema.pth -> mnist_all_cm_rf.pth) and one from its raw
    weights (mnist_all_rf.pth -> mnist_all_cm_rf_noema.pth). Each row
    is a step count (plus rows with each one's teacher, Euler, at each
    of TEACHER_STEP_COUNTS steps); the left column is the EMA variant, the
    right one the non-EMA variant. Every panel is a grid of the SAME
    N_IMAGES initial noises, so the same cell position across panels
    started from the same x_T and differences come only from the
    teacher weights / number of steps. `compare` is reused by
    `tests/compare_cm_rf_reflow.py` for other variant sets.

License:
    Proprietary. All rights reserved.
"""

import os
import sys
from datetime import datetime
from typing import Dict, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import torch

from flow_matching import RectifiedFlow
from models.consistency_wrapper import ConsistencyWrapper
from models.velocity_unet import VelocityNet
from sampling.consistency_sampler import ConsistencySampler
from sampling.rectified_flow_sampler import RectifiedFlowSampler
from tests.consistency_sample_and_plot_RF import build_intermediate_times
from utils import get_device


# ============================== CONFIG ==============================
# label -> (consistency model checkpoint, teacher it was distilled from)
VARIANTS = {
    "EMA": (
        "./checkpoints/mnist_all_cm_rf.pth",
        "./checkpoints/mnist_all_rf_ema.pth",
    ),
    "no EMA": (
        "./checkpoints/mnist_all_cm_rf_noema.pth",
        "./checkpoints/mnist_all_rf.pth",
    ),
}
STEP_COUNTS = [1, 2, 4, 8]
TEACHER_STEP_COUNTS = [100]  # Euler steps for each teacher row
GRID_SIDE = 8  # each panel is GRID_SIDE x GRID_SIDE samples
N_IMAGES = GRID_SIDE * GRID_SIDE
SIGMA_NOISE = 1.0
SIGMA_DATA = 0.5
EPS = 1.0e-5
TAU_MAX = 0.5  # see consistency_sample_and_plot_RF.py
SEED = 42
FIGURES_DIR = "./figures"
# ======================================================================


def to_grid(x: torch.Tensor) -> torch.Tensor:
    """(N, 1, 28, 28) in [-1, 1] -> (GRID_SIDE*28, GRID_SIDE*28) in [0, 1]."""
    x = ((x.clamp(-1, 1) + 1) / 2)[:, 0].cpu()
    x = x.view(GRID_SIDE, GRID_SIDE, 28, 28).permute(0, 2, 1, 3)
    return x.reshape(GRID_SIDE * 28, GRID_SIDE * 28)


def load_cm(path: str, process: RectifiedFlow, device) -> ConsistencyWrapper:
    cm = ConsistencyWrapper(
        backbone=VelocityNet(),
        sigma_data=SIGMA_DATA,
        eps=EPS,
        sigma_t=process.sigma_t,
    ).to(device)
    cm.load_state_dict(torch.load(path, map_location=device))
    return cm.eval()


def load_teacher(path: str, device) -> VelocityNet:
    teacher = VelocityNet().to(device)
    teacher.load_state_dict(torch.load(path, map_location=device))
    return teacher.eval()


def compare(variants: Dict[str, Tuple[str, str]], figure_prefix: str) -> None:
    """Samples every variant from the same noise and saves the panel figure.

    Args:
        variants: label -> (consistency model checkpoint, teacher
            checkpoint), one figure column per entry.
        figure_prefix: Saved as <FIGURES_DIR>/<figure_prefix>_<ddmmHHMM>.png.
    """
    device = get_device()
    torch.manual_seed(SEED)

    process = RectifiedFlow(sigma_noise=SIGMA_NOISE, eps=EPS)
    sampler = ConsistencySampler(process=process, eps=EPS)
    shape = (N_IMAGES, 1, 28, 28)
    x_T = process.sample_prior(shape, device=device)

    row_titles = [f"{n}-step" for n in STEP_COUNTS] + [
        f"teacher ({n})" for n in TEACHER_STEP_COUNTS
    ]
    # panels[row][col] -> image grid
    panels = [[None] * len(variants) for _ in row_titles]

    for j, (cm_path, teacher_path) in enumerate(variants.values()):
        cm = load_cm(cm_path, process, device)
        for i, n_steps in enumerate(STEP_COUNTS):
            x = sampler.sample(
                shape,
                cm,
                device=device,
                intermediate_times=build_intermediate_times(n_steps, EPS, TAU_MAX),
                x_T=x_T,
            )
            panels[i][j] = to_grid(x)

        teacher = load_teacher(teacher_path, device)
        for i, n_steps in enumerate(TEACHER_STEP_COUNTS, start=len(STEP_COUNTS)):
            teacher_sampler = RectifiedFlowSampler(process, n_steps=n_steps)
            _, x_teacher = teacher_sampler.sample(
                shape, teacher, device=device, x_T=x_T
            )
            panels[i][j] = to_grid(x_teacher * 2 - 1)

    _, axes = plt.subplots(
        len(row_titles), len(variants),
        figsize=(5 * len(variants), 5 * len(row_titles)),
    )
    for i, row_title in enumerate(row_titles):
        for j, label in enumerate(variants):
            ax = axes[i][j]
            ax.imshow(panels[i][j], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(label, fontsize=16)
            if j == 0:
                ax.set_ylabel(row_title, fontsize=14)
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%d%m%H%M")
    path = os.path.join(FIGURES_DIR, f"{figure_prefix}_{stamp}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Figure saved to {path}")
    plt.show()


if __name__ == "__main__":
    compare(VARIANTS, "cm_rf_ema_vs_noema")
