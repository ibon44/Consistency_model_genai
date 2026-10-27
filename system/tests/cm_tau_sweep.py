#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/cm_tau_sweep.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Sweeps the re-noising time of multistep consistency sampling
    (Algorithm 1 of Song et al., 2023) for a consistency model
    distilled from the Rectified Flow teacher. With n steps, the
    (n - 1) re-noising times are spaced uniformly in [tau_max, eps)
    (`tests/consistency_sample_and_plot_RF.py::build_intermediate_times`),
    so tau_max sets how much of the previous prediction is kept: a low
    tau_max only refines it (x_tau = (1 - tau) * x + tau * noise keeps
    most of x), a high one regenerates most of the digit. The paper
    picks these times by searching for the best FID; this script shows
    the effect of tau_max visually, all from the SAME initial noise.

    Output: figures/cm_tau_sweep_<ddmmHHMM>.png, one row per tau_max,
    one column per step count, each panel the same GRID_SIDE^2 noises;
    the 1-step result (no tau) is the top-left panel for reference.

License:
    Proprietary. All rights reserved.
"""

import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import torch

from flow_matching import RectifiedFlow
from sampling.consistency_sampler import ConsistencySampler
from tests.compare_cm_rf_ema_noema import GRID_SIDE, load_cm, to_grid
from tests.consistency_sample_and_plot_RF import build_intermediate_times
from utils import get_device


# ============================== CONFIG ==============================
CM_CHECKPOINT = "./checkpoints_rf/mnist_all_cm_rf.pth"
STEP_COUNTS = [2, 4]
TAU_VALUES = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
SIGMA_NOISE = 1.0
EPS = 1.0e-5
SEED = 42
FIGURES_DIR = "./figures"
# ======================================================================


def main() -> None:
    device = get_device()
    torch.manual_seed(SEED)

    process = RectifiedFlow(sigma_noise=SIGMA_NOISE, eps=EPS)
    sampler = ConsistencySampler(process=process, eps=EPS)
    cm = load_cm(CM_CHECKPOINT, process, device)
    shape = (GRID_SIDE * GRID_SIDE, 1, 28, 28)
    x_T = process.sample_prior(shape, device=device)

    def run(n_steps: int, tau) -> torch.Tensor:
        times = build_intermediate_times(n_steps, EPS, tau) if tau else None
        x = sampler.sample(
            shape, cm, device=device, intermediate_times=times, x_T=x_T
        )
        return to_grid(x)

    one_step_grid = run(1, None)

    n_cols = len(STEP_COUNTS) + 1
    _, axes = plt.subplots(
        len(TAU_VALUES), n_cols, figsize=(4 * n_cols, 4.3 * len(TAU_VALUES)),
        squeeze=False,
    )
    for i, tau in enumerate(TAU_VALUES):
        for j in range(n_cols):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                if i == 0:
                    ax.imshow(one_step_grid, cmap="gray", vmin=0, vmax=1)
                    ax.set_title("1-step (no tau)", fontsize=12)
                else:
                    ax.axis("off")
                ax.text(-0.08, 0.5, f"tau_max = {tau}", ha="right", va="center",
                        rotation=90, fontsize=13, transform=ax.transAxes)
                continue
            n_steps = STEP_COUNTS[j - 1]
            ax.imshow(run(n_steps, tau), cmap="gray", vmin=0, vmax=1)
            if i == 0:
                ax.set_title(f"{n_steps}-step", fontsize=12)
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%d%m%H%M")
    path = os.path.join(FIGURES_DIR, f"cm_tau_sweep_{stamp}.png")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    print(f"Figure saved to {path}")
    plt.show()


if __name__ == "__main__":
    main()
