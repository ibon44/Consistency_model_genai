#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/compare_distance_functions.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Compares two consistency models distilled from the same VE teacher
    but with different distance functions d(x,y) in the distillation
    loss (`consistency/distillation.py`, `consistency/distances.py`):
    squared L2 (original Consistency Models paper) vs. pseudo-Huber
    ("Improved Techniques for Training Consistency Models", Song &
    Dhariwal, 2024). Both checkpoints are sampled from the SAME initial
    noise per row and at the same step counts (Algorithm 1), so any
    difference across the L2/Huber columns comes only from the training
    loss -- this is meant to check whether pseudo-Huber reduces the
    error accumulation seen at higher step counts with squared L2.

    Edit CHECKPOINTS below if your runs are saved under different paths
    (e.g. after training a fresh Huber model with `--distance huber`,
    the previous L2 run should be backed up separately so it isn't
    overwritten -- see checkpoints/mnist_all_cm_ve_l2.pth).

License:
    Proprietary. All rights reserved.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch

from consistency.schedules import karras_time_steps
from diffusion.ve import VarianceExplodingDiffusion
from models.consistency_wrapper import ConsistencyWrapper
from models.score_unet import ScoreNet
from sampling.consistency_sampler import ConsistencySampler
from utils import get_device


# ============================== CONFIG ==============================
CHECKPOINTS: Dict[str, str] = {
    "L2": "./checkpoints/mnist_all_cm_ve_l2.pth",
    "Huber": "./checkpoints/mnist_all_cm_ve.pth",
}
N_IMAGES = 15
STEP_COUNTS = [1, 2, 4, 8, 16]  # number of consistency steps to compare
SIGMA = 25.0        # must match the VE teacher/consistency training sigma
SIGMA_DATA = 0.5    # must match consistency_distillation_training.py
EPS = 1.0e-5
SEED = 42
# ======================================================================


def build_intermediate_times(
    n_steps: int, eps: float, T: float
) -> Optional[List[float]]:
    """Builds the decreasing intermediate time points for n_steps calls.

    Same construction as `consistency_sample_and_plot.py`: (n_steps - 1)
    re-noising points taken from the Karras grid used in training.
    """
    if n_steps <= 1:
        return None
    grid = karras_time_steps(n_steps, eps, T)
    return grid[:-1].flip(0).tolist()


def load_consistency_model(
    checkpoint_path: str, process: VarianceExplodingDiffusion, device: str
) -> ConsistencyWrapper:
    """Loads a trained ConsistencyWrapper from a checkpoint."""
    backbone = ScoreNet(marginal_prob_std=None)
    cm = ConsistencyWrapper(
        backbone=backbone, sigma_data=SIGMA_DATA, eps=EPS, sigma_t=process.sigma_t,
    ).to(device)
    cm.load_state_dict(torch.load(checkpoint_path, map_location=device))
    cm.eval()
    return cm


def plot_comparison(
    images: torch.Tensor,
    column_labels: List[str],
    n_images: int,
    n_models: int,
    figsize,
) -> None:
    """Plots one row per sample, one column per (step count, model) pair.

    Args:
        images: Tensor (N, C, H, W, n_columns), values in [-1, 1].
        column_labels: Title for each column, e.g. "1-step\nL2".
        n_images: Number of rows to plot.
        n_models: Number of models compared per step count, used to
            group columns visually (a vertical separator every
            n_models columns).
        figsize: Matplotlib figure size.
    """
    n_columns = len(column_labels)
    _, axes = plt.subplots(n_images, n_columns, figsize=figsize)

    for i in range(n_images):
        for j in range(n_columns):
            img = images[i, 0, :, :, j].cpu().numpy()
            ax = axes[i][j] if n_images > 1 else axes[j]
            ax.imshow((img + 1) / 2, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_title(column_labels[j], fontsize=9)
            if j % n_models == 0 and j > 0:
                # Thin separator between step-count groups.
                ax.axvline(x=-0.5, color="red", linewidth=1.0, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main() -> None:
    device = get_device()
    torch.manual_seed(SEED)

    process = VarianceExplodingDiffusion(sigma=SIGMA)
    sampler = ConsistencySampler(process=process, eps=EPS)

    model_names = list(CHECKPOINTS.keys())
    models = {
        name: load_consistency_model(path, process, device)
        for name, path in CHECKPOINTS.items()
    }

    shape = (N_IMAGES, 1, 28, 28)
    # Same initial noise for every column, so differences come only
    # from step count / distance function, not from a different draw.
    x_T = process.sample_prior(shape, device=device)

    columns = []
    column_labels = []
    for n_steps in STEP_COUNTS:
        intermediate_times = build_intermediate_times(n_steps, EPS, process.T)
        for name in model_names:
            x = sampler.sample(
                shape, models[name], device=device,
                intermediate_times=intermediate_times, x_T=x_T,
            )
            columns.append(x)
            column_labels.append(f"{n_steps}-step\n{name}")

    images = torch.stack(columns, dim=-1)

    plot_comparison(
        images=images,
        column_labels=column_labels,
        n_images=N_IMAGES,
        n_models=len(model_names),
        figsize=(len(column_labels) * 1.6, N_IMAGES * 1.5),
    )


if __name__ == "__main__":
    main()
