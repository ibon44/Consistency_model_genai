#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/consistency_sample_and_plot.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Loads a consistency-model checkpoint distilled from the VE teacher
    (see `training/consistency_distillation_training.py`) and visualizes
    samples at several step counts side by side (Algorithm 1, `sampling/
    consistency_sampler.py`) -- the core speed/quality comparison the
    project measures (objectives 3-4). All step counts are sampled from
    the SAME initial noise per row, so differences across columns come
    only from the number of steps. Edit the CONFIG block below to point
    at a checkpoint trained on the whole dataset or a single digit.

License:
    Proprietary. All rights reserved.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List

import matplotlib.pyplot as plt
import torch

from diffusion.ve import VarianceExplodingDiffusion
from models.consistency_wrapper import ConsistencyWrapper
from models.score_unet import ScoreNet
from sampling.consistency_sampler import ConsistencySampler
from utils import get_device


# ============================== CONFIG ==============================
CHECKPOINT_PATH = "./checkpoints/mnist_all_cm_ve.pth"
N_IMAGES = 15
STEP_COUNTS = [1, 2, 4, 8, 16]  # number of consistency steps to compare
SIGMA = 25.0        # must match the VE teacher/consistency training sigma
SIGMA_DATA = 0.5    # must match consistency_distillation_training.py
EPS = 1.0e-5
SEED = 42
# ======================================================================


def plot_step_comparison(
    images: torch.Tensor, step_counts: List[int], n_images: int, figsize
) -> None:
    """Plots one row per sample, one column per step count.

    Args:
        images: Tensor (N, C, H, W, n_columns), values in [-1, 1].
        step_counts: Label for each column (e.g. [1, 2, 4]).
        n_images: Number of rows to plot.
        figsize: Matplotlib figure size.
    """
    n_columns = len(step_counts)
    _, axes = plt.subplots(n_images, n_columns, figsize=figsize)

    for i in range(n_images):
        for j, n_steps in enumerate(step_counts):
            img = images[i, 0, :, :, j].cpu().numpy()
            ax = axes[i][j] if n_images > 1 else axes[j]
            ax.imshow((img + 1) / 2, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_title(f"{n_steps}-step")

    plt.tight_layout()
    plt.show()


def main() -> None:
    device = get_device()
    torch.manual_seed(SEED)

    process = VarianceExplodingDiffusion(sigma=SIGMA)
    backbone = ScoreNet(marginal_prob_std=None)
    cm = ConsistencyWrapper(
        backbone=backbone, sigma_data=SIGMA_DATA, eps=EPS, sigma_t=process.sigma_t,
    ).to(device)
    cm.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    cm.eval()

    sampler = ConsistencySampler(process=process, eps=EPS)
    shape = (N_IMAGES, 1, 28, 28)

    # Same initial noise for every column, so 1/2/4-step differences
    # come only from the number of steps, not from a different draw.
    x_T = process.sample_prior(shape, device=device)

    columns = []
    for n_steps in STEP_COUNTS:
        intermediate_times = sampler.karras_intermediate_times(n_steps)
        x = sampler.sample(
            shape, cm, device=device,
            intermediate_times=intermediate_times, x_T=x_T,
        )
        columns.append(x)

    # Stack into (N, C, H, W, n_columns), the layout plot_image_evolution
    # expects (it was written for a single trajectory's intermediate
    # steps, but works just as well for independent step-count columns).
    images = torch.stack(columns, dim=-1)

    plot_step_comparison(
        images=images,
        step_counts=STEP_COUNTS,
        n_images=N_IMAGES,
        figsize=(len(STEP_COUNTS) * 3, N_IMAGES * 1.5),
    )


if __name__ == "__main__":
    main()
