#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/rf_sample_and_plot.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Loads a Rectified-Flow-trained VelocityNet checkpoint and visualizes
    samples generated with the RectifiedFlowSampler for several numbers
    of Euler steps. Every row reuses the same initial noise, so rows
    differ only in the number of integration steps. Edit the CONFIG
    block below to point at a checkpoint trained on the whole dataset
    or on a single digit (see training/rectified_flow_training.py).

License:
    Proprietary. All rights reserved.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import torch

from flow_matching.rectified_flow import RectifiedFlow
from models.velocity_unet import VelocityNet
from sampling.rectified_flow_sampler import RectifiedFlowSampler
from utils import get_device, set_seed


# ============================== CONFIG ==============================
CHECKPOINT_PATH = (
    "./checkpoints/mnist_all_rf_v0.pth"  # Path to a trained VelocityNet checkpoint
)
N_IMAGES = 10
STEP_COUNTS = [1, 2, 10, 50]
SEED = 42
FIGURES_DIR = "./figures"
# ======================================================================


def main() -> None:
    device = get_device()

    process = RectifiedFlow()
    velocity_model = VelocityNet().to(device)
    velocity_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    velocity_model.eval()

    fig, axes = plt.subplots(
        len(STEP_COUNTS),
        N_IMAGES,
        figsize=(1.2 * N_IMAGES, 1.3 * len(STEP_COUNTS)),
        squeeze=False,
    )

    for row, n_steps in enumerate(STEP_COUNTS):
        # Same seed per row -> same initial noise for every step count.
        set_seed(SEED)
        sampler = RectifiedFlowSampler(process=process, n_steps=n_steps)
        _, samples = sampler.sample(
            shape=(N_IMAGES, 1, 28, 28),
            velocity_model=velocity_model,
            device=device,
        )

        for col in range(N_IMAGES):
            ax = axes[row][col]
            ax.imshow(
                samples[col, 0].cpu().numpy(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"N = {n_steps}")

    run_name = os.path.splitext(os.path.basename(CHECKPOINT_PATH))[0]
    fig.suptitle(f"Rectified Flow samples ({run_name})")
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, f"{run_name}_samples.png")
    fig.savefig(out_path, dpi=150)
    print(f"Figure saved to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
