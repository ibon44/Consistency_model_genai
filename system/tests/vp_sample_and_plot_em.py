#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/vp_sample_and_plot_em.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Loads a VP-trained ScoreNet checkpoint and visualizes samples
    generated with the Euler-Maruyama sampler. Mirrors
    `ve_sample_and_plot_em.py`; only the diffusion process (and its
    beta schedule, which must match the one used in
    `training/vp_training.py`) changes. Edit the CONFIG block below to
    point at a checkpoint trained on the whole dataset or on a single
    digit.

License:
    Proprietary. All rights reserved.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from diffusion.schedules import LinearBetaSchedule
from diffusion.vp import VariancePreservingDiffusion
from models.score_unet import ScoreNet
from sampling.euler_maruyama import EulerMaruyama
from utils import get_device, plot_image_evolution


# ============================== CONFIG ==============================
CHECKPOINT_PATH = "./checkpoints/mnist_all_vp.pth"
BETA_MIN = 0.1  # must match training/vp_training.py
BETA_MAX = 20.0  # must match training/vp_training.py
N_IMAGES = 3
T = 1.0
T_END = 1.0e-3
N_STEPS = 1000
INTERMEDIATE_STEPS = [0, N_STEPS]
SEED = 42
# ======================================================================


def main() -> None:
    device = get_device()

    beta_schedule = LinearBetaSchedule(beta_min=BETA_MIN, beta_max=BETA_MAX)
    diffusion_process = VariancePreservingDiffusion(beta_schedule=beta_schedule)
    score_model = ScoreNet(marginal_prob_std=diffusion_process.sigma_t).to(device)
    score_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=device)
    )
    score_model.eval()

    sampler = EulerMaruyama(process=diffusion_process)
    x_0 = diffusion_process.sample_prior((N_IMAGES, 1, 28, 28), device=device)

    with torch.no_grad():
        _, samples = sampler.sample(
            x_0=x_0,
            t_0=T,
            t_end=T_END,
            score_model=score_model,
            n_steps=N_STEPS,
            seed=SEED,
        )

    plot_image_evolution(
        images=samples,
        n_images=N_IMAGES,
        n_intermediate_steps=INTERMEDIATE_STEPS,
        figsize=(15, 4),
    )


if __name__ == "__main__":
    main()
