#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/vp_sample_and_plot_others.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Loads a VP-trained ScoreNet checkpoint (see training/vp_training.py)
    and visualizes samples generated with every sampler in `sampling/`
    that is usable with a VP score model, besides Euler-Maruyama
    (already covered by `vp_sample_and_plot_em.py`):

    - PredictorCorrector: alternates Euler-Maruyama predictor steps
      with Langevin-dynamics corrector steps (`sampling/
      predictor_corrector.py`). Works with any process exposing
      `backward_drift_coefficient`/`diffusion_coefficient`, VP
      included.
    - ExponentialIntegrator: integrates the VP reverse SDE's linear
      part exactly via the matrix exponential, only approximating the
      score-driven forcing term (`sampling/exponential_integrator.py`).
      Requires a `beta_schedule`, so unlike the other samplers here it
      is VP/sub-VP only -- it does not apply to VE.
    - ProbabilityFlow: solves the probability-flow ODE with an
      adaptive Runge-Kutta integrator (`sampling/probability_flow.py`).
      Deterministic given the initial noise, unlike the two SDE
      samplers above.

    ConsistencySampler and RectifiedFlowSampler are not included: the
    former drives a distilled consistency model rather than a raw
    score model, and the latter targets `flow_matching.rectified_flow.
    RectifiedFlow` (and is unimplemented besides).

    Edit the CONFIG block below to point at a checkpoint trained on
    the whole dataset or a single digit, matching the beta schedule
    used to train it.

License:
    Proprietary. All rights reserved.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from diffusion.schedules import LinearBetaSchedule
from diffusion.vp import VariancePreservingDiffusion
from models.score_unet import ScoreNet
from sampling.exponential_integrator import ExponentialIntegrator
from sampling.predictor_corrector import PredictorCorrector
from sampling.probability_flow import ProbabilityFlow
from utils import get_device


# ============================== CONFIG ==============================
CHECKPOINT_PATH = "./checkpoints/mnist_all_vp.pth"
BETA_MIN = 0.1  # must match training/vp_training.py
BETA_MAX = 20.0  # must match training/vp_training.py
N_IMAGES = 5
SEED = 42

PC_NUM_STEPS = 1000
PC_CORRECTOR_STEPS = 1
PC_SNR = 0.15

EI_NUM_STEPS = 1000
EI_NOISE_SCALE = 1.0

PF_EPS = 1.0e-3
PF_METHOD = "RK45"
# ======================================================================


def plot_sampler_comparison(
    images: List[Tensor], column_titles: List[str], n_images: int, figsize
) -> None:
    """Plots one row per sample, one column per sampler.

    Args:
        images: One (n_images, C, H, W) tensor per column, each already
            rescaled to [0, 1].
        column_titles: Label for each column.
        n_images: Number of rows to plot.
        figsize: Matplotlib figure size.
    """
    n_columns = len(images)
    _, axes = plt.subplots(n_images, n_columns, figsize=figsize)

    for i in range(n_images):
        for j in range(n_columns):
            img = images[j][i, 0, :, :].cpu().numpy()
            ax = axes[i][j] if n_images > 1 else axes[j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_title(column_titles[j])

    plt.tight_layout()
    plt.show()


def main() -> None:
    device = get_device()

    beta_schedule = LinearBetaSchedule(beta_min=BETA_MIN, beta_max=BETA_MAX)
    diffusion_process = VariancePreservingDiffusion(beta_schedule=beta_schedule)
    score_model = ScoreNet(marginal_prob_std=diffusion_process.sigma_t).to(device)
    score_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=device)
    )
    score_model.eval()

    shape = (N_IMAGES, 1, 28, 28)
    columns = []
    column_titles = []

    # Predictor-Corrector (SDE): returns samples in [-1, 1].
    torch.manual_seed(SEED)
    pc_sampler = PredictorCorrector(
        process=diffusion_process,
        model=score_model,
        num_steps=PC_NUM_STEPS,
        corrector_steps=PC_CORRECTOR_STEPS,
        snr=PC_SNR,
    )
    with torch.no_grad():
        _, pc_samples = pc_sampler.sample(shape)
    columns.append((pc_samples + 1) / 2)
    column_titles.append(f"Predictor-Corrector\n({PC_NUM_STEPS} steps)")

    # Exponential Integrator (SDE, VP/sub-VP only): returns samples in
    # [-1, 1].
    torch.manual_seed(SEED)
    ei_sampler = ExponentialIntegrator(
        process=diffusion_process,
        model=score_model,
        num_steps=EI_NUM_STEPS,
        noise_scale=EI_NOISE_SCALE,
    )
    with torch.no_grad():
        _, ei_samples = ei_sampler.sample(shape)
    columns.append((ei_samples + 1) / 2)
    column_titles.append(f"Exponential Integrator\n({EI_NUM_STEPS} steps)")

    # Probability Flow ODE (deterministic): already rescaled to [0, 1].
    torch.manual_seed(SEED)
    pf_sampler = ProbabilityFlow(process=diffusion_process, eps=PF_EPS, method=PF_METHOD)
    _, pf_samples = pf_sampler.sample(shape, score_model=score_model, device=device)
    columns.append(pf_samples)
    column_titles.append(f"Probability Flow\n({PF_METHOD})")

    plot_sampler_comparison(
        images=columns,
        column_titles=column_titles,
        n_images=N_IMAGES,
        figsize=(len(columns) * 3, N_IMAGES * 1.5),
    )


if __name__ == "__main__":
    main()
