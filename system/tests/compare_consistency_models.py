#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/compare_consistency_models.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Compares several consistency models distilled from the same VE
    teacher (e.g. different distance functions or training recipes)
    against each other and against the teacher itself:

    - Image grid: one row per sample, all from the SAME initial noise.
      The first column is the teacher's probability-flow ODE solution
      (Heun, many steps) -- the exact target a perfectly distilled CM
      should reproduce in 1 step. Then, for each step count, one column
      per model side by side.
    - Printed table: per model, the MSE between its 1-step output and
      the teacher ODE solution (distillation error), over a larger
      batch than the one plotted.

    Edit MODELS below to choose what to compare (add one ModelSpec per
    checkpoint, e.g. runs trained with different --tag values).
    Models whose checkpoint does not exist yet are skipped. By default
    it compares the distilled CM (consistency_distillation_training.py)
    against the teacher-free one (consistency_training.py).

License:
    Proprietary. All rights reserved.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, NamedTuple

import matplotlib.pyplot as plt
import torch
from consistency.teachers import VETeacherAdapter
from diffusion.ve import VarianceExplodingDiffusion
from models.consistency_wrapper import ConsistencyWrapper
from models.score_unet import ScoreNet
from sampling.consistency_sampler import ConsistencySampler
from utils import get_device


class ModelSpec(NamedTuple):
    label: str
    path: str


# ============================== CONFIG ==============================
MODELS: List[ModelSpec] = [
    ModelSpec("CD (teacher)", "./checkpoints/mnist_all_cm_ve_ema.pth"),
    ModelSpec("CT (no teacher)", "./checkpoints/mnist_all_cm_ve_ct.pth"),
]
TEACHER_PATH = "./checkpoints/mnist_all_ve_ema.pth"
TEACHER_ODE_STEPS = 200  # Heun steps for the reference teacher solution
N_IMAGES = 12  # rows plotted
N_METRIC = 128  # samples used for the 1-step MSE table
STEP_COUNTS = [1, 2, 4]
SIGMA = 25.0
SIGMA_DATA = 0.5
EPS = 1.0e-5
SEED = 42
# ======================================================================


def load_consistency_model(
    spec: ModelSpec, process: VarianceExplodingDiffusion, device
) -> ConsistencyWrapper:
    cm = ConsistencyWrapper(
        backbone=ScoreNet(marginal_prob_std=None),
        sigma_data=SIGMA_DATA,
        eps=EPS,
        sigma_t=process.sigma_t,
    ).to(device)
    cm.load_state_dict(torch.load(spec.path, map_location=device))
    cm.eval()
    return cm


def load_teacher(process: VarianceExplodingDiffusion, device) -> VETeacherAdapter:
    score = ScoreNet(marginal_prob_std=process.sigma_t).to(device)
    score.load_state_dict(torch.load(TEACHER_PATH, map_location=device))
    score.eval()
    return VETeacherAdapter(process=process, model=score, eps=EPS, heun=True)


@torch.no_grad()
def teacher_ode_solution(
    teacher: VETeacherAdapter, x_T: torch.Tensor, n_steps: int
) -> torch.Tensor:
    """Solves the teacher PF-ODE from T to eps with Heun steps."""
    times = teacher.time_grid(n_steps + 1).flip(0).to(x_T.device)
    x = x_T
    for t, t_prev in zip(times[:-1], times[1:]):
        x = teacher.ode_step(x, t.expand(x.shape[0]), t_prev.expand(x.shape[0]))
    return x.clamp(-1.0, 1.0)


def plot_grid(columns: List[torch.Tensor], labels: List[str], n_images: int) -> None:
    n_columns = len(columns)
    _, axes = plt.subplots(
        n_images, n_columns, figsize=(n_columns * 1.4, n_images * 1.4)
    )
    for j, (images, label) in enumerate(zip(columns, labels)):
        for i in range(n_images):
            ax = axes[i][j]
            ax.imshow((images[i, 0].cpu() + 1) / 2, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        axes[0][j].set_title(label, fontsize=8)
    plt.tight_layout()
    plt.show()


def main() -> None:
    device = get_device()
    torch.manual_seed(SEED)

    process = VarianceExplodingDiffusion(sigma=SIGMA)
    sampler = ConsistencySampler(process=process, eps=EPS)
    teacher = load_teacher(process, device)
    models = {}
    for spec in MODELS:
        if os.path.exists(spec.path):
            models[spec.label] = load_consistency_model(spec, process, device)
        else:
            print(f"Skipping '{spec.label}': {spec.path} not found")

    # ---- 1-step distillation error vs. the teacher ODE ----
    x_T = process.sample_prior((N_METRIC, 1, 28, 28), device=device)
    reference = teacher_ode_solution(teacher, x_T, TEACHER_ODE_STEPS)
    print(f"1-step MSE vs teacher ODE (Heun {TEACHER_ODE_STEPS}), {N_METRIC} samples:")
    for label, cm in models.items():
        x = sampler.sample(x_T.shape, cm, device=device, x_T=x_T)
        print(f"  {label:<14} {((x - reference) ** 2).mean().item():.4f}")

    # ---- image grid ----
    x_T = x_T[:N_IMAGES]
    columns = [reference[:N_IMAGES]]
    labels = [f"Teacher ODE\nHeun {TEACHER_ODE_STEPS}"]
    for n_steps in STEP_COUNTS:
        intermediate_times = sampler.karras_intermediate_times(n_steps)
        for label, cm in models.items():
            columns.append(
                sampler.sample(
                    x_T.shape,
                    cm,
                    device=device,
                    intermediate_times=intermediate_times,
                    x_T=x_T,
                )
            )
            labels.append(f"{n_steps}-step\n{label}")

    plot_grid(columns, labels, N_IMAGES)


if __name__ == "__main__":
    main()
