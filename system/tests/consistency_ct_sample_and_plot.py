#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/consistency_ct_sample_and_plot.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Visualizes the teacher-free consistency model trained with
    `training/consistency_training.py` (checkpoints/mnist_all_cm_ve_ct.pth):
    samples at several step counts side by side, all from the same
    initial noise per row. Safe to run while training is still going --
    the checkpoint is overwritten every --checkpoint-every iterations
    and this script prints which iteration the loaded snapshot comes
    from (read from the matching _resume file, if present).

License:
    Proprietary. All rights reserved.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from consistency_sample_and_plot import plot_step_comparison
from diffusion.ve import VarianceExplodingDiffusion
from models.consistency_wrapper import ConsistencyWrapper
from models.score_unet import ScoreNet
from sampling.consistency_sampler import ConsistencySampler
from utils import get_device

# ============================== CONFIG ==============================
# Optional first CLI argument overrides the checkpoint, e.g.
#   python system/tests/consistency_ct_sample_and_plot.py checkpoints/mnist_all_cm_ve_ict.pth
CHECKPOINT_PATH = (
    sys.argv[1] if len(sys.argv) > 1 else "./checkpoints/mnist_all_cm_ve_ct.pth"
)
N_IMAGES = 15
STEP_COUNTS = [1, 2, 4, 8]
# Karras rho of the multistep re-noising grid (default 7). With 2 steps a
# lower rho re-noises at sigma~1.5 instead of ~0.5, which is clearly better
# for the CT models; with 4+ steps the difference is within noise.
RHO_BY_STEPS = {2: 3.0}
SIGMA = 25.0  # must match consistency_training.py
SIGMA_DATA = 0.5
EPS = 1.0e-5
SEED = 42
# ======================================================================


def describe_snapshot(checkpoint_path: str) -> None:
    """Prints the training iteration the checkpoint was saved at."""
    resume_path = checkpoint_path.replace(".pth", "_resume.pth")
    if not os.path.exists(resume_path):
        return
    state = torch.load(resume_path, map_location="cpu")
    done = state["iteration"] + 1  # 'iteration' is the last completed index
    status = (
        "finished"
        if done >= state["n_iterations"]
        else "UNFINISHED, curriculum incomplete"
    )
    print(f"Checkpoint from iteration {done}/{state['n_iterations']} ({status})")


def main() -> None:
    device = get_device()
    torch.manual_seed(SEED)

    describe_snapshot(CHECKPOINT_PATH)

    process = VarianceExplodingDiffusion(sigma=SIGMA)
    cm = ConsistencyWrapper(
        backbone=ScoreNet(marginal_prob_std=None),
        sigma_data=SIGMA_DATA,
        eps=EPS,
        sigma_t=process.sigma_t,
    ).to(device)
    cm.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    cm.eval()

    sampler = ConsistencySampler(process=process, eps=EPS)
    shape = (N_IMAGES, 1, 28, 28)
    x_T = process.sample_prior(shape, device=device)

    columns = [
        sampler.sample(
            shape,
            cm,
            device=device,
            x_T=x_T,
            intermediate_times=sampler.karras_intermediate_times(
                n_steps, rho=RHO_BY_STEPS.get(n_steps, 7.0)
            ),
        )
        for n_steps in STEP_COUNTS
    ]

    plot_step_comparison(
        images=torch.stack(columns, dim=-1),
        step_counts=STEP_COUNTS,
        n_images=N_IMAGES,
        figsize=(len(STEP_COUNTS) * 3, N_IMAGES * 1.5),
    )


if __name__ == "__main__":
    main()
