#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/compare.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Orchestration script for the project's final comparison (objectives
    3-4): loads the four trained models --

      - diffusion teacher   (checkpoints/mnist_all_ve.pth)
      - Rectified Flow teacher (checkpoints/mnist_all_rf.pth)
      - CM distilled from VE   (checkpoints/mnist_all_cm_ve.pth)
      - CM distilled from RF   (checkpoints/mnist_all_cm_rf.pth)

    -- and produces, under ../results/:
      - generation-time-vs-N plots for each model (evaluation/timing.py)
      - classifier-based quality/diversity tables at 1/2/4 steps for
        the consistency models (evaluation/metrics.py)
      - a grid of sample images per model for visual inspection

    TODO: implement `main`, wiring together the samplers in `sampling/`
    and the metrics in `evaluation/metrics.py` + `evaluation/timing.py`.

License:
    Proprietary. All rights reserved.
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ============================== CONFIG ==============================
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "../results"
STEP_COUNTS = [1, 2, 4]
SAMPLE_SIZES = [1, 8, 32, 128]
# ======================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    raise NotImplementedError(
        "TODO: load the four checkpoints, run "
        "evaluation.timing.benchmark_generation_time over SAMPLE_SIZES "
        "for each, evaluation.metrics.* over STEP_COUNTS for the "
        "consistency models, and save tables/plots to args.results_dir."
    )


if __name__ == "__main__":
    main()
