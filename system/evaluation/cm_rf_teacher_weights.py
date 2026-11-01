#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/cm_rf_teacher_weights.py

Authors:
    Eloísa Denia García <Eloodenia>

Description:
    Compares CM-RF distilled from the EMA weights of the 1-RF teacher with
    CM-RF distilled from its raw weights. It lives outside
    `evaluation/compare.py` because adding a model there changes the
    multistep rows of all the others (see the note in `compare.MODELS`).

    Both models are evaluated as in `evaluation/seed_variance.py`, on
    N_EVAL images per noise seed. The global RNG is reset before each
    configuration, so both models also get the same re-noising and the
    numbers do not depend on the order of evaluation.

    Outputs, written to RESULTS_DIR:
    - cm_rf_pesos_teacher.csv / cm_rf_pesos_teacher.md: per configuration,
      mean and standard deviation over seeds of FD-MNIST, % recognizable
      and label entropy.

    Run from the repository root:
        python system/evaluation/cm_rf_teacher_weights.py

License:
    Proprietary. All rights reserved.
"""

import csv
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

from evaluation.classifier import MNISTClassifier
from evaluation.compare import (
    CLASSIFIER_PATH, N_EVAL, N_REFERENCE, RESULTS_DIR, SEED, ModelSpec,
    batched, build_configs, evaluate_samples, load_real_images, load_state,
)
from evaluation.metrics import extract_features
from evaluation.seed_variance import SEEDS
from utils import get_device, set_seed

RENOISE_SEED_OFFSET = 1000
SPECS = [
    ModelSpec("CM-RF (EMA 1-RF)", "cm_rf", "./checkpoints/mnist_all_cm_rf.pth", [1, 2, 4]),
    ModelSpec("CM-RF (raw 1-RF)", "cm_rf", "./checkpoints/mnist_all_cm_rf_noema.pth", [1, 2, 4]),
]


def main() -> None:
    set_seed(SEED)  # same real-image reference subset as compare.py
    device = get_device()
    classifier = load_state(MNISTClassifier(), CLASSIFIER_PATH, device)
    reference = extract_features(classifier, load_real_images(train=True, n=N_REFERENCE))

    rows = []
    for spec in SPECS:
        for c in build_configs(spec, device):
            metrics = []
            for seed in SEEDS:
                noise = c.noise(N_EVAL, torch.Generator().manual_seed(seed)).to(device)
                # Not `seed`: the re-noising would repeat the initial noise.
                torch.manual_seed(RENOISE_SEED_OFFSET + seed)
                metrics.append(evaluate_samples(batched(c.generate, noise), classifier, reference)[0])
            fd = [m["fd_mnist"] for m in metrics]
            recognizable = [100 * m["recognizable"] for m in metrics]
            entropy = [m["entropy_pct"] for m in metrics]
            rows.append({
                "model": c.model, "steps": c.steps, "nfe": c.nfe,
                "fd_mean": np.mean(fd), "fd_std": np.std(fd, ddof=1),
                "recognizable_mean": np.mean(recognizable), "recognizable_std": np.std(recognizable, ddof=1),
                "entropy_pct_mean": np.mean(entropy), "entropy_pct_std": np.std(entropy, ddof=1),
            })
            print(rows[-1], flush=True)

    with open(os.path.join(RESULTS_DIR, "cm_rf_pesos_teacher.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"Mean ± std over {len(SEEDS)} noise seeds ({N_EVAL} samples each).",
        "",
        "| Model | Steps | NFE | FD-MNIST | Recognizable (%) | Entropy (%) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['steps']} | {r['nfe']} | {r['fd_mean']:.1f} ± {r['fd_std']:.1f} | "
            f"{r['recognizable_mean']:.1f} ± {r['recognizable_std']:.1f} | "
            f"{r['entropy_pct_mean']:.1f} ± {r['entropy_pct_std']:.1f} |"
        )
    with open(os.path.join(RESULTS_DIR, "cm_rf_pesos_teacher.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
