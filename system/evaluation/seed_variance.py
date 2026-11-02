#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/seed_variance.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Sampling variability of the comparison made by `evaluation/compare.py`
    (by default for the consistency models only; --kinds adds the teachers).
    Each model is trained only once, so run-to-run variation cannot be
    measured, but part of the noise in FD-MNIST and in the other metrics
    comes from the finite sample: the same trained model, evaluated on
    N_EVAL images generated from a different batch of initial noise,
    gives slightly different numbers. This script repeats the evaluation
    of every (model, steps) configuration of `compare.MODELS` with
    several noise seeds (all configurations of a family share the noise
    of each seed, as in compare.py) and reports the mean and the standard
    deviation over seeds, so differences between models smaller than that
    deviation should not be interpreted.

    Outputs, written to RESULTS_DIR:
    - variabilidad_ruido.csv / variabilidad_ruido.md: per configuration,
      mean and standard deviation over seeds of FD-MNIST, % recognizable
      and label entropy.

    Run from the repository root:
        python system/evaluation/seed_variance.py

License:
    Proprietary. All rights reserved.
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from utils import get_device, get_logger, set_seed

from evaluation.classifier import MNISTClassifier
from evaluation.compare import (
    CLASSIFIER_PATH,
    MODELS,
    N_EVAL,
    N_REFERENCE,
    RESULTS_DIR,
    batched,
    build_configs,
    evaluate_samples,
    load_real_images,
    load_state,
)
from evaluation.metrics import extract_features

# ============================== CONFIG ==============================
SEEDS = [1, 2, 3, 4, 5]   # noise seeds; compare.py itself uses 42
KINDS = ["cm_ve", "cm_rf"]   # model kinds evaluated: teachers are slow and not the point

# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--n-eval", type=int, default=N_EVAL)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--kinds", nargs="+", default=KINDS, help="Model kinds of compare.MODELS to evaluate.")
    args = parser.parse_args()

    logger = get_logger(name="seed_variance")
    set_seed(42)  # same real-image reference subset as compare.py
    device = get_device()
    t_start = time.time()
    os.makedirs(args.results_dir, exist_ok=True)

    classifier = load_state(MNISTClassifier(), CLASSIFIER_PATH, device)
    reference = extract_features(classifier, load_real_images(train=True, n=N_REFERENCE))

    configs = []
    for spec in (m for m in MODELS if m.kind in args.kinds):
        if os.path.exists(spec.path):
            configs.extend(build_configs(spec, device))
        else:
            logger.warning("Skipping %s: checkpoint %s not found", spec.name, spec.path)

    results: Dict[tuple, Dict[str, List[float]]] = {
        (c.model, c.steps): {"fd_mnist": [], "recognizable": [], "entropy_pct": []} for c in configs
    }
    for seed in args.seeds:
        noise: Dict[str, torch.Tensor] = {}
        for c in configs:
            if c.family not in noise:
                noise[c.family] = c.noise(args.n_eval, torch.Generator().manual_seed(seed)).to(device)
            samples = batched(c.generate, noise[c.family])
            metrics, _ = evaluate_samples(samples, classifier, reference)
            for key in results[(c.model, c.steps)]:
                results[(c.model, c.steps)][key].append(metrics[key])
        logger.info("Seed %d done (%.1f min elapsed)", seed, (time.time() - t_start) / 60)

    rows = []
    for c in configs:
        r = results[(c.model, c.steps)]
        rows.append({
            "model": c.model, "steps": c.steps, "nfe": c.nfe,
            "fd_mean": np.mean(r["fd_mnist"]), "fd_std": np.std(r["fd_mnist"], ddof=1),
            "recognizable_mean": 100 * np.mean(r["recognizable"]), "recognizable_std": 100 * np.std(r["recognizable"], ddof=1),
            "entropy_pct_mean": np.mean(r["entropy_pct"]), "entropy_pct_std": np.std(r["entropy_pct"], ddof=1),
        })

    columns = list(rows[0])
    with open(os.path.join(args.results_dir, "variabilidad_ruido.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"Mean ± std over {len(args.seeds)} noise seeds ({args.n_eval} samples each).",
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
    with open(os.path.join(args.results_dir, "variabilidad_ruido.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Done in %.1f min. Results in %s", (time.time() - t_start) / 60, args.results_dir)


if __name__ == "__main__":
    main()
