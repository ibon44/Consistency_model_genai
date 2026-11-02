#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/reuse_timing.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Replaces the generation times of a `compare.py` run with reference
    times measured once on an idle CPU (results/tiempos_referencia.json).

    Generation time depends only on the network and on the number of
    network evaluations, not on which weights are loaded, so every model
    of the same kind (VE teacher, VE consistency model, RF teacher, RF
    consistency model) shares the timing of its step count. Measuring it
    again for each of the ten models of a long run only adds noise when
    the CPU is shared: two teachers with the same network and steps came
    out 35 % apart. This script rewrites, from the quality metrics already
    in resultados.json, the timing columns of the table, the speed-ups,
    the JSON and tiempo_vs_n.png, leaving the quality metrics untouched.

    Run from the repository root, after compare.py:
        python system/evaluation/reuse_timing.py

License:
    Proprietary. All rights reserved.
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.compare import (
    APPENDIX_ONLY, MODELS, RESULTS_DIR, Config, build_rows, plot_time_vs_n, save_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--reference", default=os.path.join(RESULTS_DIR, "tiempos_referencia.json"))
    args = parser.parse_args()

    json_path = os.path.join(args.results_dir, "resultados.json")
    with open(json_path, encoding="utf-8") as f:
        results = json.load(f)
    with open(args.reference, encoding="utf-8") as f:
        reference = json.load(f)["timings"]

    kind_of = {spec.name: spec.kind for spec in MODELS}
    configs = []
    for entry in results["configs"]:
        kind = kind_of[entry["model"]]
        entry["timings"] = reference[kind][str(entry["steps"])]
        config = Config(
            model=entry["model"], steps=entry["steps"], nfe=entry["nfe"], family=entry["family"],
            prior_std=0.0, generate=None, is_teacher=kind.endswith("teacher"),
            metrics={k: entry[k] for k in ("fd_mnist", "recognizable", "entropy", "entropy_pct")},
            distribution=entry["distribution"],
            timing={int(n): t for n, t in entry["timings"].items()},
        )
        configs.append(config)

    n_max = max(results["config"]["timing_sizes"])
    rows, slowest_teacher = build_rows(results["baselines"], configs, n_max)
    results["config"]["speedup_reference"] = {k: v.label for k, v in slowest_teacher.items()}
    results["config"]["timing_source"] = os.path.basename(args.reference)

    save_table(rows, args.results_dir)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    # Models of the same kind have identical timings, so the figure shows
    # only the first model of each kind (otherwise the lines overlap).
    first_of_kind = {}
    for spec in MODELS:
        if spec.name not in APPENDIX_ONLY:
            first_of_kind.setdefault(spec.kind, spec.name)
    shown = [c for c, e in zip(configs, results["configs"]) if c.model == first_of_kind[kind_of[e["model"]]]]
    plot_time_vs_n(shown, os.path.join(args.results_dir, "tiempo_vs_n.png"))
    print(f"Timings of {len(configs)} configurations replaced with {args.reference}")


if __name__ == "__main__":
    main()
