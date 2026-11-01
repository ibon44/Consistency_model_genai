#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/compare.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Orchestration script for the project's final comparison (objectives
    3-4 of the proposal): how much faster the consistency models are
    than their teachers, and how much quality and diversity they lose,
    as a function of the number of sampling steps.

    For every model in MODELS and every step count listed for it, the
    script generates N_EVAL samples from the SAME initial noise (one
    shared noise batch per process family, VE or RF), and measures:

    - FD-MNIST: Fréchet distance to real MNIST in the classifier's
      feature space (main quality metric, lower is better).
    - % recognizable: fraction of samples the classifier labels with
      top-1 probability >= CONFIDENCE_THRESHOLD.
    - Label entropy and per-digit distribution (diversity).
    - Generation time for several batch sizes N (TIMING_SIZES), and
      time per image at the largest N.

    Two reference rows are added: real MNIST test images (the best
    values reachable with N_EVAL samples) and pure noise (what a
    useless generator scores, needed because the classifier is
    over-confident on noise, see `evaluation/metrics.py`).

    Outputs, written to RESULTS_DIR:
    - tabla_resultados.csv / tabla_resultados.md: one row per (model,
      steps) configuration.
    - resultados.json: everything above plus raw timings.
    - calidad_vs_nfe.png: FD-MNIST and % recognizable vs network
      evaluations (NFE), one line per model.
    - tiempo_vs_n.png: generation time vs N (log-log).
    - distribucion_digitos.png: heatmap of the predicted-digit
      distribution per configuration.
    - muestras_ve.png / muestras_rf.png: image grids, one column per
      configuration, every row starting from the same noise.
    - muestras_ve_3filas.png / muestras_rf_3filas.png: compact version
      of the previous grids (first 3 rows, columns chosen in GRID_3ROWS),
      the ones used in the report.

    Model kinds and how each one is sampled:
    - "ve_teacher": VE diffusion teacher (ScoreNet), probability-flow
      ODE solved with Heun steps on the teacher's Karras sigma grid
      (`consistency/teachers.py`). Each Heun step costs 2 network
      evaluations, so NFE = 2 * steps.
    - "rf_teacher": Rectified Flow teacher (VelocityNet), Euler steps
      (`sampling/rectified_flow_sampler.py`). NFE = steps.
    - "cm_ve": consistency model distilled (CD) or trained without a
      teacher (CT, iCT) on the VE process, Algorithm 1 with the sampler's
      default Karras re-noising times (VE_RHO_BY_STEPS can override rho
      per step count). NFE = steps.
    - "cm_rf": consistency model on the RF process, Algorithm 1 with
      re-noising times linearly spaced from RF_TAU_MAX down to EPS (as
      in `tests/consistency_sample_and_plot_RF.py`: re-noising RF close
      to T would resample the digit instead of refining it). NFE = steps.

    Configuration (CONFIG block below; MODELS is the list to edit when
    new checkpoints arrive, missing checkpoints are skipped):
    - N_EVAL: samples per configuration for the metrics (FD depends on
      it, so it must be the same for every model).
    - GEN_BATCH: samples generated at a time (memory bound).
    - N_REFERENCE: real training images used as the FD reference.
    - TIMING_SIZES / TIMING_REPEATS: batch sizes N and repeats for the
      timing benchmark.
    - GRID_IMAGES: rows in the image grids.
    - SIGMA, SIGMA_NOISE, SIGMA_DATA, EPS, RF_TAU_MAX: must match the
      values used to train the teachers and consistency models.

    Run from the repository root:
        python system/evaluation/compare.py            # full run
        python system/evaluation/compare.py --quick    # fast smoke test
        python system/evaluation/reuse_timing.py       # after a full run
        python system/evaluation/seed_variance.py      # sampling variability

    reuse_timing.py replaces the timing columns (and tiempo_vs_n.png) with
    the reference times measured once on an idle CPU, since the times only
    depend on the network and the number of evaluations. A full run takes
    about 40 minutes on CPU.

License:
    Proprietary. All rights reserved.
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, NamedTuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from consistency.teachers import VETeacherAdapter
from data import get_dataloader
from diffusion.ve import VarianceExplodingDiffusion
from evaluation.classifier import MNISTClassifier
from evaluation.metrics import (
    confident_fraction, extract_features, fd_mnist, label_distribution,
    label_entropy, max_label_entropy,
)
from evaluation.timing import benchmark_generation_time
from flow_matching.rectified_flow import RectifiedFlow
from models.consistency_wrapper import ConsistencyWrapper
from models.score_unet import ScoreNet
from models.velocity_unet import VelocityNet
from sampling.consistency_sampler import ConsistencySampler
from sampling.rectified_flow_sampler import RectifiedFlowSampler
from utils import get_device, get_logger, set_seed


class ModelSpec(NamedTuple):
    """One model to evaluate: display name, kind, checkpoint and step counts."""
    name: str
    kind: str
    path: str
    steps: List[int]


# ============================== CONFIG ==============================
#
# Every consistency model is listed next to the teacher it was distilled
# from: CM-VE (CD) from the raw VE teacher (the EMA one is the better VE
# teacher and is reported too), CM-RF (CD) from the EMA 1-RF and CM-RF
# (reflow) from the EMA 2-RF. CT and iCT use no teacher.
MODELS: List[ModelSpec] = [
    ModelSpec("VE teacher (EMA)", "ve_teacher", "./checkpoints/mnist_all_ve_ema.pth", [5, 10, 50]),
    ModelSpec("VE teacher (raw)", "ve_teacher", "./checkpoints/mnist_all_ve.pth", [5, 10, 50]),
    ModelSpec("CM-VE (CD)", "cm_ve", "./checkpoints/mnist_all_cm_ve.pth", [1, 2, 4]),
    ModelSpec("CM-VE (CT v1)", "cm_ve", "./checkpoints/mnist_all_cm_ve_ct.pth", [1, 2, 4]),
    ModelSpec("CM-VE (iCT v1, stage 1)", "cm_ve", "./checkpoints/mnist_all_cm_ve_ict.pth", [1, 2, 4]),
    ModelSpec("CM-VE (iCT v1)", "cm_ve", "./checkpoints/mnist_all_cm_ve_ict_n640.pth", [1, 2, 4]),
    ModelSpec("RF teacher (1-RF)", "rf_teacher", "./checkpoints_rf/mnist_all_rf_ema.pth", [1, 10, 50]),
    ModelSpec("RF teacher (2-RF)", "rf_teacher", "./checkpoints_rf/mnist_all_2rf_ema.pth", [1, 10, 50]),
    ModelSpec("CM-RF (CD)", "cm_rf", "./checkpoints_rf/mnist_all_cm_rf.pth", [1, 2, 4]),
    ModelSpec("CM-RF (CD, reflow)", "cm_rf", "./checkpoints_rf/mnist_all_cm_rf_reflow.pth", [1, 2, 4]),
    # Building a network and multistep sampling both draw from the global RNG, so
    # adding or removing any model here changes the multistep rows of all the others
    # (by less than the noise of Table 5). One-step and teacher rows do not change.
    ModelSpec("CM-VE (CT v2)", "cm_ve", "./checkpoints/mnist_all_cm_ve_ct_v2.pth", [1, 2, 4]),
    ModelSpec("CM-VE (iCT v2)", "cm_ve", "./checkpoints/mnist_all_cm_ve_ict_40k_ema_lrconst.pth", [1, 2, 4]),
]
# Columns (model, steps) of the compact 3-row grids muestras_<family>_3filas.png
# used in the report; a column whose checkpoint is missing is skipped.
GRID_3ROWS: Dict[str, List[tuple]] = {
    "VE": [("VE teacher (EMA)", 50), ("VE teacher (raw)", 50),
           ("CM-VE (CD)", 1), ("CM-VE (CD)", 2), ("CM-VE (CD)", 4),
           ("CM-VE (CT v1)", 4), ("CM-VE (CT v2)", 1), ("CM-VE (CT v2)", 4),
           ("CM-VE (iCT v1)", 4), ("CM-VE (iCT v2)", 1), ("CM-VE (iCT v2)", 4)],
    "RF": [("RF teacher (1-RF)", 1), ("RF teacher (1-RF)", 50),
           ("RF teacher (2-RF)", 1), ("RF teacher (2-RF)", 50),
           ("CM-RF (CD)", 1), ("CM-RF (CD)", 2), ("CM-RF (CD)", 4),
           ("CM-RF (CD, reflow)", 1), ("CM-RF (CD, reflow)", 2), ("CM-RF (CD, reflow)", 4)],
}
CLASSIFIER_PATH = "./checkpoints/mnist_classifier.pth"
# results/ keeps the outputs of memoria_final.tex; this folder is memoria_final_v2.tex.
RESULTS_DIR = "./results/memoria_final_v2"
DATA_DIR = "./data"
N_EVAL = 2000
GEN_BATCH = 500
N_REFERENCE = 10000
CONFIDENCE_THRESHOLD = 0.9
TIMING_SIZES = [1, 10, 100, 500]
TIMING_REPEATS = 3
GRID_IMAGES = 10
SIGMA = 25.0
SIGMA_NOISE = 1.0
SIGMA_DATA = 0.5
EPS = 1.0e-5
RF_TAU_MAX = 0.5
# Karras rho of the VE consistency models' multistep grid per step count
# (default 7, the one the reported tables were generated with). {2: 3.0} would
# re-noise 2-step sampling at sigma~1.5 instead of ~0.5, clearly better for all
# VE models, but it changes their 2-step rows; with 4+ steps it makes no
# difference beyond noise.
VE_RHO_BY_STEPS: Dict[int, float] = {}
SEED = 42
# ======================================================================

IMAGE_SHAPE = (1, 28, 28)


@dataclass
class Config:
    """Generator for one (model, steps) pair, plus its measured results.

    `generate(x_T)` maps a batch of initial noise to samples in
    [-1, 1]; `family` ("VE" or "RF") decides which shared noise batch
    it receives, so all configurations of a family start from the same
    noise. `prior_std` is the process' terminal noise level σ(T): the
    initial noise is a standard Gaussian times prior_std, exactly as
    the processes' own `sample_prior` builds it. `is_teacher` marks
    the diffusion/RF teachers: they are drawn dashed in the figures and
    the slowest one of each family is the reference for the speed-up.
    """
    model: str
    steps: int
    nfe: int
    family: str
    prior_std: float
    generate: Callable[[torch.Tensor], torch.Tensor]
    is_teacher: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)
    distribution: List[float] = field(default_factory=list)
    timing: Dict[int, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Short label used in tables and figures, e.g. 'CM-VE (CD) · 2 steps'."""
        return f"{self.model} · {self.steps} step{'s' if self.steps != 1 else ''}"

    def noise(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Draws `n` initial noise images for this configuration's process."""
        return torch.randn(n, *IMAGE_SHAPE, generator=generator) * self.prior_std


def parse_args() -> argparse.Namespace:
    """Parses CLI overrides. --quick shrinks everything for a smoke test."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--n-eval", type=int, default=N_EVAL)
    parser.add_argument("--no-timing", action="store_true", help="Skip the timing benchmark.")
    parser.add_argument("--quick", action="store_true", help="Few samples and tiny timings, to check the script runs.")
    args = parser.parse_args()
    if args.quick:
        args.n_eval = 200
    return args


def batched(fn: Callable[[torch.Tensor], torch.Tensor], x_T: torch.Tensor) -> torch.Tensor:
    """Applies a generator to `x_T` in chunks of GEN_BATCH and concatenates.

    Keeps memory bounded when generating thousands of samples on CPU.
    The output is clamped to [-1, 1], the classifier's input range.
    """
    with torch.no_grad():
        return torch.cat([fn(chunk) for chunk in x_T.split(GEN_BATCH)]).clamp(-1.0, 1.0)


def load_state(module: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module:
    """Loads a state dict into `module`, moves it to `device` and sets eval mode."""
    module.load_state_dict(torch.load(path, map_location=device))
    return module.to(device).eval()


def build_configs(spec: ModelSpec, device: torch.device) -> List[Config]:
    """Builds one Config per step count of `spec`, according to its kind.

    See the module docstring for how each kind is sampled and how its
    NFE (network evaluations per sample) is counted.

    Args:
        spec: The model to load.
        device: Device on which to run the model.

    Returns:
        List of Config objects, one per entry of `spec.steps`.
    """
    ve = VarianceExplodingDiffusion(sigma=SIGMA)
    rf = RectifiedFlow(sigma_noise=SIGMA_NOISE, eps=EPS)
    ve_std = ve.sigma_t(torch.full((1,), ve.T)).item()
    rf_std = rf.sigma_t(torch.full((1,), rf.T)).item()
    configs = []

    if spec.kind == "ve_teacher":
        score = load_state(ScoreNet(marginal_prob_std=ve.sigma_t), spec.path, device)
        teacher = VETeacherAdapter(process=ve, model=score, eps=EPS, heun=True)

        def make(n_steps: int) -> Callable[[torch.Tensor], torch.Tensor]:
            times = teacher.time_grid(n_steps + 1).flip(0).to(device)

            def generate(x: torch.Tensor) -> torch.Tensor:
                for t, t_prev in zip(times[:-1], times[1:]):
                    x = teacher.ode_step(x, t.expand(x.shape[0]), t_prev.expand(x.shape[0]))
                return x
            return generate

        for k in spec.steps:
            configs.append(Config(spec.name, k, 2 * k, "VE", ve_std, make(k), is_teacher=True))

    elif spec.kind == "rf_teacher":
        velocity = load_state(VelocityNet(), spec.path, device)

        def make(n_steps: int) -> Callable[[torch.Tensor], torch.Tensor]:
            sampler = RectifiedFlowSampler(rf, n_steps=n_steps)
            return lambda x: sampler.sample(tuple(x.shape), velocity, device=device, x_T=x)[1] * 2 - 1

        for k in spec.steps:
            configs.append(Config(spec.name, k, k, "RF", rf_std, make(k), is_teacher=True))

    elif spec.kind in ("cm_ve", "cm_rf"):
        is_rf = spec.kind == "cm_rf"
        process = rf if is_rf else ve
        backbone = VelocityNet() if is_rf else ScoreNet(marginal_prob_std=None)
        cm = load_state(
            ConsistencyWrapper(backbone=backbone, sigma_data=SIGMA_DATA, eps=EPS, sigma_t=process.sigma_t),
            spec.path, device,
        )
        sampler = ConsistencySampler(process, eps=EPS)

        def make(n_steps: int) -> Callable[[torch.Tensor], torch.Tensor]:
            if n_steps <= 1:
                taus = None
            elif is_rf:
                taus = torch.linspace(RF_TAU_MAX, EPS, n_steps)[:-1].tolist()
            else:
                taus = sampler.karras_intermediate_times(
                    n_steps, rho=VE_RHO_BY_STEPS.get(n_steps, 7.0)
                )
            return lambda x: sampler.sample(tuple(x.shape), cm, device=device, intermediate_times=taus, x_T=x)

        for k in spec.steps:
            configs.append(Config(spec.name, k, k, "RF" if is_rf else "VE", rf_std if is_rf else ve_std, make(k)))

    else:
        raise ValueError(f"Unknown model kind '{spec.kind}' for {spec.name}")
    return configs


def load_real_images(train: bool, n: int) -> torch.Tensor:
    """Returns the first `n` MNIST images of a split, in [-1, 1].

    The training loader shuffles, so the seed set in `main` makes the
    reference subset reproducible.
    """
    images, total = [], 0
    for x, _ in get_dataloader(batch_size=1000, train=train, data_dir=DATA_DIR):
        images.append(x)
        total += x.shape[0]
        if total >= n:
            break
    return torch.cat(images)[:n]


def evaluate_samples(
    samples: torch.Tensor, classifier: torch.nn.Module, reference: np.ndarray
) -> tuple:
    """Computes all quality/diversity metrics for one set of samples.

    Returns:
        Tuple (metrics dict, per-digit distribution as a list of 10
        floats). The metrics dict has keys fd_mnist, recognizable
        (fraction), entropy (nats) and entropy_pct (% of ln 10).
    """
    dist = label_distribution(classifier, samples)
    entropy = label_entropy(dist)
    metrics = {
        "fd_mnist": fd_mnist(classifier, samples, reference),
        "recognizable": confident_fraction(classifier, samples, CONFIDENCE_THRESHOLD),
        "entropy": entropy,
        "entropy_pct": 100 * entropy / max_label_entropy(),
    }
    return metrics, dist.tolist()


def save_table(rows: List[dict], results_dir: str) -> None:
    """Writes the results table as CSV and as a Markdown table."""
    columns = ["model", "steps", "nfe", "fd_mnist", "recognizable_pct", "entropy", "entropy_pct", "ms_per_image", "speedup"]
    with open(os.path.join(results_dir, "tabla_resultados.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def cell(v, fmt):
        return "—" if v is None or v == "" else format(v, fmt)

    lines = [
        "| Model | Steps | NFE | FD-MNIST ↓ | Recognizable ↑ | Entropy ↑ | ms / image ↓ | Speed-up ↑ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {cell(r['steps'], 'd')} | {cell(r['nfe'], 'd')} | {cell(r['fd_mnist'], '.1f')} | "
            f"{cell(r['recognizable_pct'], '.1f')} % | {cell(r['entropy'], '.3f')} ({cell(r['entropy_pct'], '.0f')} %) | "
            f"{cell(r['ms_per_image'], '.2f')} | {'×' + format(r['speedup'], '.1f') if r.get('speedup') else '—'} |"
        )
    with open(os.path.join(results_dir, "tabla_resultados.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


STEP_MARKERS = ["o", "s", "^", "D", "v"]


def model_colors(configs: List[Config]) -> Dict[str, tuple]:
    """Assigns each model a fixed color, shared by every figure.

    Colors come from matplotlib's 'tab20' palette in the order the
    models appear in MODELS, so a model keeps its color across the
    quality and timing plots.
    """
    models = list(dict.fromkeys(c.model for c in configs))
    palette = plt.get_cmap("tab20")
    return {m: palette(i % 20) for i, m in enumerate(models)}


def plot_quality_vs_nfe(configs: List[Config], baselines: Dict[str, dict], path: str) -> None:
    """Plots FD-MNIST and % recognizable against NFE, one line per model.

    The x axis is logarithmic because NFE spans 1 to ~100. Teachers are
    dashed, consistency models solid. Horizontal lines mark the
    real-MNIST and pure-noise references.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = model_colors(configs)
    for model, color in colors.items():
        cs = [c for c in configs if c.model == model]
        nfe = [c.nfe for c in cs]
        style = "--" if cs[0].is_teacher else "-"
        axes[0].plot(nfe, [c.metrics["fd_mnist"] for c in cs], style, color=color, marker="o", label=model)
        axes[1].plot(nfe, [100 * c.metrics["recognizable"] for c in cs], style, color=color, marker="o", label=model)
    for name, ls in (("Real MNIST", ":"), ("Pure noise", "-.")):
        b = baselines[name]
        axes[0].axhline(b["fd_mnist"], color="gray", linestyle=ls, label=name)
        axes[1].axhline(100 * b["recognizable"], color="gray", linestyle=ls, label=name)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("FD-MNIST (log scale, lower is better)")
    axes[1].set_ylabel(f"% recognizable (confidence ≥ {CONFIDENCE_THRESHOLD})")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Network evaluations per image (NFE, log scale)")
        ax.grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_time_vs_n(configs: List[Config], path: str) -> None:
    """Plots generation time against N (log-log), one line per configuration.

    Color identifies the model (same colors as `plot_quality_vs_nfe`)
    and the marker identifies the step count within that model (first,
    second, third entry of its `steps` list). Teachers are dashed.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = model_colors(configs)
    step_index: Dict[str, int] = {}
    for c in configs:
        if not c.timing:
            continue
        idx = step_index.get(c.model, 0)
        step_index[c.model] = idx + 1
        ns = sorted(c.timing)
        style = "--" if c.is_teacher else "-"
        ax.plot(ns, [c.timing[n] for n in ns], style, color=colors[c.model], marker=STEP_MARKERS[idx % len(STEP_MARKERS)],
                label=f"{c.label} (NFE {c.nfe})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of generated images N")
    ax.set_ylabel("Time (s)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distribution(labels: List[str], distributions: List[List[float]], path: str) -> None:
    """Heatmap of the predicted-digit distribution (%) per configuration.

    Uses a single-hue green scale starting at 0 % (almost white) and
    ending in a very dark green at the largest value, so the eye reads
    "darker = more samples of that digit". Cell labels switch to white
    on the darker half of the scale to stay readable.
    """
    data = 100 * np.array(distributions)
    vmax = max(20.0, float(data.max()))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(labels) + 1.2))
    im = ax.imshow(data, cmap="Greens", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(10))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Digit predicted by the classifier")
    for i in range(data.shape[0]):
        for j in range(10):
            ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=7, color="white" if data[i, j] > 0.5 * vmax else "black")
    fig.colorbar(im, ax=ax, label="% of samples (ideal ≈ 10 %)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grid(columns: List[torch.Tensor], labels: List[str], path: str) -> None:
    """Image grid: one column per configuration, rows share the initial noise."""
    n_rows, n_cols = columns[0].shape[0], len(columns)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.25, n_rows * 1.25 + 0.6), squeeze=False)
    for j, (images, label) in enumerate(zip(columns, labels)):
        for i in range(n_rows):
            axes[i][j].imshow((images[i, 0].cpu() + 1) / 2, cmap="gray", vmin=0, vmax=1)
            axes[i][j].axis("off")
        axes[0][j].set_title(label.replace(" · ", "\n"), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_rows(baselines: Dict[str, dict], configs: List[Config], n_max: int) -> tuple:
    """Builds the table rows (reference rows first, then one per configuration).

    The speed-up of a configuration is relative to the slowest teacher of
    its family (the first one with the largest NFE), at batch size `n_max`.

    Returns:
        Tuple (list of row dicts, dict family -> reference teacher Config).
    """
    slowest_teacher: Dict[str, Config] = {}
    for c in configs:
        if c.is_teacher and c.timing:
            if c.family not in slowest_teacher or c.nfe > slowest_teacher[c.family].nfe:
                slowest_teacher[c.family] = c

    rows = []
    for name, b in baselines.items():
        rows.append({"model": name, "steps": None, "nfe": None, "fd_mnist": b["fd_mnist"], "recognizable_pct": 100 * b["recognizable"],
                     "entropy": b["entropy"], "entropy_pct": b["entropy_pct"], "ms_per_image": None, "speedup": None})
    for c in configs:
        ms = 1000 * c.timing[n_max] / n_max if c.timing else None
        ref = slowest_teacher.get(c.family)
        speedup = (ref.timing[n_max] / c.timing[n_max]) if (ref and c.timing) else None
        rows.append({"model": c.model, "steps": c.steps, "nfe": c.nfe, "fd_mnist": c.metrics["fd_mnist"],
                     "recognizable_pct": 100 * c.metrics["recognizable"], "entropy": c.metrics["entropy"],
                     "entropy_pct": c.metrics["entropy_pct"], "ms_per_image": ms, "speedup": speedup})
    return rows, slowest_teacher


def main() -> None:
    """Runs the full comparison and writes the table and figures.

    Steps: load the classifier and real-image references, build every
    configuration, generate N_EVAL samples per configuration from a
    shared noise batch per family, compute the metrics, benchmark the
    generation time, and save the table, JSON and figures.
    """
    args = parse_args()
    timing_sizes = [1, 10] if args.quick else TIMING_SIZES
    timing_repeats = 1 if args.quick else TIMING_REPEATS
    os.makedirs(args.results_dir, exist_ok=True)
    logger = get_logger(name="compare")
    set_seed(SEED)
    device = get_device()
    t_start = time.time()

    classifier = load_state(MNISTClassifier(), CLASSIFIER_PATH, device)
    reference = extract_features(classifier, load_real_images(train=True, n=N_REFERENCE))
    logger.info("Reference: %d real training images | device=%s | N_EVAL=%d", N_REFERENCE, device, args.n_eval)

    baselines = {}
    for name, samples in (
        ("Real MNIST", load_real_images(train=False, n=args.n_eval)),
        ("Pure noise", torch.randn(args.n_eval, *IMAGE_SHAPE).clamp(-1.0, 1.0)),
    ):
        metrics, dist = evaluate_samples(samples, classifier, reference)
        baselines[name] = {**metrics, "distribution": dist}
        logger.info("%-28s FD=%.1f | recognizable=%.1f%% | entropy=%.3f", name, metrics["fd_mnist"], 100 * metrics["recognizable"], metrics["entropy"])

    configs: List[Config] = []
    for spec in MODELS:
        if not os.path.exists(spec.path):
            logger.warning("Skipping %s: checkpoint %s not found", spec.name, spec.path)
            continue
        configs.extend(build_configs(spec, device))

    noise: Dict[str, torch.Tensor] = {}
    grid_columns: Dict[str, List[tuple]] = {"VE": [], "RF": []}
    grid_samples: Dict[tuple, torch.Tensor] = {}
    for c in configs:
        if c.family not in noise:
            generator = torch.Generator().manual_seed(SEED)
            noise[c.family] = c.noise(args.n_eval, generator).to(device)
        t0 = time.time()
        samples = batched(c.generate, noise[c.family])
        c.metrics, c.distribution = evaluate_samples(samples, classifier, reference)
        grid_columns[c.family].append((c.label, samples[:GRID_IMAGES].cpu()))
        grid_samples[(c.model, c.steps)] = samples[:GRID_IMAGES].cpu()
        logger.info(
            "%-36s NFE=%4d | FD=%.1f | recognizable=%.1f%% | entropy=%.3f | %.1f s",
            c.label, c.nfe, c.metrics["fd_mnist"], 100 * c.metrics["recognizable"], c.metrics["entropy"], time.time() - t0,
        )

    if not args.no_timing:
        for c in configs:
            c.timing = benchmark_generation_time(lambda n, c=c: c.generate(c.noise(n).to(device)), timing_sizes, timing_repeats)
            logger.info("Timing %-36s %s", c.label, {n: f"{t:.3f}s" for n, t in c.timing.items()})

    rows, slowest_teacher = build_rows(baselines, configs, max(timing_sizes))

    save_table(rows, args.results_dir)
    with open(os.path.join(args.results_dir, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump({
            "config": {"n_eval": args.n_eval, "n_reference": N_REFERENCE, "timing_sizes": timing_sizes,
                       "timing_repeats": timing_repeats, "confidence_threshold": CONFIDENCE_THRESHOLD, "seed": SEED,
                       "device": str(device), "speedup_reference": {k: v.label for k, v in slowest_teacher.items()}},
            "baselines": baselines,
            "configs": [{"model": c.model, "steps": c.steps, "nfe": c.nfe, "family": c.family, **c.metrics,
                         "distribution": c.distribution, "timings": {str(n): t for n, t in c.timing.items()}} for c in configs],
        }, f, indent=2, ensure_ascii=False)

    plot_quality_vs_nfe(configs, baselines, os.path.join(args.results_dir, "calidad_vs_nfe.png"))
    if not args.no_timing:
        plot_time_vs_n(configs, os.path.join(args.results_dir, "tiempo_vs_n.png"))
    plot_distribution(
        list(baselines) + [c.label for c in configs],
        [b["distribution"] for b in baselines.values()] + [c.distribution for c in configs],
        os.path.join(args.results_dir, "distribucion_digitos.png"),
    )
    for family, cols in grid_columns.items():
        if cols:
            plot_grid([x for _, x in cols], [l for l, _ in cols], os.path.join(args.results_dir, f"muestras_{family.lower()}.png"))
        picked = [(m, s) for m, s in GRID_3ROWS[family] if (m, s) in grid_samples]
        if picked:
            plot_grid(
                [grid_samples[k][:3] for k in picked],
                [f"{m} · {s} step{'s' if s != 1 else ''}" for m, s in picked],
                os.path.join(args.results_dir, f"muestras_{family.lower()}_3filas.png"),
            )

    logger.info("Done in %.1f min. Results in %s", (time.time() - t_start) / 60, args.results_dir)


if __name__ == "__main__":
    main()
