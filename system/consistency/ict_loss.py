#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consistency/ict_loss.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Improved consistency training (iCT) loss, following Song & Dhariwal,
    2024 ("Improved Techniques for Training Consistency Models"). It
    differs from the original CT loss in `consistency/ct_loss.py` in
    four ways, all of which that paper found to matter:

    - No EMA target network (mu = 0): the target f_{theta-}(x_{t_n}, t_n)
      is the online model itself under `no_grad`. The EMA target of the
      original CT makes the target lag far behind the online model, and
      with no teacher there is nothing to correct that lag.
    - Exponential curriculum for the number of noise levels
      (`ImprovedDiscretizationSchedule`): N doubles at fixed stages
      instead of growing quadratically.
    - Noise levels are sampled from a lognormal distribution over sigma
      (P_mean = -1.1, P_std = 2.0), concentrating training on the
      intermediate noise levels where the content of the image is
      decided, instead of uniformly over the grid.
    - The loss of each pair is weighted by 1 / (sigma_{n+1} - sigma_n)
      and uses the pseudo-Huber distance.

    As in `ct_loss.py`, both points of a pair share the same noise z and
    only the VE process is supported.

License:
    Proprietary. All rights reserved.
"""

import math

import torch
import torch.nn as nn

from .schedules import karras_sigma_time_steps


class ImprovedDiscretizationSchedule:
    """Exponential curriculum N(k) = min(s0 * 2^floor(k/K'), s1) + 1.

    K' = floor(K / (log2(floor(s1/s0)) + 1)), so the run is split into
    equal stages and N doubles at the start of each one, ending at s1 + 1.
    """

    def __init__(self, K: int, s0: int = 10, s1: int = 160) -> None:
        self.K = K
        self.s0 = s0
        self.s1 = s1
        n_stages = int(math.log2(s1 // s0)) + 1
        self.stage_length = max(1, K // n_stages)

    def __call__(self, k: int) -> int:
        stage = int(k) // self.stage_length
        return min(self.s0 * 2 ** stage, self.s1) + 1


def _pseudo_huber_per_sample(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pseudo-Huber distance per sample, shape (B,), with c = 0.00054*sqrt(D)."""
    dim = a[0].numel()
    c = 0.00054 * math.sqrt(dim)
    squared = ((a - b) ** 2).flatten(1).sum(dim=1)
    return torch.sqrt(squared + c ** 2) - c


def improved_consistency_training_loss(
    consistency_model: nn.Module,
    process: object,
    x_0: torch.Tensor,
    discretization_schedule,
    iteration: int,
    eps: float = 1.0e-5,
    p_mean: float = -1.1,
    p_std: float = 2.0,
) -> torch.Tensor:
    """Computes the iCT loss for one mini-batch.

    Args:
        consistency_model: Online consistency model; it also provides the
            (stop-gradient) target, there is no EMA copy.
        process: The VE diffusion process (`sigma_t`, `sigma_inverse`, `T`).
        x_0: Mini-batch of training data.
        discretization_schedule: Callable k -> N(k), the number of grid
            points (see `ImprovedDiscretizationSchedule`).
        iteration: Current training iteration k.
        eps: Lower time bound (must match the ConsistencyWrapper's).
        p_mean: Mean of log(sigma) in the lognormal noise-level sampling.
        p_std: Standard deviation of log(sigma) in that distribution.

    Returns:
        Scalar loss.
    """
    N_k = int(discretization_schedule(iteration))
    t_grid = karras_sigma_time_steps(
        N_k, eps, process.T, process.sigma_t, process.sigma_inverse
    ).to(x_0.device)
    sigma_grid = process.sigma_t(t_grid)

    # P(pair n) is the lognormal mass between sigma_n and sigma_{n+1}.
    cdf = torch.erf((sigma_grid.log() - p_mean) / (math.sqrt(2.0) * p_std))
    probs = (cdf[1:] - cdf[:-1]).clamp_min(1.0e-12)
    n_idx = torch.multinomial(probs, x_0.size(0), replacement=True)

    t_n = t_grid[n_idx]
    t_n_plus_1 = t_grid[n_idx + 1]
    sigma_n = sigma_grid[n_idx]
    sigma_n_plus_1 = sigma_grid[n_idx + 1]

    shape = (-1, *([1] * (x_0.dim() - 1)))
    z = torch.randn_like(x_0)
    x_t_n_plus_1 = x_0 + sigma_n_plus_1.view(*shape) * z
    x_t_n = x_0 + sigma_n.view(*shape) * z

    pred_t_n_plus_1 = consistency_model(x_t_n_plus_1, t_n_plus_1)
    with torch.no_grad():
        pred_t_n = consistency_model(x_t_n, t_n)

    weight = 1.0 / (sigma_n_plus_1 - sigma_n)
    return (weight * _pseudo_huber_per_sample(pred_t_n_plus_1, pred_t_n)).mean()
