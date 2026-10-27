#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consistency/ct_loss.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Implements the consistency training (CT) loss (Song et al., 2023,
    Sec. 5; refined in Song & Dhariwal, 2024), which trains a
    consistency model directly from data without a pre-trained teacher,
    by replacing the teacher's ODE step with an unbiased estimator of
    the score/velocity target at two adjacent noise levels of the same
    trajectory.

    This is the first planned extension in the project proposal.

    Both points of the pair share the SAME noise z (x_{t_{n+1}} = x_0 +
    sigma_{n+1} z, x_{t_n} = x_0 + sigma_n z), so they lie on the same
    trajectory of the VE probability-flow ODE in expectation; the loss
    is d(f_theta(x_{t_{n+1}}, t_{n+1}), f_{theta-}(x_{t_n}, t_n)) with
    f_{theta-} the EMA target network. The time grid is the Karras grid
    in sigma (see `schedules.karras_sigma_time_steps`), refined by the
    N(k) curriculum as training progresses.

    Only the VE process is supported (it needs `sigma_t`/`sigma_inverse`
    and a zero-drift forward process, x_t = x_0 + sigma_t z).

License:
    Proprietary. All rights reserved.
"""

import torch
import torch.nn as nn
from typing import Callable

from .schedules import karras_sigma_time_steps


def consistency_training_loss(
    consistency_model: nn.Module,
    consistency_model_ema: nn.Module,
    process: object,
    x_0: torch.Tensor,
    discretization_schedule: Callable[[int], int],
    iteration: int,
    distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Computes the consistency training (teacher-free) loss.

    Args:
        consistency_model: Online consistency model f_theta.
        consistency_model_ema: EMA target copy f_{theta-}.
        process: The VE diffusion process (only used for its `sigma_t`,
            `sigma_inverse` and `T`; no teacher network is involved).
        x_0: Mini-batch of training data.
        discretization_schedule: N(k) schedule for the number of noise
            levels in the time grid.
        iteration: Current training iteration k.
        distance_fn: Distance between two consistency-model outputs.
        eps: Lower time bound (must match the ConsistencyWrapper's).

    Returns:
        Scalar consistency training loss.
    """
    N_k = int(discretization_schedule(iteration))
    t_grid = karras_sigma_time_steps(
        N_k, eps, process.T, process.sigma_t, process.sigma_inverse
    ).to(x_0.device)

    n_idx = torch.randint(0, N_k - 1, (x_0.size(0),), device=x_0.device)
    t_n = t_grid[n_idx]
    t_n_plus_1 = t_grid[n_idx + 1]

    shape = (-1, *([1] * (x_0.dim() - 1)))
    sigma_n = process.sigma_t(t_n).view(*shape)
    sigma_n_plus_1 = process.sigma_t(t_n_plus_1).view(*shape)

    z = torch.randn_like(x_0)
    x_t_n_plus_1 = x_0 + sigma_n_plus_1 * z
    x_t_n = x_0 + sigma_n * z

    pred_t_n_plus_1 = consistency_model(x_t_n_plus_1, t_n_plus_1)
    with torch.no_grad():
        pred_t_n = consistency_model_ema(x_t_n, t_n)

    return distance_fn(pred_t_n_plus_1, pred_t_n)
