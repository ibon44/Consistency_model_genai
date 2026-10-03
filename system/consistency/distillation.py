#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consistency/distillation.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Implements the consistency distillation (CD) loss (Song et al.,
    2023, Algorithm 2): given a trained teacher (through a
    `consistency.teachers.TeacherAdapter`), take one teacher ODE step
    from x_{t_{n+1}} to x_{t_n}, then minimize the distance between the
    online consistency model's prediction at t_{n+1} and the EMA
    target model's prediction at t_n.

    Works identically for a VE-diffusion or a Rectified-Flow teacher,
    since it only relies on the TeacherAdapter interface.

License:
    Proprietary. All rights reserved.
"""

from typing import Callable

import torch
import torch.nn as nn

from .distances import squared_l2_distance
from .schedules import DiscretizationSchedule, karras_time_steps
from .teachers import TeacherAdapter


def consistency_distillation_loss(
    consistency_model: nn.Module,
    consistency_model_ema: nn.Module,
    teacher: TeacherAdapter,
    x_0: torch.Tensor,
    discretization_schedule: DiscretizationSchedule,
    iteration: int,
    distance_fn: Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ] = squared_l2_distance,
) -> torch.Tensor:
    """Computes the consistency distillation loss for one mini-batch.

    Args:
        consistency_model: Online consistency model f_theta.
        consistency_model_ema: EMA target copy f_{theta-}.
        teacher: Adapter exposing the teacher's ODE step and sigma_t.
        x_0: Mini-batch of training data.
        discretization_schedule: N(k) schedule for the number of
            discretization steps of the teacher trajectory.
        iteration: Current training iteration k, passed to the
            discretization schedule.
        distance_fn: Distance/metric between two consistency-model
            outputs. Defaults to `squared_l2_distance`
            (`consistency/distances.py`); pass `pseudo_huber_distance`
            for the more stable alternative from the "Improved
            Techniques" follow-up paper, or your own callable.

    Returns:
        Scalar consistency distillation loss.
    """
    # Number of discretization points N(k) for this iteration, and the
    # non-uniform Karras time grid t_1=eps < ... < t_N=T they live on
    # (denser near eps, see consistency/schedules.py::karras_time_steps).
    N_k = discretization_schedule(iteration)
    t_grid = karras_time_steps(N_k, teacher.eps, teacher.T).to(x_0.device)

    # Sample, per example in the batch, an adjacent pair of grid points
    # (t_n, t_{n+1}) -- n uniform in {0, ..., N_k-2} (0-indexed).
    n_idx = torch.randint(0, N_k - 1, (x_0.size(0),), device=x_0.device)
    t_n = t_grid[n_idx]
    t_n_plus_1 = t_grid[n_idx + 1]

    # Sample x_{t_{n+1}} from the teacher's own forward process.
    x_t_n_plus_1 = teacher.forward_sample(x_0, t_n_plus_1)

    # Take one ODE step from t_{n+1} to t_n using the teacher (no
    # gradient: the teacher is frozen, only f_theta is trained).
    with torch.no_grad():
        x_t_n = teacher.ode_step(x_t_n_plus_1, t_n_plus_1, t_n)

    # Get the consistency model's prediction at t_{n+1}
    pred_t_n_plus_1 = consistency_model(x_t_n_plus_1, t_n_plus_1)

    # Get the EMA target model's prediction at t_n
    with torch.no_grad():
        pred_t_n = consistency_model_ema(x_t_n, t_n)

    # Compute the distance between the two predictions
    loss = distance_fn(pred_t_n_plus_1, pred_t_n)

    return loss
