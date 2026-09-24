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

    TODO: implement `consistency_distillation_loss`.

License:
    Proprietary. All rights reserved.
"""

import torch
import torch.nn as nn

from .schedules import DiscretizationSchedule
from .teachers import TeacherAdapter


def consistency_distillation_loss(
    consistency_model: nn.Module,
    consistency_model_ema: nn.Module,
    teacher: TeacherAdapter,
    x_0: torch.Tensor,
    discretization_schedule: DiscretizationSchedule,
    iteration: int,
    distance_fn: nn.Module,
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
            outputs (e.g. squared L2, LPIPS).

    Returns:
        Scalar consistency distillation loss.
    """
    raise NotImplementedError
