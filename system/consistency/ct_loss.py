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

    This is the first planned extension in the project proposal (used
    only if the essential objectives are completed with time to spare).

    TODO: implement `consistency_training_loss`.

License:
    Proprietary. All rights reserved.
"""

import torch
import torch.nn as nn

from .schedules import DiscretizationSchedule


def consistency_training_loss(
    consistency_model: nn.Module,
    consistency_model_ema: nn.Module,
    process: object,
    x_0: torch.Tensor,
    discretization_schedule: DiscretizationSchedule,
    iteration: int,
    distance_fn: nn.Module,
) -> torch.Tensor:
    """Computes the consistency training (teacher-free) loss.

    Args:
        consistency_model: Online consistency model f_theta.
        consistency_model_ema: EMA target copy f_{theta-}.
        process: The underlying diffusion or flow-matching process
            (only used for its `sigma_t`/`T`/`eps`, no teacher network
            is involved).
        x_0: Mini-batch of training data.
        discretization_schedule: N(k) schedule for the number of noise
            levels sampled along the trajectory.
        iteration: Current training iteration k.
        distance_fn: Distance/metric between two consistency-model
            outputs.

    Returns:
        Scalar consistency training loss.
    """
    raise NotImplementedError
