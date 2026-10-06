#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diffusion/schedules.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Beta schedules for the Variance Preserving (VP) and sub-Variance
    Preserving (sub-VP) diffusion processes. VariancePreservingDiffusion
    and SubVariancePreservingDiffusion both expect a `beta_schedule`
    object exposing `beta(t)` and `integrated_beta(t)`; LinearBetaSchedule
    is the standard choice used in Song et al. (2021), "Score-Based
    Generative Modeling through Stochastic Differential Equations".

License:
    Proprietary. All rights reserved.
"""

import torch


class LinearBetaSchedule:
    """Linear beta(t) schedule: beta(t) = beta_min + t * (beta_max - beta_min).

    This is the continuous-time analogue of the linear variance schedule
    used in DDPM (Ho et al., 2020), and the default schedule for VP/sub-VP
    SDEs in Song et al. (2021).
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        """Initializes the linear beta schedule.

        Args:
            beta_min: Value of beta(t) at t=0.
            beta_max: Value of beta(t) at t=1.
        """
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluates beta(t).

        Args:
            t: Time tensor with values in [0, 1].

        Returns:
            beta(t) tensor, same shape as t.
        """
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def integrated_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluates the integral of beta from 0 to t.

        Args:
            t: Time tensor with values in [0, 1].

        Returns:
            Integral_0^t beta(s) ds, same shape as t.
        """
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
