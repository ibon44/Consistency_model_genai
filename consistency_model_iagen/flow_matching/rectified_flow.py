#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flow_matching/rectified_flow.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Implements Rectified Flow (Liu et al., 2022), a flow-matching
    process using linear interpolation paths between data and noise:
    x_t = (1-t)*x_0 + t*x_1, with constant target velocity x_1 - x_0.
    Used as a second candidate teacher for consistency distillation,
    alongside the VE diffusion process in `diffusion/ve.py`.

    TODO: implement `interpolate`, `target_velocity`, `sigma_t`, and
    `loss_function` (simple velocity-regression MSE, see `diffusion/
    ve.py::loss_function` for the analogous score-matching loss).

License:
    Proprietary. All rights reserved.
"""

from __future__ import annotations

from typing import Callable

import torch

from .base import FlowProcess


class RectifiedFlow(FlowProcess):
    """Rectified Flow process with linear data-to-noise interpolation."""

    kind: str = "RF"

    def __init__(self, sigma_data: float = 1.0, eps: float = 1.0e-5) -> None:
        """Initializes the RectifiedFlow process.

        Args:
            sigma_data: Standard deviation of the noise endpoint x_1.
            eps: Lower time bound used near t=0 for numerical stability.
        """
        self.sigma_data = sigma_data
        self.eps = eps
        self.T = 1.0

    def interpolate(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def target_velocity(
        self, x_0: torch.Tensor, x_1: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss_function(
        self,
        velocity_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_0: torch.Tensor,
        eps: float = 1.0e-5,
    ) -> torch.Tensor:
        raise NotImplementedError
