#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/consistency_wrapper.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Implements the consistency-model output parameterization from Song
    et al., 2023 (Consistency Models), wrapping a raw backbone network
    (e.g. `models/score_unet.py::ScoreNet` constructed with
    `marginal_prob_std=None`) to enforce the boundary condition
    f_theta(x, eps) = x:

        f_theta(x, t) = c_skip(t) * x + c_out(t) * backbone(x, t)

    where c_skip/c_out are computed from the teacher process' sigma_t
    and T/eps (see `diffusion.base.DiffusionProcess.sigma_t` and
    `flow_matching.base.FlowProcess.sigma_t`), so this wrapper is
    agnostic to whether the teacher is the VE diffusion process or
    Rectified Flow.

    TODO: implement `c_skip`, `c_out`, and `forward`.

License:
    Proprietary. All rights reserved.
"""

from typing import Callable

import torch
import torch.nn as nn


class ConsistencyWrapper(nn.Module):
    """Wraps a backbone network with the consistency-model parameterization."""

    def __init__(
        self,
        backbone: nn.Module,
        sigma_data: float,
        eps: float,
        sigma_t: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """Initializes the ConsistencyWrapper.

        Args:
            backbone: Raw network invoked as backbone(x, t), unscaled.
            sigma_data: Standard deviation of the training data,
                used to balance c_skip/c_out at typical noise scales.
            eps: Lower time bound where the boundary condition
                f_theta(x, eps) = x must hold exactly.
            sigma_t: The teacher process' marginal noise-scale function,
                shared between `diffusion.base.DiffusionProcess` and
                `flow_matching.base.FlowProcess`.
        """
        super().__init__()
        self.backbone = backbone
        self.sigma_data = sigma_data
        self.eps = eps
        self.sigma_t = sigma_t

    def c_skip(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def c_out(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluates f_theta(x, t) = c_skip(t)*x + c_out(t)*backbone(x, t).

        Args:
            x: Input tensor of shape (B, 1, H, W).
            t: Time tensor of shape (B,).

        Returns:
            Consistency-model output tensor, same shape as x.
        """
        raise NotImplementedError
