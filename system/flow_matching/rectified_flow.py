#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flow_matching/rectified_flow.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Implements Rectified Flow (Liu et al., 2022), a flow-matching
    process using linear interpolation paths between data and noise:
    x_t = (1-t)*x_0 + t*x_1, with constant target velocity x_1 - x_0.
    Used as a second candidate teacher for consistency distillation,
    alongside the VE diffusion process in `diffusion/ve.py`.

License:
    Proprietary. All rights reserved.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .base import FlowProcess


class RectifiedFlow(FlowProcess):
    """Rectified Flow process with linear data-to-noise interpolation."""

    kind: str = "RF"

    def __init__(self, sigma_noise: float = 1.0, eps: float = 1.0e-5) -> None:
        """Initializes the RectifiedFlow process.

        Args:
            sigma_noise: Standard deviation of the noise endpoint x_1.
            eps: Lower time bound used near t=0 for numerical stability.
        """
        self.sigma_noise = sigma_noise
        self.eps = eps
        self.T = 1.0

    def interpolate(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Computes the state x_t on the straight path from x_0 to x_1.

            x_t = (1 - t) * x_0 + t * x_1

        so x_t = x_0 (data) at t = 0 and x_t = x_1 (noise) at t = T = 1.

        Args:
            x_0: Data samples, shape (B, C, H, W).
            x_1: Noise samples, same shape as x_0.
            t: Time tensor of shape (B,), in [0, T]. It is reshaped
                internally to (B, 1, ..., 1) to broadcast per sample.

        Returns:
            Interpolated state tensor x_t, same shape as x_0.
        """
        t = t.view(-1, *([1] * (x_0.ndim - 1)))
        x_t = (1 - t) * x_0 + t * x_1
        return x_t

    def target_velocity(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """Computes the regression target for the velocity field.

        Differentiating the linear path with respect to t gives

            dx_t/dt = x_1 - x_0,

        which is constant along each path, hence independent of t.

        Args:
            x_0: Data samples, shape (B, C, H, W).
            x_1: Noise samples, same shape as x_0.

        Returns:
            Target velocity tensor x_1 - x_0, same shape as x_0.
        """
        return x_1 - x_0

    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the marginal noise scale at time t.

        The noise component of x_t is t * x_1 with
        x_1 ~ N(0, sigma_noise^2 I), so

            sigma(t) = t * sigma_noise.

        Used by `sample_prior` (at t = T) and by
        `models/consistency_wrapper.py` to compute c_skip(t)/c_out(t).

        Args:
            t: Time tensor of any shape, in [0, T].

        Returns:
            Standard deviation tensor, same shape as t.
        """
        return t * self.sigma_noise

    def sigma_inverse(self, sigma_t: torch.Tensor) -> torch.Tensor:
        """Inverts `sigma_t`: returns the time t with sigma_t(t) = sigma_t.

        Used to build Karras time grids spaced in sigma (see
        `consistency/schedules.py::karras_sigma_time_steps` and
        `ConsistencySampler.karras_intermediate_times`).

        Args:
            sigma_t: Standard deviation tensor.

        Returns:
            Time tensor, same shape as sigma_t.
        """
        return sigma_t / self.sigma_noise

    def loss_function(
        self,
        velocity_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_0: torch.Tensor,
        eps: float = 1.0e-5,
        x_1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes the Rectified Flow velocity-regression loss.

        Draws t ~ U(eps, 1) and x_1 ~ N(0, sigma_noise^2 I) per sample
        (unless x_1 is given, see below), builds x_t = interpolate(x_0, x_1, t) and returns

            E[ || v_theta(x_t, t) - (x_1 - x_0) ||^2 ],

        averaged over the batch and over every pixel. Unlike the VE
        score-matching loss (`diffusion/ve.py::loss_function`), no
        per-time weighting is needed: the target has the same scale
        for every t.

        Args:
            velocity_model: Time-dependent velocity model, invoked as
                velocity_model(x_t, t).
            x_0: Mini-batch of training data, shape (B, C, H, W).
            eps: Lower bound for t, for numerical stability near t=0.
            x_1: Optional noise paired with each x_0. None (default)
                draws it independently of x_0 (1-Rectified Flow).
                Reflow (Liu et al., 2022, Sec. 2.3) passes instead the
                noise the previous flow mapped to x_0, so the new flow
                learns that coupling and its paths get straighter.

        Returns:
            Scalar loss tensor.
        """
        t = torch.rand(x_0.shape[0], device=x_0.device) * (1 - eps) + eps
        if x_1 is None:
            x_1 = self.sample_prior(x_0.shape, device=x_0.device)
        x_t = self.interpolate(x_0, x_1, t)

        v_pred = velocity_model(x_t, t)
        v_target = self.target_velocity(x_0, x_1)

        loss = torch.mean((v_pred - v_target) ** 2)
        return loss
