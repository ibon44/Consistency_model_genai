#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diffusion/ve.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    This module provides an implementation of the Variance Exploding
    (VE) diffusion process. It includes methods to compute drift and
    diffusion coefficients, as well as a loss function for training.

License:
    Proprietary. All rights reserved.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch

from .base import DiffusionProcess


class VarianceExplodingDiffusion(DiffusionProcess):
    """Variance Exploding Brownian Motion diffusion process.

    This class implements a variance exploding (VE) diffusion process,
    supporting custom drift, diffusion, mean, and standard deviation
    functions.
    """

    kind: str = "VE"

    def __init__(
        self,
        sigma: float = 25.0,
        drift_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        diffusion_fn: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
        mu_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        sigma_fn: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
    ) -> None:
        """Initializes the VarianceExplodingDiffusion process.

        Args:
            sigma: The base standard deviation parameter.
            drift_fn: Optional custom drift function.
            diffusion_fn: Optional custom diffusion function.
            mu_fn: Optional custom mean function.
            sigma_fn: Optional custom standard deviation function.
        """
        super().__init__(
            drift_coefficient=drift_fn if drift_fn is not None
            else self.drift_coefficient,
            diffusion_coefficient=diffusion_fn if diffusion_fn is not None
            else self.diffusion_coefficient,
        )
        self.sigma = sigma
        self._custom_drift_fn = drift_fn
        self._custom_diffusion_fn = diffusion_fn
        self._custom_mu_fn = mu_fn
        self._custom_sigma_fn = sigma_fn
        self.T = 1

    def mu_t(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Computes the mean of the process at time t.

        Args:
            x_0: Initial value tensor.
            t: Time tensor.

        Returns:
            Mean tensor at time t.
        """
        if self._custom_mu_fn is not None:
            return self._custom_mu_fn(x_0, t)
        return x_0

    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the standard deviation at time t.

        Args:
            t: Time tensor.

        Returns:
            Standard deviation tensor at time t.
        """
        if self._custom_sigma_fn is not None:
            return self._custom_sigma_fn(t)
        log_sigma = torch.log(
            torch.tensor(self.sigma, dtype=torch.float32, device=t.device)
        )
        return torch.sqrt(
            0.5 * (self.sigma ** (2 * t) - 1.0) / log_sigma
        )

    def sigma_inverse(self, sigma_t: torch.Tensor) -> torch.Tensor:
        """Inverts `sigma_t`: returns the time t with sigma_t(t) = sigma_t.

        Args:
            sigma_t: Standard deviation tensor.

        Returns:
            Time tensor, same shape as sigma_t.
        """
        if self._custom_sigma_fn is not None:
            raise NotImplementedError(
                "sigma_inverse is only defined for the default VE sigma_t."
            )
        log_sigma = math.log(self.sigma)
        return torch.log1p(2.0 * log_sigma * sigma_t ** 2) / (2.0 * log_sigma)

    def drift_coefficient(
        self, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Computes the drift coefficient at x_t and t.

        Args:
            x_t: Current value tensor.
            t: Time tensor.

        Returns:
            Drift coefficient tensor.
        """
        if self._custom_drift_fn is not None:
            return self._custom_drift_fn(x_t, t)
        return torch.zeros_like(x_t)

    def diffusion_coefficient(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the diffusion coefficient at time t.

        Args:
            t: Time tensor.

        Returns:
            Diffusion coefficient tensor.
        """
        if self._custom_diffusion_fn is not None:
            return self._custom_diffusion_fn(t)
        return self.sigma ** t

    def loss_function(
        self,
        score_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_0: torch.Tensor,
        eps: float = 1.0e-5
    ) -> torch.Tensor:
        """Computes the loss for training score-based generative models.

        Args:
            score_model: Time-dependent score-based model.
            x_0: Mini-batch of training data.
            eps: Tolerance value for numerical stability.

        Returns:
            Computed loss value tensor.
        """
        t = torch.rand(x_0.shape[0], device=x_0.device) * (1.0 - eps) + eps
        noise = torch.randn_like(x_0)
        sigma = self.sigma_t(t)
        sigma = sigma.view(x_0.shape[0], *([1] * (x_0.dim() - 1)))
        x_t = x_0 + sigma * noise
        score = score_model(x_t, t)
        mse_samples = torch.sum(
            (sigma * score + noise) ** 2,
            dim=list(range(1, x_0.dim()))
        )
        loss = torch.mean(mse_samples)
        return loss
