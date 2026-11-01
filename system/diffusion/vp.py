#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diffusion/vp.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    This module provides an implementation of the Variance Preserving
    (VP) diffusion process. It includes methods to compute drift and
    diffusion coefficients, as well as a loss function for training.

License:
    Proprietary. All rights reserved.
"""

from typing import Callable, Optional

import torch

from .base import DiffusionProcess


class VariancePreservingDiffusion(DiffusionProcess):
    """Variance Preserving (VP) diffusion process.

    This class implements a VP diffusion process with customizable beta
    schedule, drift, and diffusion functions.
    """

    kind: str = "VP"

    def __init__(
        self,
        beta_schedule: Callable[[torch.Tensor], torch.Tensor],
        drift_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        diffusion_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> None:
        """Initializes the VariancePreservingDiffusion.

        Args:
            beta_schedule: Function mapping time tensor to beta tensor.
            drift_fn: Optional custom drift function.
            diffusion_fn: Optional custom diffusion function.
        """
        super().__init__(
            drift_coefficient=drift_fn if drift_fn is not None
            else self.drift_coefficient,
            diffusion_coefficient=diffusion_fn if diffusion_fn is not None
            else self.diffusion_coefficient,
        )
        self.beta_schedule = beta_schedule
        self._custom_drift_fn = drift_fn
        self._custom_diffusion_fn = diffusion_fn
        self.T = 1

    def drift_coefficient(
        self, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Computes the drift coefficient.

        Args:
            x_t: State tensor at time t.
            t: Time tensor.

        Returns:
            Drift coefficient tensor.
        """
        if self._custom_drift_fn is not None:
            return self._custom_drift_fn(x_t, t)
        beta_t = self.beta_schedule.beta(t)
        return -0.5 * beta_t.view(-1, 1, 1, 1) * x_t

    def diffusion_coefficient(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the diffusion coefficient.

        Args:
            t: Time tensor.

        Returns:
            Diffusion coefficient tensor.
        """
        if self._custom_diffusion_fn is not None:
            return self._custom_diffusion_fn(t)
        beta_t = self.beta_schedule.beta(t)
        return torch.sqrt(beta_t)

    def mu_t(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Computes the mean of the process at time t.

        Args:
            x_0: Initial state tensor.
            t: Time tensor.

        Returns:
            Mean tensor at time t.
        """
        beta_int = self._integrated_beta(t)
        return x_0 * torch.exp(-0.5 * beta_int).view(-1, 1, 1, 1)

    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the standard deviation at time t.

        Args:
            t: Time tensor.

        Returns:
            Standard deviation tensor at time t.
        """
        beta_int = self._integrated_beta(t)
        val = 1.0 - torch.exp(-beta_int)
        val = torch.clamp(val, min=1e-5)  # Avoids negative or zero values
        return torch.sqrt(val)

    def _integrated_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the integrated beta for the schedule.

        Args:
            t: Time tensor.

        Returns:
            Integrated beta tensor.
        """
        return self.beta_schedule.integrated_beta(t)

    def loss_function(
        self,
        score_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_0: torch.Tensor,
        eps: float = 1e-5
    ) -> torch.Tensor:
        """Computes the loss for the VP diffusion process.

        Args:
            score_model: Model that estimates the score function.
            x_0: Initial state tensor.
            eps: Small value to avoid numerical issues.

        Returns:
            Scalar loss tensor.
        """
        t = torch.rand(x_0.shape[0], device=x_0.device) * (1.0 - eps) + eps
        noise = torch.randn_like(x_0)
        sigma = self.sigma_t(t).view(
            x_0.shape[0], *([1] * (x_0.dim() - 1))
        )
        x_t = self.mu_t(x_0, t) + sigma * noise
        score = score_model(x_t, t)
        # (sigma * score + noise) ** 2 already equals sigma(t)**2 times the
        # standard denoising score-matching error, i.e. it is already
        # weighted by lambda(t) = sigma(t)**2 (Song & Ermon, 2019). Do not
        # multiply by an extra lambda_t here or the weighting becomes
        # sigma(t)**4.
        loss_per_sample = torch.sum(
            (sigma * score + noise) ** 2,
            dim=list(range(1, x_0.dim()))
        )
        return torch.mean(loss_per_sample)
