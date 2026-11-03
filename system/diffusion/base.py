#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diffusion/base.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    This module provides a base, abstract class for diffusion processes.
    It includes methods to set drift and diffusion coefficients.

License:
    Proprietary. All rights reserved.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import torch


class DiffusionProcess(ABC):
    """Represents a generic diffusion process.

    This class models a diffusion process with customizable drift and
    diffusion coefficients. Concrete subclasses are expected to define
    ``self.T`` (the terminal time of the forward process) and a
    ``sigma_t(t)`` method returning the marginal standard deviation at
    time ``t``, which are used by the generic ``sample_prior`` and
    ``backward_drift_coefficient`` implementations below.
    """

    def __init__(
        self,
        drift_coefficient: Callable[[float, float], float] = (
            lambda x_t, t: 0.0  # type: ignore
        ),
        diffusion_coefficient: Callable[[float], float] = (
            lambda t: 1.0
        ),
    ) -> None:
        """Initializes the DiffusionProcess.

        Args:
            drift_coefficient: Function of (x_t, t) returning the drift
                coefficient. Defaults to a function returning 0.0.
            diffusion_coefficient: Function of t returning the diffusion
                coefficient. Defaults to a function returning 1.0.
        """
        self._drift_fn = drift_coefficient
        self._diffusion_fn = diffusion_coefficient

    @abstractmethod
    def drift_coefficient(self, x_t: float, t: float) -> float:
        """Abstract method to compute the drift coefficient."""

    @abstractmethod
    def diffusion_coefficient(self, t: float) -> float:
        """Abstract method to compute the diffusion coefficient."""

    def sample_prior(
        self, shape: Tuple[int, ...], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Samples from the terminal prior x_T ~ N(0, sigma(T)^2 I).

        Args:
            shape: Shape of the tensor to sample, formatted as
                (n_images, channels, *spatial_dims).
            device: Device on which to allocate the sample.

        Returns:
            A Gaussian noise tensor scaled by the process' terminal
            standard deviation.
        """
        if len(shape) < 3:
            raise ValueError(
                "shape must have at least 3 dimensions: "
                "(n_images, channels, *spatial_dims)"
            )
        t_max = torch.full((1,), self.T, device=device)
        std = self.sigma_t(t_max).reshape(())
        return std * torch.randn(*shape, device=device)

    def backward_drift_coefficient(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        score_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Computes the reverse-time SDE drift: f(x,t) - g(t)^2 * score(x,t).

        This is the general reverse-time drift shared by every diffusion
        process (Anderson, 1982); subclasses only need to provide correct
        ``drift_coefficient`` and ``diffusion_coefficient`` implementations.

        Args:
            x_t: Current state tensor.
            t: Time tensor.
            score_model: Score model callable, invoked as score_model(x_t, t).

        Returns:
            The reverse-time drift tensor.
        """
        score = score_model(x_t, t)
        diffusion = self.diffusion_coefficient(t)
        diffusion = diffusion.view(-1, *([1] * (x_t.dim() - 1)))
        drift = self.drift_coefficient(x_t, t)
        return drift - diffusion ** 2 * score
