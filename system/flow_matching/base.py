#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flow_matching/base.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Abstract base class for flow-matching processes. Unlike the SDE
    processes in `diffusion/base.py` (defined via drift/diffusion
    coefficients), a FlowProcess is defined via a deterministic
    interpolation path between data and noise, x_t = path(x_0, x_1, t),
    and a regression target for the velocity field d(x_t)/dt.

    Concrete subclasses (e.g. RectifiedFlow) must define `self.T`, an
    `eps` lower bound near t=0, and a `sigma_t(t)` method returning the
    marginal noise scale at time t. `sigma_t`/`T` are exposed so that
    `consistency/` can stay agnostic to whether its teacher is an SDE
    diffusion process or a flow-matching process.

License:
    Proprietary. All rights reserved.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import torch


class FlowProcess(ABC):
    """Represents a generic flow-matching process.

    Models a deterministic path between a data sample x_0 and a noise
    sample x_1, together with the velocity field a model is trained to
    regress onto. Concrete subclasses define `self.T` (terminal time)
    and `sigma_t(t)`, mirroring `diffusion.base.DiffusionProcess` so
    both process families can serve as a consistency-model teacher
    through a common interface.
    """

    T: float
    eps: float

    @abstractmethod
    def interpolate(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Computes the interpolated state x_t along the flow path.

        Args:
            x_0: Data samples.
            x_1: Noise samples, same shape as x_0.
            t: Time tensor in [0, T].

        Returns:
            Interpolated state tensor x_t.
        """

    @abstractmethod
    def target_velocity(
        self, x_0: torch.Tensor, x_1: torch.Tensor
    ) -> torch.Tensor:
        """Computes the regression target for the velocity field.

        Args:
            x_0: Data samples.
            x_1: Noise samples, same shape as x_0.

        Returns:
            Target velocity tensor dx_t/dt used by `loss_function`.
        """

    @abstractmethod
    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the marginal noise scale at time t.

        Shared interface with `diffusion.base.DiffusionProcess.sigma_t`
        so that `consistency/` can compute c_skip(t)/c_out(t) without
        knowing whether the teacher is an SDE or a flow process.

        Args:
            t: Time tensor.

        Returns:
            Marginal standard deviation tensor at time t.
        """

    def sample_prior(
        self, shape: Tuple[int, ...], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Samples from the terminal prior x_1 ~ N(0, sigma(T)^2 I).

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

    @abstractmethod
    def loss_function(
        self,
        velocity_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_0: torch.Tensor,
        eps: float = 1.0e-5,
    ) -> torch.Tensor:
        """Computes the flow-matching regression loss.

        Args:
            velocity_model: Time-dependent velocity model, invoked as
                velocity_model(x_t, t).
            x_0: Mini-batch of training data.
            eps: Tolerance value for numerical stability near t=0.

        Returns:
            Computed loss value tensor.
        """
