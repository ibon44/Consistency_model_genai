#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sampling/euler_maruyama.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    This module implements the Euler-Maruyama sampling method
    for diffusion models.

License:
    Proprietary. All rights reserved.
"""


from typing import Tuple, Union

import torch
from torch import Tensor

from .base import Sampler


class EulerMaruyama(Sampler):
    """Euler-Maruyama integrator for SDEs."""

    def __init__(self, process: object) -> None:
        """Initializes the Euler-Maruyama sampler.

        Args:
            process: The diffusion process object.
        """
        super().__init__(process)

    def sample(
        self,
        x_0: Tensor,
        t_0: float,
        t_end: float,
        n_steps: int,
        score_model,
        seed: Union[int, None] = None
    ) -> Tuple[Tensor, Tensor]:
        """Euler-Maruyama integrator (approximate).

        Args:
            x_0: The initial images of dimensions
                (batch_size, n_channels, image_height, image_width).
            t_0: The initial time.
            t_end: Endpoint of the integration interval.
            n_steps: Number of integration steps.
            seed: Seed for the random number generator.

        Returns:
            Tuple containing:
            - times: The time points at which the solution is computed.
            - x_t: Trajectories that result from the integration of the SDE.
                The shape is (*x_0.shape, n_steps + 1).
        """
        device = x_0.device

        times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
        dt = times[1] - times[0]

        if seed is not None:
            torch.manual_seed(seed)

        x_t = torch.empty(
            (*x_0.shape, len(times)), dtype=x_0.dtype, device=device
        )
        x_t[..., 0] = x_0

        for n, t in enumerate(times[:-1]):
            t = torch.ones(x_0.shape[0], device=device) * t
            noise = torch.randn_like(x_0)
            x_t[..., n + 1] = (
                x_t[..., n]
                + self.process.backward_drift_coefficient(x_t[..., n], t, score_model) * dt
                + self.process.diffusion_coefficient(t).view(-1, 1, 1, 1)
                * torch.sqrt(torch.abs(dt))
                * noise
            )

        return times, x_t

