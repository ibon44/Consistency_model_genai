#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sampling/rectified_flow_sampler.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    n-step Euler sampler for the Rectified Flow ODE dx/dt = v_theta(x,t)
    (see `flow_matching/rectified_flow.py`), analogous to
    `sampling/probability_flow.py` for the VE diffusion process. Used
    both as the RF teacher's own sampler (for the diffusion-vs-RF
    baseline comparison) and, with a low step count, as a reference
    point when discussing the speedup obtained by the consistency
    models distilled from it.

License:
    Proprietary. All rights reserved.
"""

from typing import Any, Tuple

import torch
from torch import Tensor

from .base import Sampler


class RectifiedFlowSampler(Sampler):
    """n-step Euler sampler for a Rectified Flow process."""

    def __init__(self, process: Any, n_steps: int = 50) -> None:
        """Initializes the RectifiedFlowSampler.

        Args:
            process: The `flow_matching.rectified_flow.RectifiedFlow`
                process instance.
            n_steps: Number of Euler integration steps.
        """
        super().__init__(process)
        self.n_steps = n_steps

    def sample(
        self,
        shape: Tuple[int, ...],
        velocity_model: Any,
        device: str = "cpu",
    ) -> Tuple[Tensor, Tensor]:
        """Samples by integrating the Rectified Flow ODE from t=T to t=0.

        Starting from noise x_T ~ `process.sample_prior`, takes
        `n_steps` uniform Euler steps backwards in time:

            x <- x + (t_{i+1} - t_i) * v_theta(x, t_i),

        where t_{i+1} - t_i < 0. Since v_theta regresses x_1 - x_0
        (pointing from data towards noise), stepping with a negative
        dt moves the sample towards the data distribution.

        Args:
            shape: The shape of the tensor to sample, formatted as
                (n_images, channels, *spatial_dims).
            velocity_model: Velocity model callable, invoked as
                velocity_model(x, t) with t of shape (n_images,).
            device: Device on which to run the integration.

        Returns:
            Tuple of (initial prior noise, final sample in [0, 1]),
            mirroring `ProbabilityFlow.sample`.
        """
        x = self.process.sample_prior(shape, device=device)
        prior = x.clone()

        # Uniform time grid from T down to 0; dt is negative.
        t = torch.linspace(
            self.process.T, 0.0, self.n_steps + 1, device=device
        )
        dt = t[1:] - t[:-1]

        with torch.no_grad():
            for i in range(self.n_steps):
                t_i = t[i].expand(shape[0])
                x = x + dt[i] * velocity_model(x, t_i)

        # Revert the [-1, 1] normalization to [0, 1] for visualization.
        x_0 = ((x + 1) / 2).clamp(0.0, 1.0)

        return prior, x_0
