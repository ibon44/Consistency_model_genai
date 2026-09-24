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

    TODO: implement `sample`.

License:
    Proprietary. All rights reserved.
"""

from typing import Any, Tuple

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
    ) -> Tensor:
        """Samples by integrating the Rectified Flow ODE.

        Args:
            shape: The shape of the tensor to sample.
            velocity_model: Velocity model callable, invoked as
                velocity_model(x, t).
            device: Device on which to run the integration.

        Returns:
            Generated sample tensor.
        """
        raise NotImplementedError
