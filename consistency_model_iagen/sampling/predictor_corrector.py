#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sampling/predictor_corrector.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    This module implements the Predictor-Corrector sampling method
    for diffusion models.

License:
    Proprietary. All rights reserved.
"""

from typing import Tuple

import torch
from torch import Tensor

from .base import Sampler


class PredictorCorrector(Sampler):
    """Predictor-Corrector sampler for diffusion models.

    This sampler alternates between predictor (Euler-Maruyama) and
    corrector (Langevin dynamics) steps to generate samples.
    """

    def __init__(
        self,
        process: object,
        model: torch.nn.Module,
        num_steps: int = 100,
        corrector_steps: int = 1,
        snr: float = 0.15
    ) -> None:
        """Initializes the PredictorCorrector sampler.

        Args:
            process: The diffusion process object.
            model: The score-based model.
            num_steps: Number of predictor steps.
            corrector_steps: Number of corrector steps per predictor step.
            snr: Signal-to-noise ratio for Langevin dynamics.
        """
        super().__init__(process)
        self.model = model
        self.num_steps = num_steps
        self.corrector_steps = corrector_steps
        self.snr = snr

    def sample(self, shape: Tuple[int, ...]) -> Tensor:
        """Samples from the diffusion process.

        Args:
            shape: The shape of the tensor to sample.

        Returns:
            Tuple of (initial prior noise, final sample tensor).
        """
        device = next(self.model.parameters()).device

        x = self.process.sample_prior(shape, device=device)
        prior = x.clone()

        t_steps = torch.linspace(
            self.process.T, 1e-3, self.num_steps, device=device
        )
        # times are decreasing, so dt is negative and already points
        # backward in time (see EulerMaruyama for the same convention).
        dt = t_steps[1] - t_steps[0] if self.num_steps > 1 else -t_steps[0]

        for t in t_steps:
            vec_t = torch.full((shape[0],), t, device=device)

            # Predictor: Euler-Maruyama step on the reverse-time SDE
            reverse_drift = self.process.backward_drift_coefficient(
                x, vec_t, self.model
            )
            diffusion = self.process.diffusion_coefficient(vec_t)
            noise = torch.randn_like(x)
            x = (
                x
                + reverse_drift * dt
                + diffusion[:, None, None, None]
                * torch.sqrt(torch.abs(dt))
                * noise
            )

            # Corrector: Langevin dynamics
            for _ in range(self.corrector_steps):
                grad = self.model(x, vec_t)
                noise = torch.randn_like(x)
                grad_norm = torch.norm(
                    grad.reshape(shape[0], -1), dim=-1
                ).mean()
                noise_norm = torch.norm(
                    noise.reshape(shape[0], -1), dim=-1
                ).mean()
                step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2
                x = (
                    x
                    + step_size * grad
                    + torch.sqrt(2.0 * step_size) * noise
                )

        return prior, x
