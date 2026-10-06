#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sampling/exponential_integrator.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    This module implements the Exponential Integrator sampling method
    for diffusion models.

License:
    Proprietary. All rights reserved.
"""

from typing import Tuple

import torch
from torch import Tensor

from diffusion.base import DiffusionProcess

from .base import Sampler


class ExponentialIntegrator(Sampler):
    
    """Exponential integrator with stochastic noise for diffusion models."""

    def __init__(
        self,
        process: DiffusionProcess,
        model: torch.nn.Module,
        num_steps: int = 1000,
        noise_scale: float = 1.0
    ) -> None:
        """Initializes the exponential integrator sampler.

        Args:
            process: A VP or sub-VP diffusion process (must expose
                `beta_schedule`).
            model: The score-based model.
            num_steps: Number of integration steps.
            noise_scale: Multiplier on the reverse-time diffusion noise.
                1.0 reproduces the exact reverse SDE; values below 1.0
                trade sample diversity for stability (SDE-ancestral
                sampling trick, akin to Song et al., 2021).
        """
        super().__init__(process)
        self.model = model
        self.num_steps = num_steps
        self.noise_scale = noise_scale

    def phi(self, z: Tensor) -> Tensor:
        return torch.where(
            torch.abs(z) < 1e-4,
            1 + z / 2 + z ** 2 / 6,
            (torch.exp(z) - 1) / z
        )

    def sample(self, shape: Tuple[int, ...]) -> Tensor:
        """Samples using an exponential integrator for the VP/sub-VP reverse SDE.

        Reverse-time SDE: dx = [-0.5*beta(t)*x - beta(t)*score(x,t)] dt
        + sqrt(beta(t)) dw_bar. The linear part (-0.5*beta(t)*x) is
        integrated exactly (matrix exponential), while the score-driven
        forcing term is integrated with the phi function to reduce local
        truncation error (Zhang & Chen, 2023, "Fast Sampling of Diffusion
        Models with Exponential Integrator").

        Args:
            shape: The shape of the tensor to sample.

        Returns:
            Tuple of (initial prior noise, final sample).
        """
        device = next(self.model.parameters()).device
        x = self.process.sample_prior(shape, device=device)
        prior = x.clone()

        time_steps = torch.linspace(1.0, 1e-3, self.num_steps, device=device)
        # time_steps is decreasing, so dt is negative and already points
        # backward in time (same convention as EulerMaruyama and
        # PredictorCorrector).
        dt = (
            time_steps[1] - time_steps[0] if self.num_steps > 1
            else -time_steps[0]
        )

        for t in time_steps:
            t_vec = torch.ones(x.shape[0], device=device) * t
            beta_t = self.process.beta_schedule.beta(t_vec).view(-1, 1, 1, 1)

            with torch.no_grad():
                score = self.model(x, t_vec)

            A = -0.5 * beta_t
            f = -beta_t * score  # reverse-time score-driven forcing term

            exp_Adt = torch.exp(A * dt)
            phi_Adt = self.phi(A * dt)

            # Deterministic exponential part
            x_deterministic = exp_Adt * x + phi_Adt * dt * f

            # Stochastic part (additive noise)
            noise = torch.randn_like(x)
            diffusion_coeff = torch.sqrt(beta_t) * self.noise_scale
            x_stochastic = diffusion_coeff * torch.sqrt(torch.abs(dt)) * noise

            # Combine
            x = x_deterministic + x_stochastic

        return prior, x
