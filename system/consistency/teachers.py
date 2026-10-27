#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consistency/teachers.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Adapters that expose a single deterministic ODE step from a trained
    teacher model, so that `consistency/distillation.py` can run
    consistency distillation against either teacher without knowing
    which one it is:

    - VETeacherAdapter wraps a `diffusion.ve.VarianceExplodingDiffusion`
      process and a trained ScoreNet, taking one Heun (default) or
      Euler step along the probability flow ODE (see `sampling/
      probability_flow.py` for the corresponding full-trajectory
      solver), on a Karras grid spaced in sigma (`time_grid`).
    - RectifiedFlowTeacherAdapter wraps a `flow_matching.rectified_flow.
      RectifiedFlow` process and a trained VelocityNet, taking one
      Euler step along the (deterministic, already-straight) RF ODE.

    Both also expose `forward_sample` (x_0 -> x_t along the process'
    own forward noising, used by `consistency/distillation.py` to build
    x_{t_{n+1}} before taking the teacher's ODE step) and `sigma_t`/
    `T`/`eps` (shared with `models.consistency_wrapper.
    ConsistencyWrapper`) so c_skip/c_out can be computed without a
    teacher-specific branch.

License:
    Proprietary. All rights reserved.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch

from .schedules import karras_sigma_time_steps, karras_time_steps


class TeacherAdapter(ABC):
    """Abstract adapter exposing a single teacher ODE step."""

    def __init__(self, process: Any, model: Callable, eps: float = 1.0e-5) -> None:
        """Initializes the TeacherAdapter.

        Args:
            process: The underlying diffusion or flow-matching process.
            model: The trained teacher network (score or velocity).
            eps: Lower time bound near t=0 where the consistency
                function's boundary condition f(x,eps)=x is enforced.
                `diffusion`/`flow_matching` processes don't store this
                themselves (it's a training-time tolerance, not part of
                the process definition), so it lives on the adapter;
                pass the same value used to build the matching
                `models.consistency_wrapper.ConsistencyWrapper`.
        """
        self.process = process
        self.model = model
        self.eps = eps
        self.T = process.T

    @abstractmethod
    def forward_sample(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Samples x_t from the process' own forward noising of x_0.

        Args:
            x_0: Mini-batch of clean data.
            t: Time tensor, shape (B,).

        Returns:
            Noised sample x_t, same shape as x_0.
        """

    @abstractmethod
    def ode_step(
        self, x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor
    ) -> torch.Tensor:
        """Takes one deterministic ODE step from t to t_prev < t.

        Args:
            x_t: Current state at time t.
            t: Current time tensor, shape (B,).
            t_prev: Target (earlier) time tensor, shape (B,).

        Returns:
            Estimated state at time t_prev.
        """

    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Delegates to the underlying process' sigma_t (see
        `diffusion.base.DiffusionProcess.sigma_t` /
        `flow_matching.base.FlowProcess.sigma_t`)."""
        return self.process.sigma_t(t)

    def time_grid(self, N: int) -> torch.Tensor:
        """N-point time grid eps=t_1 < ... < t_N=T for distillation."""
        return karras_time_steps(N, self.eps, self.T)


class VETeacherAdapter(TeacherAdapter):
    """Teacher adapter for a VE-diffusion probability-flow ODE."""

    def __init__(
        self, process: Any, model: Callable, eps: float = 1.0e-5, heun: bool = True
    ) -> None:
        """Initializes the VETeacherAdapter.

        Args:
            process: The VE diffusion process.
            model: The trained score network.
            eps: Lower time bound (see TeacherAdapter).
            heun: Take a 2nd-order Heun step (as in Song et al., 2023)
                instead of a 1st-order Euler step.
        """
        super().__init__(process, model, eps)
        self.heun = heun

    def time_grid(self, N: int) -> torch.Tensor:
        """Karras grid spaced in sigma rather than in t."""
        return karras_sigma_time_steps(
            N, self.eps, self.T, self.process.sigma_t, self.process.sigma_inverse
        )

    def _pf_ode_rhs(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Probability flow ODE: dx/dt = f(x,t) - 0.5*g(t)^2*score(x,t)."""
        shape = (-1, *([1] * (x_t.dim() - 1)))
        with torch.no_grad():
            score = self.model(x_t, t)
        drift = self.process.drift_coefficient(x_t, t)
        diffusion = self.process.diffusion_coefficient(t).view(*shape)
        return drift - 0.5 * diffusion ** 2 * score

    def forward_sample(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        shape = (-1, *([1] * (x_0.dim() - 1)))
        noise = torch.randn_like(x_0)
        mu = self.process.mu_t(x_0, t)
        sigma = self.process.sigma_t(t).view(*shape)
        return mu + sigma * noise

    def ode_step(
        self, x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor
    ) -> torch.Tensor:
        shape = (-1, *([1] * (x_t.dim() - 1)))
        dt = (t_prev - t).view(*shape)

        d1 = self._pf_ode_rhs(x_t, t)
        x_euler = x_t + dt * d1
        if not self.heun:
            return x_euler
        d2 = self._pf_ode_rhs(x_euler, t_prev)
        return x_t + dt * 0.5 * (d1 + d2)


class RectifiedFlowTeacherAdapter(TeacherAdapter):
    """Teacher adapter for the Rectified Flow ODE."""

    def forward_sample(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        noise = torch.randn_like(x_0)
        return self.process.interpolate(x_0, noise, t)

    def ode_step(
        self, x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor
    ) -> torch.Tensor:
        shape = (-1, *([1] * (x_t.dim() - 1)))

        with torch.no_grad():
            velocity = self.model(x_t, t)

        # The Rectified Flow ODE dx/dt = v(x,t) has a constant velocity
        # along a trajectory by construction, so a single Euler step is
        # exact (not an approximation, unlike VETeacherAdapter.ode_step).
        dt = (t_prev - t).view(*shape)
        return x_t + dt * velocity
