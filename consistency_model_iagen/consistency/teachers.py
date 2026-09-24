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
      process and a trained ScoreNet, taking one Euler/Heun step along
      the probability flow ODE (see `sampling/probability_flow.py` for
      the corresponding full-trajectory solver).
    - RectifiedFlowTeacherAdapter wraps a `flow_matching.rectified_flow.
      RectifiedFlow` process and a trained VelocityNet, taking one
      Euler step along the (deterministic, already-straight) RF ODE.

    Both expose `sigma_t`/`T`/`eps` (shared with `models.
    consistency_wrapper.ConsistencyWrapper`) so c_skip/c_out can be
    computed without a teacher-specific branch.

    TODO: implement `ode_step` for both adapters.

License:
    Proprietary. All rights reserved.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch


class TeacherAdapter(ABC):
    """Abstract adapter exposing a single teacher ODE step."""

    T: float
    eps: float

    def __init__(self, process: Any, model: Callable) -> None:
        """Initializes the TeacherAdapter.

        Args:
            process: The underlying diffusion or flow-matching process.
            model: The trained teacher network (score or velocity).
        """
        self.process = process
        self.model = model

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


class VETeacherAdapter(TeacherAdapter):
    """Teacher adapter for a VE-diffusion probability-flow ODE."""

    def ode_step(
        self, x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class RectifiedFlowTeacherAdapter(TeacherAdapter):
    """Teacher adapter for the Rectified Flow ODE."""

    def ode_step(
        self, x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError
