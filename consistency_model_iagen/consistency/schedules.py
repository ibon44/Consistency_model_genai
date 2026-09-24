#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consistency/schedules.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Curriculum schedules used by consistency distillation/training
    (Song et al., 2023, Sec. 3 and Appendix C):

    - DiscretizationSchedule: N(k), the number of discretization steps
      of the teacher's ODE trajectory used at training iteration k
      (typically increasing over training).
    - EMADecaySchedule: mu(k), the decay rate for the EMA copy of the
      consistency model's weights (target network), typically coupled
      to N(k).

    TODO: implement `__call__` for both, following the paper's default
    piecewise-constant / exponential schedules.

License:
    Proprietary. All rights reserved.
"""

from abc import ABC, abstractmethod


class DiscretizationSchedule(ABC):
    """Abstract base class for the N(k) discretization curriculum."""

    @abstractmethod
    def __call__(self, k: int) -> int:
        """Returns the number of discretization steps at iteration k.

        Args:
            k: Current training iteration (or epoch, depending on the
                granularity chosen by the training script).

        Returns:
            Number of discretization steps N(k).
        """


class EMADecaySchedule(ABC):
    """Abstract base class for the EMA target-network decay mu(k)."""

    @abstractmethod
    def __call__(self, k: int) -> float:
        """Returns the EMA decay rate at iteration k.

        Args:
            k: Current training iteration.

        Returns:
            EMA decay rate mu(k) in [0, 1).
        """
