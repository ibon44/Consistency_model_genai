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
      of the teacher's ODE trajectory used at training iteration k.
      N(k) = ceil( sqrt( k/K * ((s1+1)^2 - s0^2) + s0^2 ) - 1 ) + 1
    - EMADecaySchedule: mu(k), the decay rate for the EMA copy of the
      consistency model's weights (target network), coupled to N(k).
      mu(k) = exp( s0 * log(mu0) / N(k) )
    - karras_time_steps: the N(k)-point time grid t_1=eps < ... < t_N=T
      along which the teacher trajectory is discretized (Karras et al.,
      2022 noise schedule, rho=7 by default).
      t_i = ( eps^(1/rho) + (i-1)/(N-1) * (T^(1/rho) - eps^(1/rho)) )^rho

License:
    Proprietary. All rights reserved.
"""

import math

import torch


class DiscretizationSchedule:
    """N(k): number of discretization steps at training iteration k."""

    def __init__(self, K: int, s0: int = 2, s1: int = 150) -> None:
        """Initializes the DiscretizationSchedule.

        Args:
            K: Total number of training iterations.
            s0: Number of discretization steps at k=0 (start of
                training).
            s1: Number of discretization steps at k=K (end of
                training); N(k) approaches s1+1.
        """
        self.K = K
        self.s0 = s0
        self.s1 = s1

    def __call__(self, k: int) -> int:
        """Returns the number of discretization steps N(k).

        Args:
            k: Current training iteration, in [0, K].

        Returns:
            Number of discretization steps N(k), an integer >= s0.
        """
        sqrt_term = (k / self.K) * ((self.s1 + 1) ** 2 - self.s0 ** 2) + self.s0 ** 2
        N_k = math.ceil(math.sqrt(sqrt_term) - 1) + 1
        return N_k


class EMADecaySchedule:
    """mu(k): EMA decay rate for the target network, coupled to N(k)."""

    def __init__(
        self, discretization_schedule: DiscretizationSchedule, mu0: float = 0.95
    ) -> None:
        """Initializes the EMADecaySchedule.

        Args:
            discretization_schedule: The N(k) schedule this decay is
                coupled to (reuses its s0, and is called at the same k
                to get N(k)).
            mu0: Base EMA decay rate.
        """
        self.discretization_schedule = discretization_schedule
        self.mu0 = mu0

    def __call__(self, k: int) -> float:
        """Returns the EMA decay rate mu(k).

        Args:
            k: Current training iteration.

        Returns:
            EMA decay rate mu(k) in [0, 1).
        """
        N_k = self.discretization_schedule(k)
        mu_k = torch.exp(self.discretization_schedule.s0 * torch.log(torch.tensor(self.mu0)) / N_k)
        return mu_k.item()


def karras_time_steps(
    N: int, eps: float, T: float, rho: float = 7.0
) -> torch.Tensor:
    """Computes the Karras et al. (2022) time discretization grid.

    Args:
        N: Number of discretization points.
        eps: Lower time bound (t_1 = eps).
        T: Upper time bound (t_N = T).
        rho: Controls how much the grid is skewed towards eps
            (higher rho = denser sampling near eps).

    Returns:
        1-D tensor of shape (N,) with t_1=eps < ... < t_N=T.
    """
    t = torch.linspace(eps ** (1 / rho), T ** (1 / rho), N) ** rho
    return t
