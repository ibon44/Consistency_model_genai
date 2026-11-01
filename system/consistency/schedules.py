#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consistency/schedules.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
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
    - ConstantSchedule: a fixed N or mu at every iteration. This is what
      consistency *distillation* uses in the paper (N=18, mu=0.95 on
      CIFAR-10, with a Heun teacher); the N(k)/mu(k) curriculum above
      is the *consistency training* (CT) recipe.
    - karras_time_steps: the N(k)-point time grid t_1=eps < ... < t_N=T
      along which the teacher trajectory is discretized (Karras et al.,
      2022 noise schedule, rho=7 by default), applied directly in t.
      t_i = ( eps^(1/rho) + (i-1)/(N-1) * (T^(1/rho) - eps^(1/rho)) )^rho
    - karras_sigma_time_steps: the same Karras spacing applied in sigma
      (as in Karras et al. / Song et al., where sigma = t), then mapped
      back to t through the process' inverse sigma_t. For VE
      (sigma_t ~ sqrt(t) near 0) the t-space grid wastes many points
      at imperceptible noise levels; the sigma-space grid does not.

License:
    Proprietary. All rights reserved.
"""

import math
from typing import Callable

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


class ConstantSchedule:
    """A schedule that returns the same value at every iteration."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, k: int) -> float:
        return self.value


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


def karras_sigma_time_steps(
    N: int,
    eps: float,
    T: float,
    sigma_t: Callable[[torch.Tensor], torch.Tensor],
    sigma_inverse: Callable[[torch.Tensor], torch.Tensor],
    rho: float = 7.0,
) -> torch.Tensor:
    """Karras et al. (2022) grid built in sigma, returned as times.

    Args:
        N: Number of discretization points.
        eps: Lower time bound (t_1 = eps).
        T: Upper time bound (t_N = T).
        sigma_t: The process' marginal std function t -> sigma.
        sigma_inverse: Its inverse sigma -> t.
        rho: Controls how much the grid is skewed towards sigma_min.

    Returns:
        1-D tensor of shape (N,) with t_1=eps < ... < t_N=T, whose
        sigma_t values follow the Karras spacing between sigma_t(eps)
        and sigma_t(T).
    """
    sigma_min = sigma_t(torch.tensor([eps])).item()
    sigma_max = sigma_t(torch.tensor([float(T)])).item()
    sigma = torch.linspace(sigma_min ** (1 / rho), sigma_max ** (1 / rho), N) ** rho
    t = sigma_inverse(sigma)
    # Pin the endpoints exactly (float round-trip through sigma_inverse
    # would otherwise leave t_1 slightly off eps).
    t[0], t[-1] = eps, T
    return t
