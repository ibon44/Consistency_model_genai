#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sampling/consistency_sampler.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Sampler for a trained consistency model (`models.consistency_wrapper.
    ConsistencyWrapper`), implementing Algorithm 1 from Song et al.,
    2023:

    - 1-step: x = f_theta(noise, T).
    - Multistep: alternate f_theta calls with re-noising to intermediate
      time points t_2 > t_3 > ... , trading extra steps (and extra
      network evaluations) for sample quality — the core speed/quality
      tradeoff this project measures (see `evaluation/compare.py`).

    Works for a consistency model distilled from either teacher (VE or
    Rectified Flow), since it only calls f_theta(x, t) and the process'
    `T`/`sigma_t`/`sample_prior`. The re-noising step assumes a
    zero-drift forward process (mu_t(x_0,t) = x_0, as in VE) so that
    adding noise directly to the current f_theta prediction is a valid
    way to land back on the trajectory at an earlier time -- this holds
    for the VE teacher this project trains against; a model distilled
    from Rectified Flow (whose forward path also rescales x_0, not just
    adds noise) would need `process.interpolate` here instead.

    Returned samples are in the same [-1, 1] range as the training data
    (see `data/datasets.py`), not rescaled to [0, 1].

License:
    Proprietary. All rights reserved.
"""

from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor

from consistency.schedules import karras_sigma_time_steps

from .base import Sampler


class ConsistencySampler(Sampler):
    """1-step / multistep sampler for a trained consistency model."""

    def __init__(self, process: Any, eps: float = 1.0e-5) -> None:
        """Initializes the ConsistencySampler.

        Args:
            process: The teacher process the consistency model was
                distilled from (only used for `T`/`sigma_t`/
                `sample_prior`).
            eps: Boundary time where f_theta(x,eps)=x holds; must match
                the `eps` used to build the `ConsistencyWrapper` and
                the `TeacherAdapter` it was distilled with.
        """
        super().__init__(process)
        self.eps = eps

    def karras_intermediate_times(
        self, n_steps: int, rho: float = 7.0
    ) -> Optional[List[float]]:
        """Default re-noising times for n_steps consistency evaluations.

        Builds an (n_steps + 1)-point Karras grid in sigma between
        sigma_t(eps) and sigma_t(T) and keeps only its interior points,
        so every re-noising step adds a non-zero amount of noise (the
        endpoints are T, where sampling starts, and eps, where
        re-noising would add nothing).

        Args:
            n_steps: Total number of consistency-model evaluations.
            rho: Karras skew parameter.

        Returns:
            None for n_steps <= 1, otherwise (n_steps - 1) strictly
            decreasing times in (eps, T).
        """
        if n_steps <= 1:
            return None
        grid = karras_sigma_time_steps(
            n_steps + 1, self.eps, self.process.T,
            self.process.sigma_t, self.process.sigma_inverse, rho,
        )
        return grid[1:-1].flip(0).tolist()

    def sample(
        self,
        shape: Tuple[int, ...],
        consistency_model: Any,
        device: str = "cpu",
        intermediate_times: Optional[List[float]] = None,
        x_T: Optional[Tensor] = None,
    ) -> Tensor:
        """Generates samples with the consistency model.

        Args:
            shape: The shape of the tensor to sample.
            consistency_model: Trained f_theta, invoked as
                consistency_model(x, t).
            device: Device on which to run sampling.
            intermediate_times: Optional strictly decreasing sequence
                of extra time points in (eps, T) for multistep sampling
                (tau_1 > tau_2 > ... , all < T). None (default)
                performs 1-step generation.
            x_T: Optional pre-sampled initial noise (shape `shape`), so
                e.g. 1-step vs. 4-step outputs can be compared from the
                exact same starting point. Defaults to a fresh draw
                from `self.process.sample_prior`.

        Returns:
            Generated sample tensor, shape `shape`, in [-1, 1].
        """
        batch_size = shape[0]
        value_shape = (-1, *([1] * (len(shape) - 1)))

        if x_T is None:
            x_T = self.process.sample_prior(shape, device=device)
        t = torch.full((batch_size,), self.process.T, device=device)

        # Every x_0 estimate is clamped to the data range before being
        # re-noised (clip_denoised in the reference implementation).
        with torch.no_grad():
            x = consistency_model(x_T, t).clamp(-1.0, 1.0)

        if intermediate_times:
            sigma_eps = self.process.sigma_t(
                torch.full((batch_size,), self.eps, device=device)
            )
            for tau in intermediate_times:
                z = torch.randn(*shape, device=device)
                sigma_tau = self.process.sigma_t(
                    torch.full((batch_size,), tau, device=device)
                )
                noise_scale = torch.sqrt(
                    torch.clamp(sigma_tau ** 2 - sigma_eps ** 2, min=0.0)
                )
                x_tau = x + noise_scale.view(*value_shape) * z

                t_tau = torch.full((batch_size,), tau, device=device)
                with torch.no_grad():
                    x = consistency_model(x_tau, t_tau).clamp(-1.0, 1.0)

        return x
