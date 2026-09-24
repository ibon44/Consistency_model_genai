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
    Rectified Flow), since it only calls f_theta(x, t).

    TODO: implement `sample`.

License:
    Proprietary. All rights reserved.
"""

from typing import Any, List, Optional, Tuple

from torch import Tensor

from .base import Sampler


class ConsistencySampler(Sampler):
    """1-step / multistep sampler for a trained consistency model."""

    def __init__(self, process: Any) -> None:
        """Initializes the ConsistencySampler.

        Args:
            process: The teacher process the consistency model was
                distilled from (only used for `T`/`eps`/`sigma_t`).
        """
        super().__init__(process)

    def sample(
        self,
        shape: Tuple[int, ...],
        consistency_model: Any,
        device: str = "cpu",
        intermediate_times: Optional[List[float]] = None,
    ) -> Tensor:
        """Generates samples with the consistency model.

        Args:
            shape: The shape of the tensor to sample.
            consistency_model: Trained f_theta, invoked as
                consistency_model(x, t).
            device: Device on which to run sampling.
            intermediate_times: Optional decreasing sequence of extra
                time points for multistep sampling. None (default)
                performs 1-step generation.

        Returns:
            Generated sample tensor.
        """
        raise NotImplementedError
