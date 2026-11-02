#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sampling/base.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Abstract base class for sampling strategies in diffusion models.

License:
    Proprietary. All rights reserved.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

from torch import Tensor


class Sampler(ABC):
    """Abstract base class for sampling strategies.

    All concrete sampling implementations must inherit from this class
    and implement the sample method.
    """

    def __init__(self, process: Any) -> None:
        """Initializes the Sampler with a diffusion process.

        Args:
            process: The diffusion process object.
        """
        self.process = process

    @abstractmethod
    def sample(self, shape: Tuple[int, ...]) -> Tensor:
        """Samples from the diffusion process.

        Args:
            shape: The shape of the tensor to sample.

        Returns:
            A tensor sampled from the process.
        """
