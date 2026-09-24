#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/base.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Base class for score models in the diffusion framework.

    This module defines the BaseScoreModel class, which serves as a
    template for implementing various score models. The class inherits
    from torch.nn.Module and provides an abstract method for the forward
    pass, as well as a method for computing log-likelihood.

License:
    Proprietary. All rights reserved.
"""

from abc import ABC, abstractmethod

import torch


class BaseScoreModel(torch.nn.Module, ABC):
    """Abstract base class for score models in diffusion frameworks.

    This class provides a template for implementing score models.
    """

    @abstractmethod
    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Performs the forward pass of the score model.

        Args:
            x: Input tensor.
            t: Time tensor.

        Returns:
            Output tensor.
        """

    def compute_log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the log-likelihood for the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Log-likelihood tensor.

        Raises:
            NotImplementedError: If the model does not support
            log-likelihood computation.
        """
        raise NotImplementedError(
            "This model does not support log-likelihood."
        )
