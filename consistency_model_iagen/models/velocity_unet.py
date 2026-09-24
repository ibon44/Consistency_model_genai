#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/velocity_unet.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Velocity model for flow-matching processes (see
    `flow_matching/rectified_flow.py`), reusing the same UNet backbone
    architecture as `models/score_unet.py::ScoreNet`. Unlike ScoreNet,
    its raw output is *not* rescaled by a marginal_prob_std function,
    since it regresses a velocity field directly rather than a score.

    TODO: implement, either by subclassing/reusing ScoreNet's blocks
    directly (construct it with `marginal_prob_std=None`) or by
    defining a thin wrapper module here.

License:
    Proprietary. All rights reserved.
"""

from typing import List

import torch

from .base import BaseScoreModel


class VelocityNet(BaseScoreModel):
    """UNet-based velocity model for flow-matching processes."""

    def __init__(
        self,
        channels: List[int] = [32, 64, 128, 256],
        embed_dim: int = 256,
    ) -> None:
        """Initializes the VelocityNet.

        Args:
            channels: List of channel sizes for each UNet block.
            embed_dim: Embedding dimension for time encoding.
        """
        super().__init__()
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Predicts the velocity field at (x, t).

        Args:
            x: Input tensor of shape (B, 1, H, W).
            t: Time tensor of shape (B,).

        Returns:
            Predicted velocity tensor of shape (B, 1, H, W).
        """
        raise NotImplementedError
