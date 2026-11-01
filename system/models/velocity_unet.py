#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/velocity_unet.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Velocity model for flow-matching processes (see
    `flow_matching/rectified_flow.py`), reusing the same UNet backbone
    architecture as `models/score_unet.py::ScoreNet`. Unlike ScoreNet,
    its raw output is *not* rescaled by a marginal_prob_std function,
    since it regresses a velocity field directly rather than a score.

License:
    Proprietary. All rights reserved.
"""

from typing import List

import torch

from .base import BaseScoreModel
from .score_unet import ScoreNet


class VelocityNet(BaseScoreModel):
    """UNet-based velocity model for flow-matching processes."""

    def __init__(
        self,
        channels: List[int] = [32, 64, 128, 256],
        embed_dim: int = 256,
    ) -> None:
        """Initializes the VelocityNet.

        Reuses the same UNet backbone as ScoreNet, but without the
        marginal_prob_std output scaling.

        Args:
            channels: List of channel sizes for each UNet block.
            embed_dim: Embedding dimension for time encoding.
        """
        super().__init__()
        self.backbone = ScoreNet(
            channels=channels, embed_dim=embed_dim, marginal_prob_std=None
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predicts the velocity field at (x, t).

        Args:
            x: Input tensor of shape (B, 1, H, W).
            t: Time tensor of shape (B,).

        Returns:
            Predicted velocity tensor of shape (B, 1, H, W).
        """
        return self.backbone(x, t)
