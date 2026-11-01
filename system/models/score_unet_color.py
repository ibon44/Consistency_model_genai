#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/score_unet_color.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    RGB (3-channel) variant of the UNet-based score model defined in
    score_unet.py.

License:
    Proprietary. All rights reserved.
"""

from typing import Callable, List

import torch
import torch.nn as nn

from .base import BaseScoreModel
from .score_unet import Dense, GaussianRandomFourierFeatures


class ScoreNetColor(BaseScoreModel):
    """UNet-based score model for diffusion with time embedding (RGB version)."""

    def __init__(
        self,
        marginal_prob_std: Callable[[torch.Tensor], torch.Tensor],
        channels: List[int] = [32, 64, 128, 256],
        embed_dim: int = 256
    ) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            GaussianRandomFourierFeatures(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        self.conv1 = nn.Conv2d(3, channels[0], 3, stride=1, padding=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, channels[3])

        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, channels[2])

        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 3, 3, stride=1, padding=1)

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W).
            t: Time tensor of shape (B,).

        Returns:
            Output tensor of shape (B, 3, H, W).
        """
        embed = self.act(self.embed(t))

        h1 = self.conv1(x)
        h1 = h1 + self.dense1(embed)
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)

        h2 = self.conv2(h1)
        h2 = h2 + self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        h3 = self.conv3(h2)
        h3 = h3 + self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)

        h4 = self.conv4(h3)
        h4 = h4 + self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        h = self.tconv4(h4)
        h = h + self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)

        # Asegura que h y h3 tengan la misma forma espacial antes del cat
        if h.shape[2:] != h3.shape[2:]:
            min_h, min_w = min(h.shape[2], h3.shape[2]), min(h.shape[3], h3.shape[3])
            h = h[:, :, :min_h, :min_w]
            h3 = h3[:, :, :min_h, :min_w]

        h = self.tconv3(torch.cat([h, h3], dim=1))
        h = h + self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)

        if h.shape[2:] != h2.shape[2:]:
            min_h, min_w = min(h.shape[2], h2.shape[2]), min(h.shape[3], h2.shape[3])
            h = h[:, :, :min_h, :min_w]
            h2 = h2[:, :, :min_h, :min_w]

        h = self.tconv2(torch.cat([h, h2], dim=1))
        h = h + self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)

        if h.shape[2:] != h1.shape[2:]:
            min_h, min_w = min(h.shape[2], h1.shape[2]), min(h.shape[3], h1.shape[3])
            h = h[:, :, :min_h, :min_w]
            h1 = h1[:, :, :min_h, :min_w]

        h = self.tconv1(torch.cat([h, h1], dim=1))

        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h

