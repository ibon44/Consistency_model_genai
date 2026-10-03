#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consistency/distances.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Default distance functions d(x, y) for the consistency
    distillation/training losses (`consistency/distillation.py`,
    `consistency/ct_loss.py`). Unlike the teacher/schedule/model
    arguments (which change per run), these are fixed formulas from
    the literature, so they are provided as ready-to-use defaults:

    - squared_l2_distance: d(x,y) = ||x-y||_2^2, the distance used in
      the original Consistency Models paper (Song et al., 2023) for
      simple datasets (they use LPIPS instead for natural images,
      which needs a pretrained perceptual network we don't have here
      -- L2 is the right choice for MNIST).
    - pseudo_huber_distance: d(x,y) = sqrt(||x-y||_2^2 + c^2) - c, the
      distance recommended in the follow-up "Improved Techniques for
      Training Consistency Models" (Song & Dhariwal, 2024) as a more
      stable alternative to squared L2 (less sensitive to outlier
      pixels/gradients early in training).

License:
    Proprietary. All rights reserved.
"""

from typing import Optional

import torch


def squared_l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes the mean squared L2 distance between two batches.

    Args:
        a: Tensor of shape (B, ...).
        b: Tensor of shape (B, ...), same shape as a.

    Returns:
        Scalar tensor: mean over the batch of the per-sample squared
        L2 distance (summed over all non-batch dimensions).
    """
    return torch.mean(
        torch.sum((a - b) ** 2, dim=list(range(1, a.dim())))
    )


def pseudo_huber_distance(
    a: torch.Tensor, b: torch.Tensor, c: Optional[float] = None
) -> torch.Tensor:
    """Computes the mean Pseudo-Huber distance between two batches.

    Args:
        a: Tensor of shape (B, ...).
        b: Tensor of shape (B, ...), same shape as a.
        c: Huber constant. If None, uses the data-dimension-adaptive
            default from Song & Dhariwal (2024): c = 0.00054*sqrt(D),
            where D is the number of elements per sample (product of
            all non-batch dimensions).

    Returns:
        Scalar tensor: mean over the batch of the per-sample
        Pseudo-Huber distance (summed over all non-batch dimensions).
    """
    if c is None:
        dim = 1
        for s in a.shape[1:]:
            dim *= s
        c = 0.00054 * (dim ** 0.5)
    squared_dist = torch.sum((a - b) ** 2, dim=list(range(1, a.dim())))
    return torch.mean(torch.sqrt(squared_dist + c ** 2) - c)
