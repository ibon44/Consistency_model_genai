"""
imageGenerativeAI.diffusion

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    This module provides implementations of various diffusion models.
    It includes Variance Preserving, Variance Exploding, and Sub-Variance
    Preserving Diffusion models.

License:
    Proprietary. All rights reserved.
"""

__all__ = [
    "DiffusionProcess",
    "VariancePreservingDiffusion",
    "VarianceExplodingDiffusion",
    "SubVariancePreservingDiffusion",
    "LinearBetaSchedule",
]

from .base import DiffusionProcess
from .schedules import LinearBetaSchedule
from .subvp import SubVariancePreservingDiffusion
from .ve import VarianceExplodingDiffusion
from .vp import VariancePreservingDiffusion
