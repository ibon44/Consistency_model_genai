"""
imageGenerativeAI.diffusion

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    This module provides the Variance Exploding (VE) diffusion process
    used as the diffusion teacher.

License:
    Proprietary. All rights reserved.
"""

__all__ = [
    "DiffusionProcess",
    "VarianceExplodingDiffusion",
]

from .base import DiffusionProcess
from .ve import VarianceExplodingDiffusion
