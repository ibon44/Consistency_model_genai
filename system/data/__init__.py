"""
data/__init__.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    This module provides data loading utilities for image datasets.
    It includes functions to create data loaders for training and testing.

License:
    Proprietary. All rights reserved.
"""

__all__ = [
    "get_dataloader",
]

from .datasets import get_dataloader
