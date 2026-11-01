"""
flow_matching/__init__.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    This module provides flow-matching processes, an alternative to the
    SDE-based diffusion processes in `diffusion/`. It includes the base
    FlowProcess abstraction and the Rectified Flow implementation, used
    as a second candidate teacher for consistency distillation.

License:
    Proprietary. All rights reserved.
"""

__all__ = [
    "FlowProcess",
    "RectifiedFlow",
]

from .base import FlowProcess
from .rectified_flow import RectifiedFlow
