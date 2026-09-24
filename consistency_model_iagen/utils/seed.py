#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/seed.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    This module provides a utility function to set random seeds for
    reproducibility across various libraries including Python's random,
    NumPy, and PyTorch. It also configures PyTorch's CUDA backend for
    deterministic behavior.

License:
    Proprietary. All rights reserved.
"""


import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across libraries.

    This function sets the seed for Python's random module, NumPy,
    and PyTorch (CPU and all GPUs). It also configures PyTorch's
    CUDA backend for deterministic behavior and sets the
    PYTHONHASHSEED environment variable.

    Args:
        seed: The seed value to use. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multiple GPUs

    # Ensure deterministic behavior in CUDA (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[INFO] Seed set to {seed}")
