#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/device.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    This module provides utility functions for device management in PyTorch.
    It includes functions to get the current device, move tensors to the device,
    convert tensors to NumPy arrays, and extract scalar values from tensors.

License:
    Proprietary. All rights reserved.
"""


from typing import Any, Union

import torch

DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def get_device() -> torch.device:
    """Returns the current torch device.

    Returns:
        The torch.device object in use.
    """
    return DEVICE


def to_device(obj: Any) -> Any:
    """Moves tensors or collections of tensors to the current device.

    Args:
        obj: Tensor, list, tuple, or dict to move.

    Returns:
        The object moved to the current device.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(DEVICE)
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(x) for x in obj)
    if isinstance(obj, dict):
        return {k: to_device(v) for k, v in obj.items()}
    return obj


def to_numpy(tensor: torch.Tensor) -> Any:
    """Converts a tensor to a NumPy array.

    Args:
        tensor: The tensor to convert.

    Returns:
        The tensor as a NumPy array.

    Raises:
        TypeError: If the input is not a torch.Tensor.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    raise TypeError("Expected a torch.Tensor as input.")


def scalar(tensor: torch.Tensor) -> Union[int, float]:
    """Extracts a Python scalar from a single-element tensor.

    Args:
        tensor: The tensor to extract from.

    Returns:
        The scalar value.

    Raises:
        ValueError: If the tensor does not contain exactly one element.
    """
    if isinstance(tensor, torch.Tensor) and tensor.numel() == 1:
        return tensor.item()
    raise ValueError("Expected a scalar tensor with exactly one element.")
