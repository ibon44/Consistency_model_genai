#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data/datasets.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Defines a function to create a DataLoader for the MNIST dataset,
    normalized to [-1, 1] as expected by the diffusion processes in
    this project (see diffusion/*.loss_function and sample_prior).

License:
    Proprietary. All rights reserved.
"""

from typing import Optional

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_dataloader(
    batch_size: int = 64,
    digit: Optional[int] = None,
    train: bool = True,
    data_dir: str = "./data",
) -> DataLoader:
    """Creates a DataLoader for the MNIST dataset.

    Downloads MNIST if not present, scales pixel values to [-1, 1], and
    optionally filters the dataset down to a single digit class.

    Args:
        batch_size: Number of samples per batch.
        digit: If given, only images with this label (0-9) are kept.
        train: Whether to load the training or the test split.
        data_dir: Root directory for the downloaded dataset.

    Returns:
        DataLoader iterating over the (optionally filtered) dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])

    dataset = datasets.MNIST(
        root=data_dir, train=train, download=True, transform=transform
    )

    if digit is not None:
        indices = np.where(np.array(dataset.targets) == digit)[0]
        dataset = Subset(dataset, indices)

    return DataLoader(dataset, batch_size=batch_size, shuffle=train)
