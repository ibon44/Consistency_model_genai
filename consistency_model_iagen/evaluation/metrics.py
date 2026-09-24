#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/metrics.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Quality and diversity metrics computed over a batch of generated
    samples using the reference classifier (`evaluation/classifier.py`),
    as required by objective 4 of the project proposal:

    - classification_accuracy: mean top-1 confidence / accuracy against
      the *conditioning* digit, when sampling from a digit-restricted
      model (see the DIGIT config knob in the training scripts).
    - label_distribution: histogram of predicted labels, used as a
      cheap proxy for mode coverage/diversity of an unconditional model.
    - label_entropy: Shannon entropy of `label_distribution`, a single
      scalar diversity summary (higher = more uniform across digits).

    TODO: implement all three.

License:
    Proprietary. All rights reserved.
"""

from typing import Optional

import torch


def classification_accuracy(
    classifier: torch.nn.Module,
    samples: torch.Tensor,
    target_digit: int,
) -> float:
    """Computes the fraction of samples classified as `target_digit`.

    Args:
        classifier: Trained MNISTClassifier.
        samples: Generated images, shape (N, 1, H, W).
        target_digit: Expected digit label in [0, 9].

    Returns:
        Fraction of samples predicted as target_digit, in [0, 1].
    """
    raise NotImplementedError


def label_distribution(
    classifier: torch.nn.Module, samples: torch.Tensor
) -> torch.Tensor:
    """Computes the histogram of predicted labels over `samples`.

    Args:
        classifier: Trained MNISTClassifier.
        samples: Generated images, shape (N, 1, H, W).

    Returns:
        Tensor of shape (10,) with the normalized predicted-label
        distribution.
    """
    raise NotImplementedError


def label_entropy(distribution: torch.Tensor) -> float:
    """Computes the Shannon entropy of a predicted-label distribution.

    Args:
        distribution: Tensor of shape (10,), as returned by
            `label_distribution`.

    Returns:
        Entropy in nats; log(10) is the maximum (fully uniform).
    """
    raise NotImplementedError
