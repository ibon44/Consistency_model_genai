#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/metrics.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Quality and diversity metrics computed over a batch of generated
    samples using the reference classifier (`evaluation/classifier.py`),
    as required by objectives 3-4 of the project proposal:

    - predict_proba: classifier probabilities for every sample; the
      building block the other metrics share.
    - confident_fraction: fraction of samples the classifier recognizes
      with high confidence, the quality metric for an unconditional
      model (all digits).
    - classification_accuracy: fraction of samples classified as the
      *conditioning* digit, when sampling from a digit-restricted model
      (see the DIGIT config knob in the training scripts).
    - label_distribution: histogram of predicted labels, used as a
      cheap proxy for mode coverage/diversity of an unconditional model.
    - label_entropy: Shannon entropy of `label_distribution`, a single
      scalar diversity summary (higher = more uniform across digits).
    - fd_mnist: Fréchet distance between the classifier's penultimate
      features of generated and real images (FID with this classifier
      instead of Inception). The main quality metric: unlike
      `confident_fraction`, it is not fooled by out-of-distribution
      inputs (the classifier labels ~60% of pure noise as a digit with
      > 0.9 confidence, but noise gets a far larger FD than any
      generator), and it also captures within-class diversity.

    All functions expect samples in [-1, 1], the range the classifier
    was trained on (`data/datasets.py`). Samplers returning [0, 1]
    (e.g. `sampling/probability_flow.py`) must be rescaled first with
    x * 2 - 1.

License:
    Proprietary. All rights reserved.
"""

import math

import numpy as np
import torch
from scipy import linalg


@torch.no_grad()
def predict_proba(
    classifier: torch.nn.Module,
    samples: torch.Tensor,
    batch_size: int = 500,
) -> torch.Tensor:
    """Computes the classifier's class probabilities for each sample.

    Runs in eval mode and without gradients, in chunks of `batch_size`
    so that large sample sets do not have to fit in memory at once.
    The samples are moved to the classifier's device, and the softmax
    turns the classifier's logits into probabilities that sum to 1.

    Args:
        classifier: Trained MNISTClassifier (returns logits).
        samples: Images in [-1, 1], shape (N, 1, H, W).
        batch_size: Number of samples classified at a time.

    Returns:
        Tensor of shape (N, 10) with one probability per digit, on CPU.
    """
    classifier.eval()
    device = next(classifier.parameters()).device
    probs = []
    for chunk in samples.split(batch_size):
        logits = classifier(chunk.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(probs)


def confident_fraction(
    classifier: torch.nn.Module,
    samples: torch.Tensor,
    threshold: float = 0.9,
) -> float:
    """Computes the fraction of samples recognized with high confidence.

    A sample counts as "recognizable" when the classifier assigns at
    least `threshold` probability to its most likely digit. Blurry or
    ambiguous samples spread their probability over several digits and
    fall below the threshold. Real MNIST test images score close to
    1.0, which is the reference value for a perfect generator.

    Args:
        classifier: Trained MNISTClassifier.
        samples: Generated images in [-1, 1], shape (N, 1, H, W).
        threshold: Minimum top-1 probability to count a sample.

    Returns:
        Fraction of samples whose top-1 probability is >= threshold,
        in [0, 1].
    """
    confidence = predict_proba(classifier, samples).max(dim=1).values
    return (confidence >= threshold).float().mean().item()


def classification_accuracy(
    classifier: torch.nn.Module,
    samples: torch.Tensor,
    target_digit: int,
) -> float:
    """Computes the fraction of samples classified as `target_digit`.

    Only meaningful for a model trained on a single digit (--digit),
    where every sample is supposed to be that digit. For a model that
    generates all digits use `confident_fraction` instead.

    Args:
        classifier: Trained MNISTClassifier.
        samples: Generated images in [-1, 1], shape (N, 1, H, W).
        target_digit: Expected digit label in [0, 9].

    Returns:
        Fraction of samples predicted as target_digit, in [0, 1].
    """
    predicted = predict_proba(classifier, samples).argmax(dim=1)
    return (predicted == target_digit).float().mean().item()


def label_distribution(
    classifier: torch.nn.Module, samples: torch.Tensor
) -> torch.Tensor:
    """Computes the histogram of predicted labels over `samples`.

    Each sample is assigned its most likely digit (argmax), the digits
    are counted with bincount and the counts are divided by the number
    of samples, so the result sums to 1. A perfectly diverse generator
    gives roughly 0.1 for every digit.

    Args:
        classifier: Trained MNISTClassifier.
        samples: Generated images in [-1, 1], shape (N, 1, H, W).

    Returns:
        Tensor of shape (10,) with the normalized predicted-label
        distribution.
    """
    predicted = predict_proba(classifier, samples).argmax(dim=1)
    counts = torch.bincount(predicted, minlength=10).float()
    return counts / counts.sum()


def label_entropy(distribution: torch.Tensor) -> float:
    """Computes the Shannon entropy of a predicted-label distribution.

    H = -sum_i p_i * ln(p_i), skipping digits with p_i = 0 (their term
    is 0 by convention, and ln(0) would give nan). The maximum, ln(10)
    ≈ 2.303, is reached when all 10 digits are equally frequent; it
    drops towards 0 as the generator collapses onto fewer digits.

    Args:
        distribution: Tensor of shape (10,), as returned by
            `label_distribution`.

    Returns:
        Entropy in nats; log(10) is the maximum (fully uniform).
    """
    p = distribution[distribution > 0]
    return -(p * p.log()).sum().item()


def max_label_entropy(n_classes: int = 10) -> float:
    """Returns the maximum possible entropy, ln(n_classes), in nats.

    Used to report the entropy as a percentage of its maximum.
    """
    return math.log(n_classes)


@torch.no_grad()
def extract_features(
    classifier: torch.nn.Module,
    samples: torch.Tensor,
    batch_size: int = 500,
) -> np.ndarray:
    """Computes the classifier's penultimate features for each sample.

    Same chunked, gradient-free evaluation as `predict_proba`, but
    calling `classifier.embed` (128 features per image) instead of the
    logits. Returned as float64 NumPy so the covariance and matrix
    square root in `frechet_distance` are numerically stable.

    Args:
        classifier: Trained MNISTClassifier (must provide `embed`).
        samples: Images in [-1, 1], shape (N, 1, H, W).
        batch_size: Number of samples processed at a time.

    Returns:
        Array of shape (N, 128).
    """
    classifier.eval()
    device = next(classifier.parameters()).device
    feats = [classifier.embed(chunk.to(device)).cpu() for chunk in samples.split(batch_size)]
    return torch.cat(feats).numpy().astype(np.float64)


def frechet_distance(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    """Computes the Fréchet distance between two sets of feature vectors.

    Fits a Gaussian (mean μ, covariance Σ) to each set and returns

        FD = ||μ_a − μ_b||² + Tr(Σ_a + Σ_b − 2·(Σ_a Σ_b)^½)

    The first term compares where the two clouds of features are
    centred; the trace term compares their spread and shape. FD = 0
    only for identical Gaussians; lower is better.

    Only the trace of (Σ_a Σ_b)^½ is needed, and it equals the trace of
    (Σ_a^½ Σ_b Σ_a^½)^½, whose argument is symmetric positive
    semi-definite. That form is computed with eigenvalue decompositions
    (eigh / eigvalsh), clipping tiny negative eigenvalues from rounding
    error to 0. This stays stable when a covariance is singular, which
    happens here because some of the classifier's 128 ReLU features are
    always 0 ("dead" units); a generic matrix square root
    (scipy.linalg.sqrtm) warns and may lose accuracy in that case.

    Args:
        feats_a: Array of shape (N_a, D).
        feats_b: Array of shape (N_b, D).

    Returns:
        The Fréchet distance (a non-negative float).
    """
    mu_a, mu_b = feats_a.mean(axis=0), feats_b.mean(axis=0)
    sigma_a = np.cov(feats_a, rowvar=False)
    sigma_b = np.cov(feats_b, rowvar=False)

    eigval_a, eigvec_a = linalg.eigh(sigma_a)
    sqrt_sigma_a = (eigvec_a * np.sqrt(np.clip(eigval_a, 0.0, None))) @ eigvec_a.T
    inner = sqrt_sigma_a @ sigma_b @ sqrt_sigma_a
    trace_covmean = np.sqrt(np.clip(linalg.eigvalsh(inner), 0.0, None)).sum()

    diff = mu_a - mu_b
    fd = diff @ diff + np.trace(sigma_a) + np.trace(sigma_b) - 2.0 * trace_covmean
    return max(0.0, float(fd))


def fd_mnist(
    classifier: torch.nn.Module,
    samples: torch.Tensor,
    reference_features: np.ndarray,
) -> float:
    """Computes FD-MNIST: Fréchet distance to real MNIST in classifier space.

    `reference_features` should be computed once with
    `extract_features` on a fixed set of real images (e.g. 10,000
    MNIST training images) and reused for every model, so all models
    are compared against the same reference.

    FD is biased by the number of samples (fewer samples give a larger
    value even for a perfect generator), so every model must be
    evaluated with the same N, and the FD of real held-out images with
    that same N is the "perfect generator" baseline to report next to
    the models (≈ 45 with N = 2000 against 10,000 training images).

    Args:
        classifier: Trained MNISTClassifier.
        samples: Generated images in [-1, 1], shape (N, 1, H, W).
        reference_features: Features of real images, shape (M, 128).

    Returns:
        FD-MNIST value; lower is better.
    """
    return frechet_distance(extract_features(classifier, samples), reference_features)
