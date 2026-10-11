# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 12:16:12 2026

@author: Alberto Suárez <alberto.suarez@uam.es>
"""
from typing import Optional, Sequence, Union, Callable

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def gmm_pdf(
    x: NDArray[np.float64],
    weights: NDArray[np.float64],
    means: NDArray[np.float64],
    covariances: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate the GMM density p(x) = sum_k pi_k * N(x; mu_k, Sigma_k).

    Args:
        x: Points at which to evaluate the density, shape (..., D).
        weights: Mixing coefficients, shape (K,).
        means: Component means, shape (K, D).
        covariances: Component covariances, shape (K, D, D).

    Returns:
        Array of density values with shape equal to `x.shape[:-1]`.

    Examples:
        >>> import numpy as np
        >>> weights = np.array([0.3, 0.7])
        >>> means = np.array([[0.0, 0.0], [5.0, 5.0]])
        >>> covariances = np.array([
        ...     [[1.0, 0.0], [0.0, 1.0]],
        ...     [[1.5, 0.2], [0.2, 1.5]],
        ... ])

        >>> # Evaluate at a single point, shape (D,) -> scalar output.
        >>> x_single = np.array([0.0, 0.0])
        >>> gmm_pdf(x_single, weights, means, covariances)  # doctest: +SKIP
        0.04773...

        >>> # Evaluate at a batch of points, shape (N, D) -> shape (N,) output.
        >>> x_batch = np.array([[0.0, 0.0], [5.0, 5.0], [2.5, 2.5]])
        >>> gmm_pdf(x_batch, weights, means, covariances).shape
        (3,)

        >>> # Evaluate on a 2D grid, shape (H, W, D) -> shape (H, W) output.
        >>> xx, yy = np.meshgrid(np.linspace(-3, 8, 50), np.linspace(-3, 8, 50))
        >>> grid_points = np.stack([xx, yy], axis=-1)  # (50, 50, 2)
        >>> gmm_pdf(grid_points, weights, means, covariances).shape
        (50, 50)
    """
    n_components, n_dims = means.shape
    density = np.zeros(x.shape[:-1], dtype=np.float64)

    for k in range(n_components):
        diff = x - means[k]  # (..., D)
        cov_inv = np.linalg.inv(covariances[k])
        cov_det = np.linalg.det(covariances[k])
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** n_dims * cov_det)
        exponent = -0.5 * np.einsum("...i,ij,...j->...", diff, cov_inv, diff)
        density += weights[k] * norm_const * np.exp(exponent)

    return density

def gmm_score(
    x: NDArray[np.float64],
    weights: NDArray[np.float64],
    means: NDArray[np.float64],
    covariances: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the score function s(x) = grad_x log p(x) of a GMM.

    s(x) = -sum_k gamma_k(x) * Sigma_k^{-1} (x - mu_k),
    where gamma_k(x) = pi_k N_k(x) / sum_j pi_j N_j(x).

    Args:
        x: Point(s) at which to evaluate the score, shape (..., D).
        weights: Mixing coefficients, shape (K,).
        means: Component means, shape (K, D).
        covariances: Component covariances, shape (K, D, D).

    Returns:
        Score vector(s) with the same shape as `x`.

    Examples:
        >>> import numpy as np
        >>> weights = np.array([0.3, 0.7])
        >>> means = np.array([[0.0, 0.0], [5.0, 5.0]])
        >>> covariances = np.array([
        ...     [[1.0, 0.0], [0.0, 1.0]],
        ...     [[1.5, 0.2], [0.2, 1.5]],
        ... ])

        >>> # Score at a single point, shape (D,) -> shape (D,) output.
        >>> gmm_score(np.array([0.0, 0.0]), weights, means, covariances).shape
        (2,)

        >>> # Score at a batch of points, shape (N, D) -> shape (N, D) output.
        >>> x_batch = np.array([[0.0, 0.0], [5.0, 5.0], [2.5, 2.5]])
        >>> gmm_score(x_batch, weights, means, covariances).shape
        (3, 2)
    """
    n_components, n_dims = means.shape

    x_centered = x[..., None, :] - means  # (..., K, D)
    cov_inv = np.linalg.inv(covariances)  # (K, D, D)
    cov_det = np.linalg.det(covariances)  # (K,)

    mahalanobis_distance = np.einsum(
        "...ki,kij,...kj->...k", x_centered, cov_inv, x_centered
    )  # (..., K)
    
    # [TO COMPLETE]
    
    
    grad_per_component = -np.einsum("kij,...kj->...ki", cov_inv, x_centered) # (..., K, D)

    densities = np.exp(-0.5 * mahalanobis_distance) / np.sqrt((2 * np.pi) ** n_dims * cov_det)
    weighted_densities = weights * densities
    responsibilities = weighted_densities / weighted_densities.sum(axis=-1, keepdims=True)

    score = np.einsum(
        "...k,...ki->...i", responsibilities, grad_per_component
        )  # (..., D)

    return score

def gmm_sample(
    n_samples: int,
    weights: Sequence[float] | NDArray[np.float64],
    means: NDArray[np.float64],
    covariances: NDArray[np.float64],
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> NDArray[np.float64]:
    """Generate samples from a mixture of K  Gaussians in D dimensions.

    Draws `n_samples` points from a Gaussian Mixture Model (GMM) defined by
    K components, each with its own mixing weight, mean vector, and
    covariance matrix. Sampling proceeds in three steps:

    1. Draw N component assignments from a Categorical distribution with
       parameter vector `weights` (i.e. `z_n ~ Categorical(weights)`).
    2. Draw N standard normal vectors `eps_n ~ N(0, I_D)`.
    3. For each sample, transform the standard normal draw using the
       Cholesky factor `L_k` of the assigned component's covariance
       `Sigma_k = L_k L_k^T`, via `x_n = mu_{z_n} + L_{z_n} @ eps_n`.

    Args:
        n_samples: Number of samples (N) to generate.
        weights: Mixing coefficients for the K components, shape (K,).
            Must be non-negative and sum to 1.
        means: Mean vectors for the K components, shape (K, D), where D is
            the dimensionality of the data.
        covariances: Covariance matrices for the K components, shape
            (K, D, D). Each (D, D) slice must be symmetric positive
            definite (required for the Cholesky decomposition to exist).
        random_state: Seed or `numpy.random.Generator` instance for
            reproducibility. If None, a fresh, unseeded generator is used.

    Returns:
        An array of shape (N, D) containing the generated samples.

    Raises:
        ValueError: If `weights`, `means`, and `covariances` have
            inconsistent shapes, or if `weights` are negative or do not sum to 1.
        numpy.linalg.LinAlgError: If any covariance matrix is not
            symmetric positive definite (Cholesky decomposition fails).

    Examples:
        >>> import numpy as np
        >>> weights = [0.3, 0.7]
        >>> means = np.array([[0.0, 0.0], [5.0, 5.0]])
        >>> covariances = np.array([
        ...     [[1.0, 0.0], [0.0, 1.0]],
        ...     [[1.5, 0.2], [0.2, 1.5]],
        ... ])
        >>> samples = sample_gaussian_mixture(
        ...     n_samples=1000,
        ...     weights=weights,
        ...     means=means,
        ...     covariances=covariances,
        ...     random_state=42,
        ... )
        >>> samples.shape
        (1000, 2)

        >>> # Single-component "mixture" (reduces to a single normal)
        >>> samples = gmm_sample(
        ...     n_samples=500,
        ...     weights=[1.0],
        ...     means=np.zeros((1, 3)),
        ...     covariances=np.eye(3)[None, :, :],
        ...     random_state=0,
        ... )
        >>> samples.shape
        (500, 3)
    """
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    weights = np.asarray(weights, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    covariances = np.asarray(covariances, dtype=np.float64)

    n_components, n_dims = means.shape

    if weights.shape != (n_components,):
        raise ValueError(
            f"`weights` has shape {weights.shape}, expected ({n_components},) "
            "to match the number of components inferred from `means`."
        )
    if np.any(weights < 0):
        raise ValueError(
            f"`weights` must be non-negative, got weights={weights}."
        )
        
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"`weights` must sum to 1, got sum={weights.sum()}.")

    if covariances.shape != (n_components, n_dims, n_dims):
        raise ValueError(
            f"`covariances` has shape {covariances.shape}, expected "
            f"({n_components}, {n_dims}, {n_dims})."
        )

    # Step 1: N draws from Categorical(weights) 

    component_ids = rng.choice(n_components, size = n_samples, p = weights)

    # Step 2: N draws from a standard normal in D dimensions.

    eps = rng.standard_normal((n_samples,n_dims))

    # Step 3: Cholesky factor per component, then affine transform per sample.
    cholesky_factors = np.linalg.cholesky(covariances)  # shape (K, D, D)

    L_n = cholesky_factors[component_ids]  # shape (N, D, D)
    mu_n = means[component_ids]  # shape (N, D)

    # x_n = mu_{z_n} + L_{z_n} @ eps_n, done as a batched matmul.
    samples = mu_n + np.einsum("nij,nj->ni", L_n, eps)

    return samples

def gmm_plot_sample_with_density(
    samples: NDArray[np.float64],
    weights: NDArray[np.float64],
    means: NDArray[np.float64],
    covariances: NDArray[np.float64],
    pad: float = 3.0,
    grid_res: int = 300,
    n_levels: int = 8,
    figsize: tuple[float, float] = (8.0, 7.0),
    ax: Optional[Axes] = None,
) -> tuple[Figure, Axes]:
    """Plot a 2D sample overlaid with the GMM density as contours and a heatmap.

    Draws a scatterplot of `samples`, and on the same axes overlays a
    heatmap of the mixture density's height together with contours of
    equal probability density.

    Args:
        samples: iid sample from the GMM, shape (N, 2).
        weights: Mixing coefficients, shape (K,).
        means: Component means, shape (K, 2).
        covariances: Component covariances, shape (K, 2, 2).
        pad: Margin added around the sample's bounding box when building
            the evaluation grid.
        grid_res: Number of grid points per axis used to evaluate the pdf.
        n_levels: Number of contour levels to draw.
        figsize: Figure size, used only if `ax` is None.
        ax: Optional existing Axes to draw on. If None, a new figure and
            axes are created.

    Returns:
        A tuple `(fig, ax)` of the Matplotlib Figure and Axes used.

    Examples:
        >>> import numpy as np
        >>> weights = np.array([0.3, 0.4, 0.3])
        >>> means = np.array([[0.0, 0.0], [5.0, 4.0], [-4.0, 5.0]])
        >>> covariances = np.array([
        ...     [[1.0, 0.0], [0.0, 1.0]],
        ...     [[1.5, 0.6], [0.6, 0.8]],
        ...     [[0.8, -0.4], [-0.4, 1.2]],
        ... ])
        >>> samples = sample_gaussian_mixture(
        ...     n_samples=100, weights=weights, means=means,
        ...     covariances=covariances, random_state=42,
        ... )
        >>> fig, ax = plot_gmm_sample_with_density(samples, weights, means, covariances)
        >>> fig.savefig("gmm_scatter_contours.png")  # doctest: +SKIP
    """
    x_min, x_max = samples[:, 0].min() - pad, samples[:, 0].max() + pad
    y_min, y_max = samples[:, 1].min() - pad, samples[:, 1].max() + pad

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
    )
    grid_points = np.stack([xx, yy], axis=-1)  # (grid_res, grid_res, 2)
    zz = gmm_pdf(grid_points, weights, means, covariances)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Heatmap of the pdf height.
    heatmap = ax.pcolormesh(xx, yy, zz, shading="auto", cmap="viridis")
    fig.colorbar(heatmap, ax=ax, label=r"$p(\mathbf{x})$")

    # Contours of equal probability density.
    contours = ax.contour(xx, yy, zz, levels=n_levels, colors="white", linewidths=0.8)
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.1e")

    # Scatterplot of the iid sample.
    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        s=25,
        facecolors="white",
        edgecolors="black",
        linewidths=0.6,
        label=f"Sample ($N={samples.shape[0]}$)",
        zorder=3,
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("GMM sample with density contours and heatmap")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")

    fig.tight_layout()

    return fig, ax

from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray


from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray


def sample_langevin(
    n_samples: int,
    score_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x0: NDArray[np.float64],
    n_steps: int = 1000,
    step_size: float = 0.05,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> NDArray[np.float64]:
    """Sample approximately from a distribution using the Unadjusted Langevin Algorithm.

    Runs `n_samples` independent Langevin chains in parallel, all
    initialized at `x0`, and updated for `n_steps` iterations via
        x <- x + (eta / 2) * score(x) + sqrt(eta) * eps,   eps ~ N(0, I_D)
    returning the final state of each chain.

    Args:
        n_samples: Number of parallel chains / samples (N).
        score: Callable computing the score function s(x) = grad_x log p(x).
            Must accept an array of shape (N, D) and return an array of
            the same shape (N, D), i.e. it must be vectorized over chains.
        x0: Initial point for the chains. Either shape (D,), in which case
            all N chains start at the same point `x0` (D is inferred from
            `x0.shape[0]`), or shape (N, D) to give each chain its own
            starting point directly.
        n_steps: Number of Langevin steps (T) per chain.
        step_size: Langevin step size (eta).
        random_state: Seed or `numpy.random.Generator` for reproducibility.

    Returns:
        Array of shape (N, D) with the (approximate) samples.

    Examples:
        >>> import numpy as np
        >>> weights = np.array([0.3, 0.4, 0.3])
        >>> means = np.array([[0.0, 0.0], [5.0, 4.0], [-4.0, 5.0]])
        >>> covariances = np.array([
        ...     [[1.0, 0.0], [0.0, 1.0]],
        ...     [[1.5, 0.6], [0.6, 0.8]],
        ...     [[0.8, -0.4], [-0.4, 1.2]],
        ... ])
        >>> score_fn = lambda x: gmm_score(x, weights, means, covariances)
        >>> samples = sample_langevin(
        ...     n_samples=100,
        ...     score_fn=score_fn,
        ...     x0=np.zeros(2),
        ...     n_steps=2000,
        ...     step_size=0.05,
        ...     random_state=0,
        ... )
        >>> samples.shape
        (100, 2)
    """
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    x0 = np.asarray(x0, dtype=np.float64)

    if x0.ndim == 1:
        # Same starting point for every chain: broadcast to (N, D).
        x = np.tile(x0, (n_samples, 1))
    elif x0.ndim == 2:
        if x0.shape[0] != n_samples:
            raise ValueError(
                f"`x0` has shape {x0.shape}, expected first dimension "
                f"{n_samples} (n_samples) when given per-chain starting points."
            )
        x = x0.copy()
    else:
        raise ValueError(f"`x0` must have shape (D,) or (N, D), got shape {x0.shape}.")

    # [TO COMPLETE]

    return x