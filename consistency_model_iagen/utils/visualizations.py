#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/visualizations.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    This module provides a utility function to visualize the comparison between
    initial and final images. It automatically detects whether to use grayscale
    based on the number of channels.

License:
    Proprietary. All rights reserved.
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_image_comparison(
    images: torch.Tensor,
    n_images: int = 3,
    figsize: Tuple[int, int] = (15, 4)
) -> None:
    """Plots the initial and final images for comparison.

    Args:
        images: Tensor of shape (N, C, H, W, 2), where N is the number
            of images, C is the number of channels, H and W are height
            and width, and the last dimension contains [initial, final].
        n_images: Number of images to plot (rows).
        figsize: Size of the matplotlib figure.

    Returns:
        None. Displays the plot.
    """
    # Creating subplots with 2 columns (initial and final)
    _, axes = plt.subplots(
        n_images, 2, figsize=figsize
    )

    # Column titles
    titles = ["Initial", "Final"]
    
    # Determine if grayscale based on number of channels
    num_channels = images.shape[1]
    is_grayscale = (num_channels == 1)

    # Looping through each image
    for i in range(n_images):
        for j in range(2):  # Always 2 columns: initial and final
            if is_grayscale:
                # For grayscale, use only first channel
                img = images[i, 0, :, :, j].cpu().numpy()
                normalized_img = (img + 1) / 2
                
                ax = axes[i][j] if n_images > 1 else axes[j]
                ax.imshow(normalized_img, cmap="gray")
            else:
                # For color images, use all channels
                img = images[i, :, :, :, j].cpu().numpy()
                
                # Transpose to (H, W, C) format for proper display
                img = np.transpose(img, (1, 2, 0))

                # Normalize the image from [-1, 1] to [0, 1]
                img = (img + 1) / 2
                
                # Ensure the image is in range [0, 1] to avoid display issues
                img = np.clip(img, 0, 1)

                ax = axes[i][j] if n_images > 1 else axes[j]
                ax.imshow(img)
            
            ax.axis("off")
            
            # Set titles for columns on the first row only
            if i == 0:
                ax.set_title(titles[j])
            
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt

# def plot_image_evolution(images, n_images=3, n_intermediate_steps=None, figsize=(15, 4)):
#     if n_intermediate_steps is None:
#         n_intermediate_steps = [0, -1]

#     fig, axes = plt.subplots(n_images, len(n_intermediate_steps), figsize=figsize)

#     for i in range(n_images):
#         for j, step in enumerate(n_intermediate_steps):
#             img = images[i, 0, :, :, step].cpu().numpy()
#             ax = axes[i][j] if n_images > 1 else axes[j]
#             ax.imshow((img + 1) / 2, cmap='gray')
#             ax.axis("off")
#             if i == 0:
#                 ax.set_title(f"Step {step}")
#     plt.tight_layout()
#     plt.show()

def plot_image_evolution(images, n_images=3, n_intermediate_steps=None, figsize=(15, 4)):
    if n_intermediate_steps is None:
        n_intermediate_steps = [0, -1]

    fig, axes = plt.subplots(n_images, len(n_intermediate_steps), figsize=figsize)

    for i in range(n_images):
        for j, step in enumerate(n_intermediate_steps):
            img = images[i, :, :, :, step].cpu().numpy()

            if img.shape[0] == 1:
                # Imagen en escala de grises
                img = img[0]  # (H, W)
                ax = axes[i][j] if n_images > 1 else axes[j]
                ax.imshow((img + 1) / 2, cmap='gray')
            else:
                # Imagen RGB
                img = img.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                ax = axes[i][j] if n_images > 1 else axes[j]
                ax.imshow((img + 1) / 2)

            ax.axis("off")
            if i == 0:
                ax.set_title(f"Step {step}")

    plt.tight_layout()
    plt.show()
