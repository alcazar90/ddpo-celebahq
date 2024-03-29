"""Utility functions for the ddpo package."""

import gc

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt


def flush() -> None:
    """Flush the memory."""
    gc.collect()
    torch.cuda.empty_cache()


def decode_tensor_to_img(x: torch.Tensor, num_rows_per_grid: int = 5) -> None:
    """Decode a tensor into a plt.imshow."""
    grid = torchvision.utils.make_grid(x, nrow=num_rows_per_grid).permute(1, 2, 0)
    plt.imshow(grid.cpu().clip(-1, 1) * 0.5 + 0.5)
    plt.axis("off")
    plt.show()


def decode_tensor_to_np_img(
    x: torch.Tensor,
    melt_batch: bool = True,
    num_rows_per_grid: int = 1,
) -> np.ndarray:
    """Decode pytorch tensor batches or single images to numpy array images.

    Useful for converting the output of this function to Image.fromarray (melt_batch = True).
    """
    check_tensor_dim = 4
    if x.ndim == check_tensor_dim and melt_batch:
        # process the batch of images as a single one
        x = torchvision.utils.make_grid(x, nrow=num_rows_per_grid)
    if melt_batch:
        images = (
            ((x.detach().cpu().clip(-1, 1) * 0.5 + 0.5) * 255)
            .permute(1, 2, 0)
            .numpy()
            .astype("uint8")
        )
    else:
        images = (
            ((x.detach().cpu().clip(-1, 1) * 0.5 + 0.5) * 255)
            .permute(0, 2, 3, 1)
            .numpy()
            .astype("uint8")
        )
    return images
