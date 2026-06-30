"""Small image helpers shared by the inference backends and scorers.

Chart images are frequently exported with transparency. Before feeding them to
an image model or an OCR engine we flatten the alpha channel onto a solid white
background (so transparent regions read as white rather than black), or onto a
checkerboard when we want transparency to remain visible to a vision LLM.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _has_alpha(image_array: np.ndarray) -> bool:
    return image_array.shape[2] == 4


def alpha_over_white(image: Image.Image) -> Image.Image:
    """Composite an RGBA image over a white background, returning RGB."""
    if image.mode == "RGB":
        return image
    image = image.convert("RGBA")
    arr = np.array(image) / 255.0
    blended = _alpha_over_white_array(arr)
    return Image.fromarray((blended * 255).astype(np.uint8))


def alpha_over_checker(image: Image.Image) -> Image.Image:
    """Composite an RGBA image over a checkerboard (keeps transparency visible)."""
    if image.mode == "RGB":
        return image
    image = image.convert("RGBA")
    arr = np.array(image) / 255.0
    blended = _alpha_over_checker_array(arr)
    return Image.fromarray((blended * 255).astype(np.uint8))


def _alpha_over_white_array(image_array: np.ndarray) -> np.ndarray:
    if not _has_alpha(image_array):
        return image_array
    rgb = image_array[:, :, :3]
    alpha = image_array[:, :, 3][:, :, None]
    white = 1.0
    return (1 - alpha) * white + alpha * rgb


def _alpha_over_checker_array(image_array: np.ndarray) -> np.ndarray:
    if not _has_alpha(image_array):
        return image_array
    rgb = image_array[:, :, :3]
    alpha = image_array[:, :, 3][:, :, None]
    checker = (np.indices(rgb.shape[:3]).sum(axis=0) % 2)
    return (1 - alpha) * checker + alpha * rgb
