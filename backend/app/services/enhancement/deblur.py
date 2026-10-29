import cv2
import numpy as np


def sharpen(image: np.ndarray, amount: float = 0.45, sigma: float = 1.1) -> np.ndarray:
    """Apply mild unsharp masking to reduce perceived blur.

    The *amount* parameter controls sharpening strength.  The default 0.45 is
    intentionally conservative to avoid halo artefacts on handwritten content.
    A Gaussian blur radius (*sigma*) of 1.1 px targets fine strokes without
    amplifying large-scale tonal gradients.

    The output is clamped to [0, 255] to prevent overflow artefacts.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    # Clamp explicitly; addWeighted with uint8 already saturates, but be safe.
    return np.clip(sharpened, 0, 255).astype(np.uint8)
