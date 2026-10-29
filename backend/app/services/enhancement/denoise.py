import cv2
import numpy as np


def denoise(image: np.ndarray, strength: int = 7) -> np.ndarray:
    """Apply fast non-local means denoising.

    Uses the coloured variant for 3-channel BGR images and the greyscale
    variant for single-channel images, with the same luminance strength.
    The template window (7 px) and search window (21 px) are standard defaults
    that work well on photographic document noise without blurring fine strokes.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    gray = image if image.ndim == 2 else image[:, :, 0]
    return cv2.fastNlMeansDenoising(gray, None, strength, 7, 21)
