import cv2
import numpy as np


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to the L channel of a LAB-converted image.

    Works on both 3-channel BGR images and single-channel greyscale images.
    For greyscale input the CLAHE is applied directly to the single channel and
    the result is returned as greyscale.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if image.ndim == 2 or image.shape[2] == 1:
        gray = image if image.ndim == 2 else image[:, :, 0]
        return clahe.apply(gray)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    enhanced_lightness = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((enhanced_lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
