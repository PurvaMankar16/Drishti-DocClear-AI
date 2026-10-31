import cv2
import numpy as np


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 3.5,
    tile_grid_size: tuple[int, int] = (16, 16),
) -> np.ndarray:
    """Apply document-optimized contrast enhancement.

    Uses CLAHE with parameters tuned for document images:
    - Higher clip limit (3.5 vs 2.0) for stronger local contrast
    - Larger tiles (16x16 vs 8x8) to respect document structure
    - Works on LAB L-channel for color images, direct on grayscale

    Args:
        image: BGR or grayscale image
        clip_limit: CLAHE clip limit (default 3.5 for documents)
        tile_grid_size: CLAHE tile size (default 16x16 for documents)

    Returns:
        Contrast-enhanced image
    """
    if image.size == 0:
        return image.copy()

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        # Grayscale
        gray = image if image.ndim == 2 else image[:, :, 0]
        enhanced = clahe.apply(gray)
        return enhanced if image.ndim == 2 else enhanced[:, :, np.newaxis]

    # Color image: apply to LAB L-channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    enhanced_lightness = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((enhanced_lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def enhance_contrast_bilateral(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Alternative: bilateral filtering for edge-preserving smoothing.

    Reduces noise while preserving text edges. Can be used instead of or
    in addition to CLAHE for documents with noise.

    Args:
        image: BGR or grayscale image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space

    Returns:
        Smoothed image with preserved edges
    """
    if image.size == 0:
        return image.copy()

    if image.ndim == 2:
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    # Apply to each channel
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
