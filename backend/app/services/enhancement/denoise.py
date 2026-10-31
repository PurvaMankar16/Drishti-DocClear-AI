import cv2
import numpy as np


def denoise(image: np.ndarray, strength: int = 10, adaptive: bool = True) -> np.ndarray:
    """Apply adaptive fast non-local means denoising.

    Increased default strength (10 vs 7) for better noise reduction while
    preserving document details.

    Args:
        image: BGR or grayscale image
        strength: Denoising strength (7-15 typical for documents)
        adaptive: If True, adjust strength based on detected noise level

    Returns:
        Denoised image
    """
    if image.size == 0:
        return image.copy()

    # Adaptive strength selection
    if adaptive:
        # Measure noise level
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        smooth = cv2.GaussianBlur(gray, (3, 3), 0)
        noise_level = float(np.std(gray.astype(np.float32) - smooth.astype(np.float32)))

        # Adjust strength based on noise level
        if noise_level < 3:  # Very low noise
            strength = max(strength // 2, 5)
        elif noise_level < 5:  # Low noise
            strength = max(int(strength * 0.7), 7)
        elif noise_level > 12:  # High noise
            strength = min(int(strength * 1.3), 15)

    # Apply denoising
    if image.ndim == 3 and image.shape[2] == 3:
        # Color image: use colored variant
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    else:
        # Grayscale
        gray = image if image.ndim == 2 else image[:, :, 0]
        denoised = cv2.fastNlMeansDenoising(gray, None, strength, 7, 21)
        return denoised if image.ndim == 2 else denoised[:, :, np.newaxis]


def denoise_bilateral(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Alternative: bilateral filtering for gentle denoising.

    Faster than NLM denoising, preserves edges well. Good for light noise.

    Args:
        image: BGR or grayscale image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space

    Returns:
        Denoised image with preserved edges
    """
    if image.size == 0:
        return image.copy()

    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
