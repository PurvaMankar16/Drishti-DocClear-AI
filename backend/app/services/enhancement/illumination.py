import cv2
import numpy as np


def normalize_background(image: np.ndarray, target_background: float = 240.0) -> np.ndarray:
    """Normalize document background to white while preserving foreground text.

    Uses Otsu thresholding to separate text from background, then brightens
    the background without affecting text darkness. This preserves sharpness
    and text contrast while creating a clean white background.

    Args:
        image: BGR or grayscale image
        target_background: Target brightness for background (default 240 for near-white)

    Returns:
        Image with normalized background
    """
    if image.size == 0:
        return image.copy()

    # Convert to grayscale for analysis
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Check if background is already bright enough
    # Use 90th percentile to avoid text affecting background estimate
    background_estimate = float(np.percentile(gray, 90))
    if background_estimate >= target_background - 10:
        # Background already white enough
        return image.copy()

    # Use Otsu thresholding to separate foreground (text) from background
    # Blur slightly to reduce noise in threshold calculation
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # binary=255 is background, binary=0 is text/foreground
    # Create smooth mask to avoid hard edges
    mask = cv2.GaussianBlur(binary, (11, 11), 0).astype(np.float32) / 255.0

    # Calculate gain for background pixels
    # Use median of bright pixels as current background level
    background_pixels = gray[binary == 255]
    if len(background_pixels) < 100:
        # Not enough background detected, apply conservative global brightening
        return correct_illumination_conservative(image)

    current_background = float(np.median(background_pixels))
    if current_background < 50:
        # Background too dark, might not be a document
        return correct_illumination_conservative(image)

    # Calculate gain to bring background to target
    gain = target_background / current_background
    # Cap gain to avoid overexposure
    gain = min(gain, 1.5)

    if gain <= 1.05:
        # Minimal improvement needed
        return image.copy()

    # Apply gain only to background regions
    if image.ndim == 3:
        # Color image: apply to all channels
        result = image.astype(np.float32)
        for c in range(3):
            # Blend original (text) with brightened (background)
            brightened = np.clip(result[:, :, c] * gain, 0, 255)
            result[:, :, c] = mask * brightened + (1 - mask) * result[:, :, c]
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        # Grayscale
        result = gray.astype(np.float32)
        brightened = np.clip(result * gain, 0, 255)
        result = mask * brightened + (1 - mask) * result
        return np.clip(result, 0, 255).astype(np.uint8)


def correct_illumination_conservative(image: np.ndarray, target_brightness: float = 180.0) -> np.ndarray:
    """Fallback: conservative global brightness adjustment.

    Used when background detection fails. Targets brighter level (180 vs 128)
    and caps gain more aggressively to preserve sharpness.
    """
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        gray = image if image.ndim == 2 else image[:, :, 0]
        mean_brightness = float(gray.mean())
        if mean_brightness <= 1 or mean_brightness >= target_brightness - 20:
            return image.copy()
        gain = float(np.clip(target_brightness / mean_brightness, 0.85, 1.15))
        corrected = np.clip(gray.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return corrected if image.ndim == 2 else corrected[:, :, np.newaxis]

    # Color image: work in HSV space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    mean_brightness = float(value.mean())
    if mean_brightness <= 1 or mean_brightness >= target_brightness - 20:
        return image.copy()
    gain = float(np.clip(target_brightness / mean_brightness, 0.85, 1.15))
    hsv[:, :, 2] = np.clip(value * gain, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
