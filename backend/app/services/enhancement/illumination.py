import cv2
import numpy as np


def correct_illumination(image: np.ndarray, target_brightness: float = 128.0) -> np.ndarray:
    """Apply a conservative multiplicative brightness correction.

    The gain is capped to ±25% (i.e. between 0.75× and 1.25×) to avoid
    clipping bright regions or washing out dark ones.

    Supports both 3-channel BGR and single-channel greyscale images.
    """
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        gray = image if image.ndim == 2 else image[:, :, 0]
        mean_brightness = float(gray.mean())
        if mean_brightness <= 1:
            return image.copy()
        gain = float(np.clip(target_brightness / mean_brightness, 0.75, 1.25))
        corrected = np.clip(gray.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return corrected if image.ndim == 2 else corrected[:, :, np.newaxis]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    mean_brightness = float(value.mean())
    if mean_brightness <= 1:
        return image.copy()
    gain = float(np.clip(target_brightness / mean_brightness, 0.75, 1.25))
    hsv[:, :, 2] = np.clip(value * gain, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
