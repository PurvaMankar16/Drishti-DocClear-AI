import cv2
import numpy as np


def correct_illumination(image: np.ndarray, target_brightness: float = 128.0) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    mean_brightness = float(value.mean())
    if mean_brightness <= 1:
        return image.copy()
    gain = float(np.clip(target_brightness / mean_brightness, 0.85, 1.25))
    hsv[:, :, 2] = np.clip(value * gain, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
