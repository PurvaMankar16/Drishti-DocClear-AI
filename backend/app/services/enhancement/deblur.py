import cv2
import numpy as np


def sharpen(image: np.ndarray, amount: float = 0.55) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), 1.1)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
