import cv2
import numpy as np


def upscale(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LANCZOS4)
