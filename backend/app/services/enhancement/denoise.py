import cv2
import numpy as np


def denoise(image: np.ndarray, strength: int = 7) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
