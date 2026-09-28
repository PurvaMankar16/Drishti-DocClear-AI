import cv2
import numpy as np


def prepare_for_ocr(image: np.ndarray, target_width: int = 1600) -> np.ndarray:
    height, width = image.shape[:2]
    if width > target_width:
        scale = target_width / width
        image = cv2.resize(image, (target_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
