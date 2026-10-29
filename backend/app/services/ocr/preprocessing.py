import cv2
import numpy as np


def prepare_for_ocr(image: np.ndarray, target_width: int = 1600) -> np.ndarray:
    """Resize the image to a manageable width while preserving colour channels.

    PaddleOCR performs better on colour (BGR) images because its internal
    pre-processing already converts to the appropriate colour space.  The
    previous greyscale conversion discarded colour cues used by the detection
    and recognition heads and could reduce accuracy on coloured documents.

    The image is returned as a BGR uint8 array (unchanged if already within the
    target width), ready to be written to a temporary PNG by the OCR engine.
    """
    height, width = image.shape[:2]
    if width > target_width:
        scale = target_width / width
        image = cv2.resize(
            image,
            (target_width, max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    # Ensure uint8 output regardless of input dtype.
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image
