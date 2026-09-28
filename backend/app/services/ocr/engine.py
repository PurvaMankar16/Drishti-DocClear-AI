from typing import Protocol

import numpy as np

from app.schemas.ocr import OCRResult


class OCREngine(Protocol):
    name: str

    def recognize(self, image: np.ndarray) -> OCRResult:
        """Recognize text from a BGR image without logging or persisting its content."""
