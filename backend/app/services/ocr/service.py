import numpy as np

from app.schemas.ocr import OCRResult
from app.services.ocr.engine import OCREngine
from app.services.ocr.paddle_engine import PaddleOCREngine
from app.services.ocr.preprocessing import prepare_for_ocr


class OCRService:
    def __init__(self, engine: OCREngine | None = None) -> None:
        self._engine = engine

    def recognize(self, image: np.ndarray) -> OCRResult:
        if self._engine is None:
            self._engine = PaddleOCREngine()
        prepared = prepare_for_ocr(image)
        return self._engine.recognize(prepared)
