from io import BytesIO

import cv2
import numpy as np
import pymupdf as fitz
from PIL import Image, UnidentifiedImageError


class DocumentProcessingError(ValueError):
    pass


def image_from_upload(content: bytes, extension: str) -> np.ndarray:
    try:
        if extension == ".pdf":
            return _first_pdf_page(content)
        with Image.open(BytesIO(content)) as image:
            image.load()
            rgb_image = image.convert("RGB")
            return cv2.cvtColor(np.asarray(rgb_image), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, RuntimeError, ValueError) as exc:
        raise DocumentProcessingError("The uploaded document could not be processed.") from exc


def _first_pdf_page(content: bytes) -> np.ndarray:
    try:
        with fitz.open(stream=content, filetype="pdf") as document:
            if document.page_count == 0:
                raise DocumentProcessingError("The uploaded PDF has no pages.")
            page = document.load_page(0)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except DocumentProcessingError:
        raise
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        raise DocumentProcessingError("The uploaded PDF could not be converted for analysis.") from exc
