import json
import subprocess
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.core.config import OCR_INPUT_DIR, OCR_RUNNER_PATH, OCR_RUNTIME_PYTHON, OCR_TIMEOUT_SECONDS
from app.schemas.ocr import OCRResult


class OCREngineUnavailableError(RuntimeError):
    pass


class PaddleOCREngine:
    """Run PaddleOCR in the fixed Python 3.12 OCR environment."""

    name = "paddleocr"

    def recognize(self, image: np.ndarray) -> OCRResult:
        if not OCR_RUNTIME_PYTHON.is_file():
            raise OCREngineUnavailableError("The isolated PaddleOCR runtime is unavailable.")

        OCR_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        input_path = OCR_INPUT_DIR / f"{uuid.uuid4().hex}.png"
        if not cv2.imwrite(str(input_path), image):
            raise RuntimeError("OCR input preparation failed.")

        command = [str(OCR_RUNTIME_PYTHON), str(OCR_RUNNER_PATH), str(input_path)]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=OCR_TIMEOUT_SECONDS,
                shell=False,
                cwd=str(OCR_INPUT_DIR.parent.parent),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("OCR processing timed out.") from exc
        except OSError as exc:
            raise OCREngineUnavailableError("The isolated PaddleOCR runtime could not be started.") from exc
        finally:
            input_path.unlink(missing_ok=True)

        if completed.returncode != 0:
            raise RuntimeError("OCR processing failed.")
        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            if payload.get("error"):
                raise RuntimeError("OCR processing failed.")
            return OCRResult.model_validate(payload)
        except (IndexError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError("OCR returned an invalid result.") from exc
