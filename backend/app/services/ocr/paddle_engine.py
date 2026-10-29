import json
import logging
import os
import subprocess
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.core.config import (
    OCR_INPUT_DIR,
    OCR_RUNNER_PATH,
    OCR_RUNTIME_PYTHON,
    OCR_TIMEOUT_SECONDS,
)
from app.schemas.ocr import OCRResult

logger = logging.getLogger(__name__)


class OCREngineUnavailableError(RuntimeError):
    pass


def get_ocr_python_executable() -> Path:
    env_override = os.getenv("OCR_RUNTIME_PYTHON")
    if env_override:
        return Path(env_override)
    return OCR_RUNTIME_PYTHON


class PaddleOCREngine:
    """Run PaddleOCR in the isolated Python PaddleOCR runtime."""

    name = "paddleocr"

    def recognize(self, image: np.ndarray) -> OCRResult:
        python_executable = get_ocr_python_executable()
        if not python_executable.is_file():
            logger.warning(
                "OCR runtime executable does not exist at %s",
                python_executable,
            )
            raise OCREngineUnavailableError(
                f"The isolated PaddleOCR runtime is unavailable at {python_executable}."
            )

        OCR_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        input_path = OCR_INPUT_DIR / f"{uuid.uuid4().hex}.png"
        if not cv2.imwrite(str(input_path), image):
            logger.error("Failed to write temporary image for OCR to %s", input_path.name)
            raise RuntimeError("OCR input preparation failed.")

        command = [str(python_executable), str(OCR_RUNNER_PATH), str(input_path)]
        env = {
            **os.environ,
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
            "PYTHONIOENCODING": "utf-8",
        }

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
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            logger.error(
                "OCR subprocess timed out after %d seconds.",
                OCR_TIMEOUT_SECONDS,
            )
            raise RuntimeError("OCR processing timed out.") from exc
        except OSError as exc:
            logger.error(
                "Failed to spawn OCR subprocess with executable %s: %s",
                python_executable,
                exc,
            )
            raise OCREngineUnavailableError(
                "The isolated PaddleOCR runtime could not be started."
            ) from exc
        finally:
            input_path.unlink(missing_ok=True)

        if completed.returncode != 0:
            stderr_excerpt = (
                completed.stderr.strip()[:400] if completed.stderr else "none"
            )
            logger.error(
                "OCR subprocess failed with exit code %d. Stderr: %s",
                completed.returncode,
                stderr_excerpt,
            )
            raise RuntimeError(
                f"OCR processing failed with exit code {completed.returncode}."
            )

        payload_json = self._extract_json_payload(completed.stdout)
        if not payload_json:
            stderr_excerpt = (
                completed.stderr.strip()[:300] if completed.stderr else "none"
            )
            logger.error(
                "OCR subprocess did not produce valid JSON. Stderr: %s",
                stderr_excerpt,
            )
            raise RuntimeError("OCR returned an invalid result.")

        try:
            payload = json.loads(payload_json)
            if payload.get("error"):
                logger.error("OCR runner reported error: %s", payload.get("error"))
                raise RuntimeError(f"OCR processing failed: {payload.get('error')}")
            return OCRResult.model_validate(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Failed to parse OCR result payload: %s", exc)
            raise RuntimeError("OCR returned an invalid result.") from exc

    def _extract_json_payload(self, stdout: str) -> str | None:
        """Extract structured JSON from stdout, ignoring C++ and library log lines."""
        lines = stdout.splitlines()

        # Check for marker block __OCR_OUTPUT_START__ ... __OCR_OUTPUT_END__
        in_marker = False
        marker_lines: list[str] = []
        for line in lines:
            line_stripped = line.strip()
            if line_stripped == "__OCR_OUTPUT_START__":
                in_marker = True
                marker_lines = []
            elif line_stripped == "__OCR_OUTPUT_END__":
                in_marker = False
                return "\n".join(marker_lines)
            elif in_marker:
                marker_lines.append(line)

        # Fallback: scan backwards for the last valid JSON object containing expected fields
        for line in reversed(lines):
            line_stripped = line.strip()
            if line_stripped.startswith("{") and line_stripped.endswith("}"):
                try:
                    data = json.loads(line_stripped)
                    if isinstance(data, dict) and ("engine" in data or "error" in data):
                        return line_stripped
                except json.JSONDecodeError:
                    continue

        return None
