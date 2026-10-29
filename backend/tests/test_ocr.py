import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import UPLOAD_DIR
from app.main import app
from app.schemas.ocr import OCRDetection, OCRResult
from app.services.ocr.metrics import compare_ocr, text_similarity
from app.services.ocr.paddle_engine import (
    OCREngineUnavailableError,
    PaddleOCREngine,
)
from app.services.ocr.service import OCRService

client = TestClient(app)


class FakeEngine:
    name = "fake-ocr"

    def __init__(self, results=None, error=False, unavailable=False):
        self.results = list(results or [])
        self.error = error
        self.unavailable = unavailable

    def recognize(self, image):
        if self.unavailable:
            raise OCREngineUnavailableError("simulated unavailable OCR runtime")
        if self.error:
            raise RuntimeError("simulated OCR failure")
        return self.results.pop(0)


def ocr_result(text: str, confidence: float, words: int, lines: int) -> OCRResult:
    return OCRResult(
        engine="fake-ocr",
        text=text,
        confidence=confidence,
        word_count=words,
        line_count=lines,
        detections=[
            OCRDetection(
                text=text,
                confidence=confidence,
                bbox=[[0, 0], [10, 0], [10, 10], [0, 10]],
            )
        ]
        if text
        else [],
    )


def upload_bytes() -> bytes:
    image = np.full((500, 700, 3), 220, dtype=np.uint8)
    cv2.putText(
        image,
        "DOCLEAR DOCUMENT",
        (80, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (20, 20, 20),
        3,
    )
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    import app.api.routes.documents as documents_route

    monkeypatch.setattr(documents_route, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(documents_route, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(
        documents_route, "ENHANCED_DIR", tmp_path / "processed" / "enhanced"
    )
    monkeypatch.setattr(
        documents_route, "REPORTS_DIR", tmp_path / "processed" / "reports"
    )


# 1. Successful OCR execution (mocked subprocess output with markers and library noise)
def test_successful_ocr_execution(monkeypatch):
    engine = PaddleOCREngine()

    mock_stdout = (
        "INFO: ccache not found\n"
        "oneDNN v3.6.2\n"
        "__OCR_OUTPUT_START__\n"
        + json.dumps(
            {
                "engine": "paddleocr",
                "text": "Recognized invoice text",
                "confidence": 0.94,
                "word_count": 3,
                "line_count": 1,
                "detections": [
                    {
                        "text": "Recognized invoice text",
                        "confidence": 0.94,
                        "bbox": [[0.0, 0.0], [50.0, 0.0], [50.0, 10.0], [0.0, 10.0]],
                    }
                ],
            }
        )
        + "\n__OCR_OUTPUT_END__\n"
        "C++ exit cleanup\n"
    )

    mock_run = MagicMock(returncode=0, stdout=mock_stdout, stderr="")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: mock_run)

    dummy_image = np.full((100, 100, 3), 200, dtype=np.uint8)
    result = engine.recognize(dummy_image)

    assert result.engine == "paddleocr"
    assert result.text == "Recognized invoice text"
    assert result.confidence == 0.94
    assert result.word_count == 3
    assert len(result.detections) == 1


# 2. Empty OCR results (inconclusive validation)
def test_empty_ocr_results_produce_inconclusive_validation():
    original = ocr_result("", 0.0, 0, 0)
    enhanced = ocr_result("", 0.0, 0, 0)

    validation = compare_ocr(original, enhanced)
    assert validation.available is True
    assert validation.status == "inconclusive"
    assert validation.improvement.improved is False
    assert validation.improvement.readability_score == 0.0
    assert validation.content_preservation_warning is False
    assert "insufficient" in validation.error.lower()


# 3. OCR subprocess missing or unavailable
def test_ocr_runtime_missing_raises_unavailable(monkeypatch):
    import app.services.ocr.paddle_engine as paddle_engine_module

    fake_missing_path = Path("/nonexistent/python/path/python.exe")
    monkeypatch.setattr(
        paddle_engine_module, "get_ocr_python_executable", lambda: fake_missing_path
    )

    engine = PaddleOCREngine()
    dummy_image = np.full((50, 50, 3), 255, dtype=np.uint8)

    with pytest.raises(OCREngineUnavailableError, match="unavailable"):
        engine.recognize(dummy_image)


# 4. OCR subprocess returning a non-zero exit code
def test_ocr_subprocess_non_zero_exit_code_raises_runtime_error(monkeypatch):
    engine = PaddleOCREngine()
    mock_run = MagicMock(
        returncode=2, stdout="", stderr="ModuleNotFoundError: no module named paddle"
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: mock_run)

    dummy_image = np.full((50, 50, 3), 255, dtype=np.uint8)
    with pytest.raises(RuntimeError, match="exit code 2"):
        engine.recognize(dummy_image)


# 5. OCR timeout handling
def test_ocr_subprocess_timeout_raises_runtime_error(monkeypatch):
    engine = PaddleOCREngine()

    def mock_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=180)

    monkeypatch.setattr(subprocess, "run", mock_timeout)

    dummy_image = np.full((50, 50, 3), 255, dtype=np.uint8)
    with pytest.raises(RuntimeError, match="timed out"):
        engine.recognize(dummy_image)


# 6. Invalid OCR output / malformed JSON
def test_ocr_subprocess_invalid_output_raises_runtime_error(monkeypatch):
    engine = PaddleOCREngine()
    mock_run = MagicMock(
        returncode=0, stdout="Random unparseable process crash output\n", stderr=""
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: mock_run)

    dummy_image = np.full((50, 50, 3), 255, dtype=np.uint8)
    with pytest.raises(RuntimeError, match="invalid result"):
        engine.recognize(dummy_image)


# 7. Original-versus-enhanced text comparison
def test_ocr_comparison_and_text_similarity():
    original = ocr_result("Meeting notes October 2026", 0.70, 4, 1)
    enhanced = ocr_result("Meeting notes October 2026", 0.95, 4, 1)

    validation = compare_ocr(original, enhanced)
    assert validation.available is True
    assert validation.status == "validated"
    assert validation.improvement.confidence_delta == 0.25
    assert validation.improvement.improved is True
    assert validation.text_similarity == 1.0
    assert validation.content_preservation_warning is False
    assert validation.improvement.readability_score > 0


# 8. Enhancement rejected because text readability deteriorates
def test_enhancement_rejected_on_text_deterioration():
    # Content alteration / loss of similarity
    orig_text = "Important contract clause section 5"
    corrupted_text = "Imprtant xxx c1ause sctn"
    orig = ocr_result(orig_text, 0.85, 5, 1)
    enh = ocr_result(corrupted_text, 0.60, 4, 1)

    validation = compare_ocr(orig, enh)
    assert validation.status == "rejected"
    assert validation.improvement.improved is False
    assert validation.content_preservation_warning is True
    assert validation.improvement.readability_score == 0.0


# 9. Enhancement not recommended when OCR is unavailable
def test_enhancement_not_recommended_when_ocr_unavailable(monkeypatch):
    import app.api.routes.documents as documents_route

    upload = client.post(
        "/api/documents/analyze",
        files={"file": ("document.jpg", upload_bytes(), "image/jpeg")},
    )
    document_id = upload.json()["document_id"]

    monkeypatch.setattr(
        documents_route,
        "ocr_service",
        OCRService(engine=FakeEngine(unavailable=True)),
    )
    response = client.post(f"/api/documents/{document_id}/enhance")
    assert response.status_code == 200
    body = response.json()

    assert body["ocr_validation"]["available"] is False
    assert body["ocr_validation"]["status"] == "unavailable"
    assert body["final_recommendation"] == "ocr_validation_unavailable"
    assert body["enhanced_file"]["available"] is False
    assert body["enhanced_file"]["download_url"] is None


# 10. Original file remains unchanged after upload and enhancement
def test_original_file_remains_unchanged(monkeypatch):
    import app.api.routes.documents as documents_route

    raw_bytes = upload_bytes()
    upload = client.post(
        "/api/documents/analyze",
        files={"file": ("original_notes.jpg", raw_bytes, "image/jpeg")},
    )
    document_id = upload.json()["document_id"]

    stored_file = documents_route.UPLOAD_DIR / f"{document_id}.jpg"
    assert stored_file.is_file()
    assert stored_file.read_bytes() == raw_bytes

    original = ocr_result("Notes text sample", 0.70, 3, 1)
    enhanced = ocr_result("Notes text sample", 0.90, 3, 1)
    monkeypatch.setattr(
        documents_route,
        "ocr_service",
        OCRService(engine=FakeEngine([original, enhanced])),
    )

    response = client.post(f"/api/documents/{document_id}/enhance")
    assert response.status_code == 200

    # Ensure source upload is still 100% identical
    assert stored_file.is_file()
    assert stored_file.read_bytes() == raw_bytes


# 11. Successful validation allows recommendation only when all other criteria pass
def test_successful_validation_recommends_only_when_quality_improves(monkeypatch):
    import app.api.routes.documents as documents_route

    raw_bytes = upload_bytes()
    upload = client.post(
        "/api/documents/analyze",
        files={"file": ("document.jpg", raw_bytes, "image/jpeg")},
    )
    document_id = upload.json()["document_id"]

    # When OCR validates and quality improves
    orig_ocr = ocr_result("Text line header", 0.60, 3, 1)
    enh_ocr = ocr_result("Text line header", 0.92, 3, 1)
    monkeypatch.setattr(
        documents_route,
        "ocr_service",
        OCRService(engine=FakeEngine([orig_ocr, enh_ocr])),
    )

    response = client.post(f"/api/documents/{document_id}/enhance")
    assert response.status_code == 200
    body = response.json()
    assert body["ocr_validation"]["status"] == "validated"
    assert body["ocr_validation"]["improvement"]["improved"] is True

    if body["improvement"]["improved"]:
        assert body["final_recommendation"] == "recommended"
        assert body["enhanced_file"]["available"] is True
        assert (
            body["enhanced_file"]["download_url"]
            == f"/api/documents/{document_id}/enhanced"
        )
    else:
        assert body["final_recommendation"] == "review_required"
        assert body["enhanced_file"]["available"] is False
