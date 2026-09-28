from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.ocr import OCRDetection, OCRResult
from app.services.ocr.metrics import compare_ocr, text_similarity
from app.services.ocr.service import OCRService


client = TestClient(app)


class FakeEngine:
    name = "fake-ocr"

    def __init__(self, results=None, error=False):
        self.results = list(results or [])
        self.error = error

    def recognize(self, image):
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
        detections=[OCRDetection(text=text, confidence=confidence, bbox=[[0, 0], [10, 0], [10, 10], [0, 10]])],
    )


def upload_bytes() -> bytes:
    import cv2

    image = np.full((500, 700, 3), 220, dtype=np.uint8)
    cv2.putText(image, "DOCLEAR DOCUMENT", (80, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    import app.api.routes.documents as documents_route

    monkeypatch.setattr(documents_route, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(documents_route, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(documents_route, "ENHANCED_DIR", tmp_path / "processed" / "enhanced")
    monkeypatch.setattr(documents_route, "REPORTS_DIR", tmp_path / "processed" / "reports")


def test_ocr_service_initialization_and_schema():
    engine = FakeEngine([ocr_result("Candidate Name", 0.91, 2, 1)])
    result = OCRService(engine=engine).recognize(np.zeros((20, 20, 3), dtype=np.uint8))
    assert result.engine == "fake-ocr"
    assert result.confidence == 0.91
    assert result.word_count == 2
    assert result.detections[0].bbox


def test_ocr_comparison_and_readability_score():
    original = ocr_result("name purva mankar date 2026 marks 90", 0.61, 7, 2)
    enhanced = ocr_result("name purva mankar date 2026 marks 90", 0.92, 8, 3)
    comparison = compare_ocr(original, enhanced)
    assert comparison.improvement.confidence_delta == 0.31
    assert comparison.improvement.readability_score > 0
    assert comparison.improvement.improved is True
    assert comparison.content_preservation_warning is False


def test_degraded_ocr_result_and_text_similarity():
    original = ocr_result("candidate name marks date", 0.92, 5, 2)
    enhanced = ocr_result("unrelated content", 0.50, 2, 1)
    comparison = compare_ocr(original, enhanced)
    assert comparison.improvement.improved is False
    assert comparison.content_preservation_warning is True
    assert text_similarity("Name: Purva Mankar", "name: purva mankar") == 1.0


def test_ocr_failure_is_controlled():
    service = OCRService(engine=FakeEngine(error=True))
    with pytest.raises(RuntimeError, match="simulated OCR failure"):
        service.recognize(np.zeros((20, 20, 3), dtype=np.uint8))


def test_enhancement_endpoint_includes_ocr_validation_and_does_not_persist_text(monkeypatch):
    import app.api.routes.documents as documents_route
    from app.services.ocr.service import OCRService

    upload = client.post("/api/documents/analyze", files={"file": ("document.jpg", upload_bytes(), "image/jpeg")})
    document_id = upload.json()["document_id"]
    original = ocr_result("candidate name marks date 2026", 0.61, 5, 2)
    enhanced = ocr_result("candidate name marks date 2026", 0.92, 6, 3)
    monkeypatch.setattr(documents_route, "ocr_service", OCRService(engine=FakeEngine([original, enhanced])))

    response = client.post(f"/api/documents/{document_id}/enhance")
    assert response.status_code == 200
    body = response.json()
    assert body["ocr_validation"]["available"] is True
    assert body["ocr_validation"]["original"]["word_count"] == 5
    assert body["ocr_validation"]["enhanced"]["word_count"] == 6
    assert body["final_recommendation"] in {"recommended", "review_required"}

    report_path = documents_route.REPORTS_DIR / f"{document_id}_enhancement.json"
    persisted = report_path.read_text(encoding="utf-8")
    assert "candidate name marks date" not in persisted


def test_ocr_validation_unavailable_does_not_claim_success(monkeypatch):
    import app.api.routes.documents as documents_route
    from app.services.ocr.service import OCRService

    upload = client.post("/api/documents/analyze", files={"file": ("document.jpg", upload_bytes(), "image/jpeg")})
    document_id = upload.json()["document_id"]
    monkeypatch.setattr(documents_route, "ocr_service", OCRService(engine=FakeEngine(error=True)))
    response = client.post(f"/api/documents/{document_id}/enhance")
    body = response.json()
    assert response.status_code == 200
    assert body["ocr_validation"]["available"] is False
    assert body["final_recommendation"] == "ocr_validation_unavailable"
    assert body["enhanced_file"]["available"] is False
