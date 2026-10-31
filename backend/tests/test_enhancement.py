from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.enhancement.pipeline import EnhancementResult


client = TestClient(app)


def synthetic_document() -> bytes:
    image = np.full((700, 900, 3), 80, dtype=np.uint8)
    cv2.rectangle(image, (80, 70), (820, 630), (230, 230, 230), -1)
    cv2.putText(image, "DOCLEAR AI", (180, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 30, 30), 5)
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


def upload_document():
    response = client.post("/api/documents/analyze", files={"file": ("document.jpg", synthetic_document(), "image/jpeg")})
    assert response.status_code == 200
    return response.json()


def test_enhancement_endpoint_and_output_creation():
    analyzed = upload_document()
    response = client.post(f"/api/documents/{analyzed['document_id']}/enhance")
    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == analyzed["document_id"]
    assert body["status"] in {"enhanced", "enhancement_not_beneficial", "enhancement_rejected"}
    assert body["operations"]
    assert "before" in body and "after" in body
    import app.api.routes.documents as documents_route

    # Enhanced file should exist (even if rejected, for debugging)
    assert (documents_route.ENHANCED_DIR / f"{analyzed['document_id']}_enhanced.png").is_file()


def test_original_file_preservation():
    analyzed = upload_document()
    import app.api.routes.documents as documents_route

    original_path = next(documents_route.UPLOAD_DIR.glob(f"{analyzed['document_id']}.*"))
    original_bytes = original_path.read_bytes()
    client.post(f"/api/documents/{analyzed['document_id']}/enhance")
    assert original_path.read_bytes() == original_bytes


def test_enhanced_download_is_not_a_filesystem_path():
    analyzed = upload_document()
    response = client.post(f"/api/documents/{analyzed['document_id']}/enhance")
    body = response.json()
    if body["enhanced_file"]["available"]:
        download = client.get(body["enhanced_file"]["download_url"])
        assert download.status_code == 200
        assert "C:\\" not in download.text


def test_invalid_document_id_and_missing_source():
    assert client.post("/api/documents/not-a-valid-id/enhance").status_code == 404
    assert client.post("/api/documents/aaaaaaaaaaaaaaaa/enhance").status_code == 404


def test_safety_gate_rejects_non_improving_enhancement(monkeypatch):
    """Test that enhancement with no improvement is rejected by validation."""
    analyzed = upload_document()
    import app.api.routes.documents as documents_route

    def unchanged(image, quality):
        return EnhancementResult(image=image.copy(), operations=[{"name": "none", "applied": False, "reason": "test"}])

    monkeypatch.setattr(documents_route, "enhance_document", unchanged)
    response = client.post(f"/api/documents/{analyzed['document_id']}/enhance")
    body = response.json()
    assert response.status_code == 200
    # With validation, status should be enhancement_rejected
    assert body["status"] in {"enhancement_not_beneficial", "enhancement_rejected"}
    assert body["improvement"]["improved"] is False
    assert body["enhanced_file"]["available"] is False


def test_pipeline_selects_adaptive_operations_for_low_quality():
    """Test that pipeline applies appropriate operations for low-quality images."""
    from app.services.quality_analysis import analyze_image
    from app.services.enhancement.pipeline import enhance_document

    image = np.full((120, 120, 3), 20, dtype=np.uint8)
    metrics = analyze_image(image)
    result = enhance_document(image, metrics)
    names = {operation["name"] for operation in result.operations}
    # background_normalization replaces illumination_correction
    assert "background_normalization" in names
    assert result.image.shape[0] > 0
