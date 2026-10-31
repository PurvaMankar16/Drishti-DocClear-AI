from io import BytesIO

import numpy as np
import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services.quality_analysis import calculate_blur, calculate_brightness, calculate_contrast, calculate_skew
from app.services.quality_scoring import calculate_overall_score
from app.schemas.quality import QualityMetrics


client = TestClient(app)


def image_bytes(format_name: str = "JPEG") -> bytes:
    image = Image.new("RGB", (1200, 900), "white")
    pixels = image.load()
    for x in range(100, 1100):
        for y in range(100, 800):
            if x % 80 < 40 and y % 80 < 40:
                pixels[x, y] = (30, 30, 30)
    stream = BytesIO()
    image.save(stream, format=format_name)
    return stream.getvalue()


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    import app.api.routes.documents as documents_route

    monkeypatch.setattr(documents_route, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(documents_route, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(documents_route, "ENHANCED_DIR", tmp_path / "processed" / "enhanced")
    monkeypatch.setattr(documents_route, "REPORTS_DIR", tmp_path / "processed" / "reports")


def test_health_endpoint_regression():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "DocClear AI"}


def test_valid_jpg_upload():
    response = client.post("/api/documents/analyze", files={"file": ("marksheet.jpg", image_bytes(), "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["status"] == "analyzed"


def test_valid_png_upload():
    response = client.post("/api/documents/analyze", files={"file": ("marksheet.png", image_bytes("PNG"), "image/png")})
    assert response.status_code == 200
    assert "metrics" in response.json()


def test_valid_pdf_upload():
    pdf = fitz.open()
    page = pdf.new_page(width=600, height=450)
    page.insert_image(fitz.Rect(0, 0, 600, 450), stream=image_bytes("PNG"))
    response = client.post("/api/documents/analyze", files={"file": ("marksheet.pdf", pdf.tobytes(), "application/pdf")})
    assert response.status_code == 200
    assert response.json()["file_type"] == "application/pdf"


def test_unsupported_extension():
    response = client.post("/api/documents/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_oversized_file():
    response = client.post("/api/documents/analyze", files={"file": ("large.jpg", b"0" * (10 * 1024 * 1024 + 1), "image/jpeg")})
    assert response.status_code == 400


def test_corrupted_image():
    response = client.post("/api/documents/analyze", files={"file": ("broken.jpg", b"\xff\xd8\xffnot-an-image", "image/jpeg")})
    assert response.status_code == 422


def test_quality_analysis_response_structure():
    response = client.post("/api/documents/analyze", files={"file": ("marksheet.jpg", image_bytes(), "image/jpeg")})
    body = response.json()
    assert {"document_id", "filename", "file_type", "status", "overall_score", "classification", "metrics"} <= body.keys()
    assert {"blur", "brightness", "contrast", "noise", "skew", "cropping", "resolution"} <= body["metrics"].keys()


def test_blur_brightness_contrast_and_skew_calculations():
    gray = np.zeros((200, 200), dtype=np.uint8)
    gray[50:150, 50:150] = 255
    assert calculate_blur(gray).raw_score >= 0
    assert calculate_brightness(gray).mean_brightness >= 0
    assert calculate_contrast(gray).raw_score >= 0
    assert abs(calculate_skew(gray).angle) <= 15


def test_overall_quality_score_is_bounded():
    metrics = QualityMetrics.model_validate({
        "blur": {"score": 80, "raw_score": 200, "status": "good"},
        "brightness": {"mean_brightness": 128, "normalized_score": 100, "status": "good", "condition": "acceptable"},
        "contrast": {"score": 80, "raw_score": 72, "status": "good"},
        "noise": {"score": 90, "raw_score": 4, "status": "good"},
        "skew": {"angle": 0, "status": "good"},
        "cropping": {"document_detected": True, "centered": True, "appears_cropped": False, "status": "good"},
        "resolution": {"width": 1200, "height": 900, "megapixels": 1.08, "score": 100, "status": "good"},
    })
    score, classification = calculate_overall_score(metrics)
    assert 0 <= score <= 100
    assert classification == "acceptable"
