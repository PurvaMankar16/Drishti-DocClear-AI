"""Tests for document layout detection functionality."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.ml.layout_detector import (
    detect_layout,
    model_is_available,
    LAYOUT_CLASS_NAMES,
    NUM_CLASSES,
    LayoutDetectionError,
)


client = TestClient(app)


def synthetic_document_image() -> np.ndarray:
    """Create a simple synthetic document image for testing."""
    # Create a white canvas
    image = np.full((800, 600, 3), 255, dtype=np.uint8)

    # Add some dark regions to simulate text blocks
    # These won't be accurately detected, but allow testing the pipeline
    image[100:200, 50:550] = 30  # Header-like region
    image[250:600, 50:550] = 50  # Text-like region
    image[650:750, 50:550] = 30  # Footer-like region

    return image


class TestLayoutDetectorService:
    """Tests for layout_detector.py service."""

    def test_model_availability_check(self):
        """Test that model availability can be checked without loading."""
        # Just verify this doesn't raise an exception
        available = model_is_available()
        assert isinstance(available, bool)

    def test_class_names_configuration(self):
        """Verify layout class names match DocLayNet specification."""
        expected_classes = [
            "Caption", "Footnote", "Formula", "List-item",
            "Page-footer", "Page-header", "Picture",
            "Section-header", "Table", "Text", "Title"
        ]
        assert LAYOUT_CLASS_NAMES == expected_classes
        assert NUM_CLASSES == 12  # 11 classes + background

    def test_empty_image_raises_error(self):
        """Empty images should raise LayoutDetectionError."""
        empty_image = np.array([], dtype=np.uint8)

        with pytest.raises(LayoutDetectionError, match="empty image"):
            detect_layout(empty_image)

    def test_none_image_raises_error(self):
        """None image should raise LayoutDetectionError."""
        with pytest.raises(LayoutDetectionError, match="empty image"):
            detect_layout(None)

    @pytest.mark.skipif(
        not model_is_available(),
        reason="Layout detection checkpoint not available"
    )
    def test_detect_layout_returns_valid_structure(self):
        """Test layout detection returns properly structured results."""
        image = synthetic_document_image()

        # Run detection with lenient thresholds to get some results
        detections = detect_layout(
            image,
            confidence_threshold=0.1,
            nms_threshold=0.5
        )

        # Verify structure even if no detections (model might not detect anything on synthetic image)
        assert isinstance(detections, list)

        # If we get detections, verify their structure
        for detection in detections:
            assert "label" in detection
            assert "label_id" in detection
            assert "confidence" in detection
            assert "bbox" in detection

            # Verify label is valid
            assert detection["label"] in LAYOUT_CLASS_NAMES or detection["label"].startswith("Unknown")

            # Verify label_id is in valid range (1-11, background is 0)
            assert 1 <= detection["label_id"] <= 11

            # Verify confidence is in range
            assert 0.0 <= detection["confidence"] <= 1.0

            # Verify bbox structure
            bbox = detection["bbox"]
            assert "x1" in bbox and "y1" in bbox
            assert "x2" in bbox and "y2" in bbox
            assert bbox["x1"] < bbox["x2"]
            assert bbox["y1"] < bbox["y2"]

    @pytest.mark.skipif(
        not model_is_available(),
        reason="Layout detection checkpoint not available"
    )
    def test_confidence_filtering_works(self):
        """Test that confidence threshold filters low-confidence detections."""
        image = synthetic_document_image()

        # Get detections with low threshold
        low_threshold_detections = detect_layout(
            image,
            confidence_threshold=0.1,
            nms_threshold=0.5
        )

        # Get detections with high threshold
        high_threshold_detections = detect_layout(
            image,
            confidence_threshold=0.8,
            nms_threshold=0.5
        )

        # High threshold should return same or fewer detections
        assert len(high_threshold_detections) <= len(low_threshold_detections)

    @pytest.mark.skipif(
        not model_is_available(),
        reason="Layout detection checkpoint not available"
    )
    def test_nms_reduces_overlapping_detections(self):
        """Test that NMS reduces overlapping bounding boxes."""
        image = synthetic_document_image()

        # NMS with high IoU threshold (less aggressive)
        loose_nms = detect_layout(
            image,
            confidence_threshold=0.3,
            nms_threshold=0.9
        )

        # NMS with low IoU threshold (more aggressive)
        strict_nms = detect_layout(
            image,
            confidence_threshold=0.3,
            nms_threshold=0.1
        )

        # Stricter NMS should return same or fewer detections
        assert len(strict_nms) <= len(loose_nms)


class TestLayoutDetectionAPI:
    """Tests for layout detection API endpoint."""

    @pytest.fixture(autouse=True)
    def isolated_storage(self, tmp_path, monkeypatch):
        """Isolate file storage for tests."""
        import app.api.routes.documents as documents_route

        monkeypatch.setattr(documents_route, "UPLOAD_DIR", tmp_path / "uploads")
        monkeypatch.setattr(documents_route, "PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr(documents_route, "ENHANCED_DIR", tmp_path / "processed" / "enhanced")
        monkeypatch.setattr(documents_route, "REPORTS_DIR", tmp_path / "processed" / "reports")

    def upload_test_document(self) -> str:
        """Helper to upload a synthetic document and return its ID."""
        import cv2
        image = synthetic_document_image()
        success, encoded = cv2.imencode(".jpg", image)
        assert success

        response = client.post(
            "/api/documents/analyze",
            files={"file": ("test_doc.jpg", encoded.tobytes(), "image/jpeg")}
        )
        assert response.status_code == 200
        return response.json()["document_id"]

    def test_layout_detection_endpoint_exists(self):
        """Verify layout detection endpoint is registered."""
        # Invalid ID should return 404, not 405 Method Not Allowed
        response = client.post("/api/documents/nonexistent_id/layout")
        assert response.status_code in (404, 422, 500)  # Not 405

    def test_invalid_document_id_returns_404(self):
        """Test that invalid document ID returns 404."""
        response = client.post("/api/documents/invalid_id_12345/layout")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_layout_detection_response_structure(self):
        """Test that layout detection returns properly structured response."""
        document_id = self.upload_test_document()

        response = client.post(f"/api/documents/{document_id}/layout")

        # Should succeed or gracefully handle missing model
        assert response.status_code == 200

        body = response.json()

        # Verify required fields
        assert "document_id" in body
        assert body["document_id"] == document_id
        assert "filename" in body
        assert "status" in body
        assert body["status"] in ["detected", "no_detections", "model_unavailable"]
        assert "model_available" in body
        assert isinstance(body["model_available"], bool)
        assert "confidence_threshold" in body
        assert "nms_threshold" in body
        assert "pages" in body
        assert isinstance(body["pages"], list)
        assert "total_detections" in body
        assert isinstance(body["total_detections"], int)

    @pytest.mark.skipif(
        not model_is_available(),
        reason="Layout detection checkpoint not available"
    )
    def test_layout_detection_with_available_model(self):
        """Test layout detection when model is available."""
        document_id = self.upload_test_document()

        response = client.post(f"/api/documents/{document_id}/layout")
        assert response.status_code == 200

        body = response.json()
        assert body["model_available"] is True
        assert body["status"] in ["detected", "no_detections"]

        # If detections found, verify structure
        if body["status"] == "detected":
            assert len(body["pages"]) > 0
            page = body["pages"][0]
            assert "page_number" in page
            assert "image_width" in page
            assert "image_height" in page
            assert "detections" in page
            assert "detection_count" in page

    def test_layout_report_saved_to_filesystem(self):
        """Test that layout detection results are saved to reports directory."""
        import app.api.routes.documents as documents_route

        document_id = self.upload_test_document()
        client.post(f"/api/documents/{document_id}/layout")

        # Check if report was saved
        report_path = documents_route.REPORTS_DIR / f"{document_id}_layout.json"
        assert report_path.is_file()

        # Verify report contains valid JSON
        import json
        report_data = json.loads(report_path.read_text(encoding="utf-8"))
        assert report_data["document_id"] == document_id


def test_existing_endpoints_still_work():
    """Regression test: verify existing endpoints were not broken."""
    # Health endpoint
    response = client.get("/health")
    assert response.status_code == 200

    # Analyze endpoint still exists
    import cv2
    image = synthetic_document_image()
    success, encoded = cv2.imencode(".jpg", image)

    response = client.post(
        "/api/documents/analyze",
        files={"file": ("test.jpg", encoded.tobytes(), "image/jpeg")}
    )
    assert response.status_code == 200
    assert "document_id" in response.json()
