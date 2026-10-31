"""
Test PDF enhancement with validation and debugging.
"""
from pathlib import Path
from io import BytesIO

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def create_test_pdf() -> bytes:
    """Create a simple test PDF with visible content."""
    # Create a simple image
    image = np.full((700, 900, 3), 240, dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (800, 600), (50, 50, 50), -1)
    cv2.putText(image, "TEST DOCUMENT", (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 4)

    # Convert to RGB for PIL
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Create PDF with PyMuPDF
    pdf_doc = fitz.open()
    page = pdf_doc.new_page(width=612, height=792)  # Letter size in points

    # Convert PIL image to PNG bytes
    img_bytes = BytesIO()
    pil_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Insert image into PDF
    page.insert_image(page.rect, stream=img_bytes.read())

    # Save to bytes
    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    return pdf_bytes


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    import app.api.routes.documents as documents_route

    monkeypatch.setattr(documents_route, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(documents_route, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(documents_route, "ENHANCED_DIR", tmp_path / "processed" / "enhanced")
    monkeypatch.setattr(documents_route, "REPORTS_DIR", tmp_path / "processed" / "reports")


def upload_pdf():
    """Upload test PDF and return analysis response."""
    pdf_content = create_test_pdf()
    response = client.post(
        "/api/documents/analyze",
        files={"file": ("test.pdf", pdf_content, "application/pdf")}
    )
    assert response.status_code == 200
    return response.json()


def test_pdf_enhancement_creates_valid_output():
    """Test that PDF enhancement creates a valid, non-blank PDF."""
    analyzed = upload_pdf()

    # Enhance the document
    response = client.post(f"/api/documents/{analyzed['document_id']}/enhance")
    assert response.status_code == 200

    body = response.json()
    assert body["document_id"] == analyzed["document_id"]
    assert body["status"] in {"enhanced", "enhancement_rejected"}

    # If enhanced, verify the PDF exists and is valid
    if body["status"] == "enhanced":
        import app.api.routes.documents as documents_route

        enhanced_path = documents_route.ENHANCED_DIR / f"{analyzed['document_id']}_enhanced.pdf"
        assert enhanced_path.exists(), "Enhanced PDF file should exist"

        # Verify PDF can be opened and has content
        pdf_doc = fitz.open(str(enhanced_path))
        assert len(pdf_doc) == 1, "Enhanced PDF should have 1 page"

        # Render first page and verify it's not blank
        page = pdf_doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
        img_data = pix.tobytes("png")
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        assert img is not None, "Rendered page should not be None"

        # Check if page has actual content (not blank white)
        mean_pixel = np.mean(img)
        std_pixel = np.std(img)

        print(f"Enhanced PDF page stats: mean={mean_pixel:.2f}, std={std_pixel:.2f}")

        # A blank white page would have high mean and low std
        assert not (std_pixel < 1.0 and mean_pixel > 250), \
            f"Enhanced PDF page appears blank (mean={mean_pixel:.2f}, std={std_pixel:.2f})"

        pdf_doc.close()


def test_pdf_preview_endpoint_returns_inline():
    """Test that preview endpoint returns PDF with inline disposition."""
    analyzed = upload_pdf()

    # Enhance
    enhance_response = client.post(f"/api/documents/{analyzed['document_id']}/enhance")
    assert enhance_response.status_code == 200

    body = enhance_response.json()

    if body["enhanced_file"]["available"]:
        # Test preview endpoint
        preview_url = body["enhanced_file"]["preview_url"]
        preview_response = client.get(preview_url)

        assert preview_response.status_code == 200
        assert preview_response.headers["content-type"] == "application/pdf"
        assert "inline" in preview_response.headers.get("content-disposition", "").lower()

        # Test download endpoint
        download_url = body["enhanced_file"]["download_url"]
        download_response = client.get(download_url)

        assert download_response.status_code == 200
        assert download_response.headers["content-type"] == "application/pdf"
        assert "attachment" in download_response.headers.get("content-disposition", "").lower()


def test_pdf_debug_files_created():
    """Verify debug files are created during PDF enhancement."""
    analyzed = upload_pdf()

    # Enhance
    response = client.post(f"/api/documents/{analyzed['document_id']}/enhance")
    assert response.status_code == 200

    body = response.json()

    if body["status"] == "enhanced":
        import app.api.routes.documents as documents_route

        doc_id = analyzed["document_id"]

        # Check for debug files
        original_debug = documents_route.ENHANCED_DIR / f"{doc_id}_page_0_original_debug.png"
        enhanced_debug = documents_route.ENHANCED_DIR / f"{doc_id}_page_0_enhanced_debug.png"
        validation_debug = documents_route.ENHANCED_DIR / f"{doc_id}_page_0_validation.png"

        assert original_debug.exists(), "Original debug image should exist"
        assert enhanced_debug.exists(), "Enhanced debug image should exist"
        assert validation_debug.exists(), "Validation debug image should exist"

        # Verify debug images are not blank
        for debug_path in [original_debug, enhanced_debug, validation_debug]:
            img = cv2.imread(str(debug_path))
            assert img is not None
            assert img.size > 0
            print(f"{debug_path.name}: shape={img.shape}, mean={np.mean(img):.2f}")
