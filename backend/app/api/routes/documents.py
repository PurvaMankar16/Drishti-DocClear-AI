import json
import logging
import secrets
import re
from pathlib import Path
from io import BytesIO

import cv2
import fitz  # PyMuPDF
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, Response
from PIL import Image

from app.core.config import (
    ENHANCED_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    UPLOAD_DIR,
    LAYOUT_CONFIDENCE_THRESHOLD,
    LAYOUT_NMS_IOU_THRESHOLD,
)
from app.schemas.enhancement import (
    EnhancedFile,
    EnhancementOperation,
    EnhancementResponse,
    ImprovementReport,
    QualityComparison,
)
from app.schemas.document import DocumentAnalysisResponse
from app.schemas.layout import (
    LayoutDetectionResponse,
    PageLayoutDetections,
    LayoutDetection,
    BoundingBox,
    LayoutDetectionError,
)
from app.services.document_preprocessing import DocumentProcessingError, image_from_upload
from app.services.enhancement.pipeline import enhance_document
from app.services.quality_analysis import analyze_image
from app.services.ml.defect_classifier import predict_defects, DefectPredictionModelError
from app.services.ml.layout_detector import detect_layout, LayoutDetectionError as LayoutModelError
from app.services.quality_scoring import calculate_overall_score, validate_enhancement
from app.utils.file_validation import FileValidationError, validate_file_signature, validate_upload_metadata


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/analyze", response_model=DocumentAnalysisResponse)
async def analyze_document(file: UploadFile = File(...)) -> DocumentAnalysisResponse:
    content = await file.read()
    try:
        extension = validate_upload_metadata(file.filename, file.content_type, len(content))
        validate_file_signature(extension, content)
    except FileValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    document_id = secrets.token_urlsafe(16)
    safe_original_name = Path(file.filename or "document").name
    stored_path = UPLOAD_DIR / f"{document_id}{extension}"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stored_path.write_bytes(content)

    try:
        image = image_from_upload(content, extension)
        metrics = analyze_image(image)
        overall_score, classification = calculate_overall_score(metrics)
        try:
            defect_predictions = predict_defects(image)
            ml_available = True
        except DefectPredictionModelError:
            # Keep classical quality analysis available if the optional model is absent/unloadable.
            defect_predictions = []
            ml_available = False
        report = DocumentAnalysisResponse(
            document_id=document_id,
            filename=safe_original_name,
            file_type=file.content_type or _mime_type_for_extension(extension),
            status="analyzed",
            overall_score=overall_score,
            classification=classification,
            metrics=metrics,
            defect_predictions=defect_predictions,
            ml_model_available=ml_available,
        )
        (PROCESSED_DIR / f"{document_id}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return report
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Document analysis failed.") from exc


@router.post("/{document_id}/enhance", response_model=EnhancementResponse)
async def enhance_document_endpoint(document_id: str) -> EnhancementResponse:
    """Enhance document with adaptive operations and validation.

    For PDFs: Processes all pages and reconstructs multi-page PDF
    For images: Returns enhanced image in PNG format

    Args:
        document_id: ID of previously uploaded document

    Returns:
        EnhancementResponse with validation results
    """
    source_path = _find_source_path(document_id)
    if source_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    source_content = source_path.read_bytes()
    extension = source_path.suffix.lower()

    try:
        # Determine if PDF or image
        is_pdf = extension == ".pdf"

        if is_pdf:
            # Process PDF with multi-page support
            return await _enhance_pdf_document(document_id, source_path, source_content)
        else:
            # Process single image
            return await _enhance_image_document(document_id, source_path, source_content, extension)

    except DocumentProcessingError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error(f"Enhancement failed for {document_id}: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Document enhancement failed.") from exc


async def _enhance_image_document(document_id: str, source_path: Path, source_content: bytes, extension: str) -> EnhancementResponse:
    """Enhance single image document."""
    source_image = image_from_upload(source_content, extension)
    original_report = _load_or_create_report(document_id, source_path, source_image)

    # Enhance the document image
    enhancement_result = enhance_document(source_image, original_report.metrics)
    enhanced_image = enhancement_result.image
    enhanced_metrics = analyze_image(enhanced_image)
    enhanced_score, enhanced_classification = calculate_overall_score(enhanced_metrics)

    # Validate enhancement
    is_valid, validation_message, validation_details = validate_enhancement(
        original_report.metrics,
        enhanced_metrics
    )

    # Predict defects on enhanced image
    try:
        enhanced_defect_predictions = predict_defects(enhanced_image)
        enhanced_ml_available = True
    except DefectPredictionModelError:
        enhanced_defect_predictions = []
        enhanced_ml_available = False

    # Save enhanced image
    enhanced_filename = f"{document_id}_enhanced.png"
    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    enhanced_path = ENHANCED_DIR / enhanced_filename
    if not cv2.imwrite(str(enhanced_path), enhanced_image):
        raise OSError("Unable to save enhanced document.")

    enhanced_report = DocumentAnalysisResponse(
        document_id=document_id,
        filename=enhanced_filename,
        file_type="image/png",
        status="analyzed",
        overall_score=enhanced_score,
        classification=enhanced_classification,
        metrics=enhanced_metrics,
        defect_predictions=enhanced_defect_predictions,
        ml_model_available=enhanced_ml_available,
    )

    before = _comparison(original_report)
    after = _comparison(enhanced_report)
    improvement = ImprovementReport(
        absolute_score_change=round(after.quality - before.quality, 2),
        sharpness_change=round(after.sharpness - before.sharpness, 2),
        brightness_change=round(after.brightness - before.brightness, 2),
        contrast_change=round(after.contrast - before.contrast, 2),
        noise_change=round(after.noise - before.noise, 2),
        skew_change=round(after.skew - before.skew, 2),
        improved=is_valid,  # Use validation result instead of simple score comparison
    )

    # Recommendation based on validation
    final_recommendation = "recommended" if is_valid else "review_required"
    recommended = is_valid

    # Add validation info to operations
    operations_with_validation = enhancement_result.operations.copy()
    operations_with_validation.append({
        "name": "validation",
        "applied": True,
        "result": {
            "is_valid": is_valid,
            "message": validation_message,
            "details": validation_details,
        }
    })

    response = EnhancementResponse(
        document_id=document_id,
        status="enhanced" if is_valid else "enhancement_rejected",
        original_quality=original_report,
        enhanced_quality=enhanced_report,
        before=before,
        after=after,
        improvement=improvement,
        operations=[EnhancementOperation.model_validate(operation) for operation in operations_with_validation],
        enhanced_file=EnhancedFile(
            available=recommended,
            preview_url=f"/api/documents/{document_id}/enhanced-preview" if recommended else None,
            download_url=f"/api/documents/{document_id}/enhanced" if recommended else None,
            filename=enhanced_filename if recommended else None,
        ),
        final_recommendation=final_recommendation,
    )

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    persisted = response.model_dump()
    (REPORTS_DIR / f"{document_id}_enhancement.json").write_text(json.dumps(persisted, indent=2), encoding="utf-8")

    return response


async def _enhance_pdf_document(document_id: str, source_path: Path, source_content: bytes) -> EnhancementResponse:
    """Enhance multi-page PDF document."""
    # Open PDF
    pdf_doc = fitz.open(stream=source_content, filetype="pdf")
    page_count = len(pdf_doc)

    # Process first page for quality analysis (representative)
    first_page = pdf_doc[0]
    pix = first_page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
    img_data = pix.tobytes("png")
    source_image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

    original_report = _load_or_create_report(document_id, source_path, source_image)

    # Enhance first page to check if enhancement is beneficial
    enhancement_result = enhance_document(source_image, original_report.metrics)
    enhanced_first = enhancement_result.image
    enhanced_metrics = analyze_image(enhanced_first)

    # Validate enhancement
    is_valid, validation_message, validation_details = validate_enhancement(
        original_report.metrics,
        enhanced_metrics
    )

    enhanced_images = []

    if is_valid:
        # Enhancement is beneficial - process all pages
        logger.info(f"Enhancement validated for {document_id}, processing {page_count} page(s)")

        for page_num in range(page_count):
            page = pdf_doc[page_num]
            # Render at 2x for quality
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img_data = pix.tobytes("png")
            page_image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

            if page_image is None:
                raise ValueError(f"Failed to decode page {page_num} from PDF")

            logger.info(f"Page {page_num} original image shape: {page_image.shape}")

            # Save original page for comparison
            original_debug_path = ENHANCED_DIR / f"{document_id}_page_{page_num}_original_debug.png"
            cv2.imwrite(str(original_debug_path), page_image)

            # Analyze page quality
            page_quality = analyze_image(page_image)

            # Enhance page
            page_result = enhance_document(page_image, page_quality)
            enhanced_page = page_result.image

            if enhanced_page is None:
                raise ValueError(f"Enhancement returned None for page {page_num}")

            logger.info(f"Page {page_num} enhanced image shape: {enhanced_page.shape}")
            enhanced_images.append(enhanced_page)

        # Create multi-page PDF from enhanced images
        enhanced_filename = f"{document_id}_enhanced.pdf"
        ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
        enhanced_path = ENHANCED_DIR / enhanced_filename

        # Use PyMuPDF to create PDF from enhanced images
        output_pdf = fitz.open()

        for img_idx, img in enumerate(enhanced_images):
            try:
                # Validate enhanced image is not empty
                if img is None or img.size == 0:
                    logger.error(f"Page {img_idx} enhanced image is None or empty")
                    raise ValueError(f"Enhanced image for page {img_idx} is invalid")

                img_height, img_width = img.shape[:2]
                logger.info(f"Processing page {img_idx}: enhanced image shape = {img.shape}")

                # Save diagnostic image to verify enhancement worked
                debug_path = ENHANCED_DIR / f"{document_id}_page_{img_idx}_enhanced_debug.png"
                cv2.imwrite(str(debug_path), img)
                logger.info(f"Saved debug image to {debug_path}")

                # Convert BGR to RGB for PDF
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Encode as high-quality PNG
                success, encoded = cv2.imencode('.png', img_rgb, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if not success:
                    raise ValueError(f"Failed to encode page {img_idx} as PNG")

                png_data = encoded.tobytes()
                png_size = len(png_data)
                logger.info(f"Page {img_idx} PNG size: {png_size} bytes")

                if png_size == 0:
                    raise ValueError(f"PNG encoding resulted in 0 bytes for page {img_idx}")

                # Create PDF page with dimensions matching the image
                # Use 150 DPI for conversion (2x rendering = ~144 DPI, rounded)
                dpi = 150.0
                page_width = float(img_width) * 72.0 / dpi
                page_height = float(img_height) * 72.0 / dpi

                logger.info(f"Creating PDF page {img_idx}: {page_width:.2f}x{page_height:.2f} points")
                page = output_pdf.new_page(width=page_width, height=page_height)

                # Define rectangle for image (full page)
                rect = fitz.Rect(0, 0, page_width, page_height)

                # Insert the image
                # Note: PyMuPDF's insert_image returns the xref of the inserted image
                xref = page.insert_image(
                    rect,
                    stream=png_data,
                    keep_proportion=False,  # Scale to fill rect exactly
                    overlay=True  # Place on top of page content (even though page is empty)
                )

                logger.info(f"Page {img_idx} inserted with xref: {xref}")

                if xref == 0:
                    raise ValueError(f"insert_image returned 0 (failure) for page {img_idx}")

            except Exception as e:
                logger.error(f"Failed to add page {img_idx} to PDF: {e}", exc_info=True)
                raise  # Don't continue - fail fast so we can diagnose

        output_pdf.save(str(enhanced_path))

        # Phase 2: Validate the generated PDF
        logger.info(f"Validating generated PDF at {enhanced_path}")
        try:
            # Reopen and verify the PDF
            validation_pdf = fitz.open(str(enhanced_path))
            validation_page_count = len(validation_pdf)

            if validation_page_count != page_count:
                raise ValueError(f"PDF validation failed: expected {page_count} pages, got {validation_page_count}")

            logger.info(f"PDF has correct page count: {validation_page_count}")

            # Verify each page is not blank
            for page_idx in range(validation_page_count):
                val_page = validation_pdf[page_idx]
                # Render page to check if it's blank
                val_pix = val_page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
                val_img_data = val_pix.tobytes("png")
                val_img = cv2.imdecode(np.frombuffer(val_img_data, np.uint8), cv2.IMREAD_COLOR)

                if val_img is None:
                    raise ValueError(f"Page {page_idx} rendered as None - PDF may be corrupt")

                # Check if image is completely white/blank
                mean_pixel = np.mean(val_img)
                std_pixel = np.std(val_img)

                logger.info(f"Page {page_idx} validation: mean={mean_pixel:.2f}, std={std_pixel:.2f}")

                # If std is near zero and mean is high, page is likely blank
                if std_pixel < 1.0 and mean_pixel > 250:
                    raise ValueError(f"Page {page_idx} appears to be blank (mean={mean_pixel:.2f}, std={std_pixel:.2f})")

                # Save validation render for comparison
                validation_render_path = ENHANCED_DIR / f"{document_id}_page_{page_idx}_validation.png"
                cv2.imwrite(str(validation_render_path), val_img)
                logger.info(f"Saved validation render to {validation_render_path}")

            validation_pdf.close()
            logger.info("PDF validation passed - all pages contain visible content")

        except Exception as e:
            logger.error(f"PDF validation failed: {e}", exc_info=True)
            output_pdf.close()
            pdf_doc.close()
            # Delete the invalid PDF
            if enhanced_path.exists():
                enhanced_path.unlink()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Enhanced PDF validation failed: {str(e)}"
            )

        output_pdf.close()
        pdf_doc.close()

        file_type = "application/pdf"
    else:
        # Enhancement not beneficial - don't process remaining pages
        logger.info(f"Enhancement rejected for {document_id}: {validation_message}")
        enhanced_filename = f"{document_id}_enhanced.pdf"  # Name it but don't create
        file_type = "application/pdf"
        pdf_doc.close()

    # Calculate overall scores
    enhanced_score, enhanced_classification = calculate_overall_score(enhanced_metrics)

    try:
        enhanced_defect_predictions = predict_defects(enhanced_first)
        enhanced_ml_available = True
    except DefectPredictionModelError:
        enhanced_defect_predictions = []
        enhanced_ml_available = False

    enhanced_report = DocumentAnalysisResponse(
        document_id=document_id,
        filename=enhanced_filename,
        file_type=file_type,
        status="analyzed",
        overall_score=enhanced_score,
        classification=enhanced_classification,
        metrics=enhanced_metrics,
        defect_predictions=enhanced_defect_predictions,
        ml_model_available=enhanced_ml_available,
    )

    before = _comparison(original_report)
    after = _comparison(enhanced_report)
    improvement = ImprovementReport(
        absolute_score_change=round(after.quality - before.quality, 2),
        sharpness_change=round(after.sharpness - before.sharpness, 2),
        brightness_change=round(after.brightness - before.brightness, 2),
        contrast_change=round(after.contrast - before.contrast, 2),
        noise_change=round(after.noise - before.noise, 2),
        skew_change=round(after.skew - before.skew, 2),
        improved=is_valid,
    )

    final_recommendation = "recommended" if is_valid else "review_required"

    # Add validation and page count info
    operations_with_validation = enhancement_result.operations.copy()
    operations_with_validation.append({
        "name": "pdf_processing",
        "applied": True,
        "parameters": {
            "page_count": page_count,
            "pages_processed": len(enhanced_images) if is_valid else 1,
        }
    })
    operations_with_validation.append({
        "name": "validation",
        "applied": True,
        "result": {
            "is_valid": is_valid,
            "message": validation_message,
            "details": validation_details,
        }
    })

    response = EnhancementResponse(
        document_id=document_id,
        status="enhanced" if is_valid else "enhancement_rejected",
        original_quality=original_report,
        enhanced_quality=enhanced_report,
        before=before,
        after=after,
        improvement=improvement,
        operations=[EnhancementOperation.model_validate(operation) for operation in operations_with_validation],
        enhanced_file=EnhancedFile(
            available=is_valid,
            preview_url=f"/api/documents/{document_id}/enhanced-preview" if is_valid else None,
            download_url=f"/api/documents/{document_id}/enhanced" if is_valid else None,
            filename=enhanced_filename if is_valid else None,
        ),
        final_recommendation=final_recommendation,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    persisted = response.model_dump()
    (REPORTS_DIR / f"{document_id}_enhancement.json").write_text(json.dumps(persisted, indent=2), encoding="utf-8")
    return response


@router.get("/{document_id}/enhanced")
async def download_enhanced_document(document_id: str) -> FileResponse:
    """Download enhanced document (PDF or PNG based on original format).

    Sets Content-Disposition: attachment to trigger browser download.
    """
    if not _valid_document_id(document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Enhanced document not found.")

    # Check for PDF first
    enhanced_pdf_path = ENHANCED_DIR / f"{document_id}_enhanced.pdf"
    if enhanced_pdf_path.is_file():
        return FileResponse(
            enhanced_pdf_path,
            media_type="application/pdf",
            filename=enhanced_pdf_path.name,
            headers={"Content-Disposition": "attachment; filename=document_enhanced.pdf"}
        )

    # Fall back to PNG
    enhanced_png_path = ENHANCED_DIR / f"{document_id}_enhanced.png"
    if enhanced_png_path.is_file():
        return FileResponse(
            enhanced_png_path,
            media_type="image/png",
            filename=enhanced_png_path.name,
            headers={"Content-Disposition": "attachment; filename=document_enhanced.png"}
        )

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Enhanced document not found.")


@router.get("/{document_id}/enhanced-preview")
async def preview_enhanced_document(document_id: str) -> FileResponse:
    """Preview enhanced document inline (no download).

    Sets Content-Disposition: inline to display in browser.
    """
    if not _valid_document_id(document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Enhanced document not found.")

    # Check for PDF first
    enhanced_pdf_path = ENHANCED_DIR / f"{document_id}_enhanced.pdf"
    if enhanced_pdf_path.is_file():
        return FileResponse(
            enhanced_pdf_path,
            media_type="application/pdf",
            headers={"Content-Disposition": "inline"}
        )

    # Fall back to PNG
    enhanced_png_path = ENHANCED_DIR / f"{document_id}_enhanced.png"
    if enhanced_png_path.is_file():
        return FileResponse(
            enhanced_png_path,
            media_type="image/png",
            headers={"Content-Disposition": "inline"}
        )

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Enhanced document not found.")


@router.get("/{document_id}/original")
async def get_original_document(document_id: str) -> FileResponse:
    """Serve the original uploaded document image.

    Args:
        document_id: ID of the uploaded document

    Returns:
        FileResponse with the original document image

    Raises:
        404: Document not found
    """
    source_path = _find_source_path(document_id)
    if source_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Original document not found."
        )

    # Determine media type based on extension
    extension = source_path.suffix.lower()
    media_type = _mime_type_for_extension(extension)

    return FileResponse(source_path, media_type=media_type, filename=source_path.name)


@router.get("/{document_id}/preview")
async def get_document_preview(
    document_id: str,
    page: int = Query(1, ge=1, description="Page number (1-indexed)")
) -> Response:
    """Serve a rendered PNG preview of the document page.

    For PDF documents, renders the requested page as PNG.
    For image documents, returns the original image.

    Args:
        document_id: ID of the uploaded document
        page: Page number (1-indexed, default=1)

    Returns:
        Response with image/png content

    Raises:
        404: Document not found
        422: Invalid page number or rendering failed
    """
    source_path = _find_source_path(document_id)
    if source_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )

    extension = source_path.suffix.lower()

    # Handle PDF documents - render to PNG
    if extension == ".pdf":
        try:
            pdf_doc = fitz.open(str(source_path))
            page_count = len(pdf_doc)

            # Validate page number
            if page < 1 or page > page_count:
                pdf_doc.close()
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid page number. Document has {page_count} page(s)."
                )

            # Render page to PNG (0-indexed)
            pdf_page = pdf_doc[page - 1]

            # Render at 2x resolution for better quality
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            pix = pdf_page.get_pixmap(matrix=mat, alpha=False)

            # Convert to PNG bytes
            png_bytes = pix.tobytes("png")
            pdf_doc.close()

            return Response(content=png_bytes, media_type="image/png")

        except fitz.FileDataError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to read PDF: {exc}"
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to render PDF page: {exc}"
            ) from exc

    # Handle image documents - return as PNG
    elif extension in {".jpg", ".jpeg", ".png"}:
        try:
            # Read image using OpenCV
            image = cv2.imread(str(source_path))
            if image is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Failed to read image file."
                )

            # Encode as PNG
            success, buffer = cv2.imencode('.png', image)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to encode image as PNG."
                )

            png_bytes = buffer.tobytes()
            return Response(content=png_bytes, media_type="image/png")

        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process image: {exc}"
            ) from exc

    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file format: {extension}"
        )


@router.post("/{document_id}/layout", response_model=LayoutDetectionResponse)
async def detect_document_layout(document_id: str) -> LayoutDetectionResponse:
    """Detect document layout elements using Faster R-CNN.

    Identifies document regions such as text blocks, tables, figures, headers,
    footers, captions, and other layout elements with bounding boxes.

    Args:
        document_id: ID of a previously uploaded document

    Returns:
        LayoutDetectionResponse with per-page detections and bounding boxes

    Raises:
        404: Document not found
        422: Document processing error
        500: Layout detection failed
    """
    # Find the uploaded source document
    source_path = _find_source_path(document_id)
    if source_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )

    source_content = source_path.read_bytes()
    extension = source_path.suffix.lower()
    safe_filename = Path(source_path.name).name

    try:
        # Load the document image(s)
        # For now, process first page only (multi-page support can be added later)
        image = image_from_upload(source_content, extension)

        # Get image dimensions
        image_height, image_width = image.shape[:2]

        # Run layout detection
        try:
            detections = detect_layout(
                image,
                confidence_threshold=LAYOUT_CONFIDENCE_THRESHOLD,
                nms_threshold=LAYOUT_NMS_IOU_THRESHOLD,
            )
            model_available = True
            error_message = None

            # Convert detections to schema objects
            layout_detections = [
                LayoutDetection(
                    label=det["label"],
                    label_id=det["label_id"],
                    confidence=det["confidence"],
                    bbox=BoundingBox(**det["bbox"])
                )
                for det in detections
            ]

            # Create page detection result
            page_result = PageLayoutDetections(
                page_number=1,
                image_width=image_width,
                image_height=image_height,
                detections=layout_detections,
                detection_count=len(layout_detections)
            )

            pages = [page_result]
            status_value = "detected" if len(detections) > 0 else "no_detections"

        except LayoutModelError as exc:
            logger.warning("Layout detection model unavailable: %s", exc)
            model_available = False
            error_message = str(exc)
            pages = []
            status_value = "model_unavailable"

        # Build response
        response = LayoutDetectionResponse(
            document_id=document_id,
            filename=safe_filename,
            status=status_value,
            model_available=model_available,
            confidence_threshold=LAYOUT_CONFIDENCE_THRESHOLD,
            nms_threshold=LAYOUT_NMS_IOU_THRESHOLD,
            pages=pages,
            total_detections=sum(page.detection_count for page in pages),
            error=error_message
        )

        # Save layout detection report
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / f"{document_id}_layout.json"
        report_path.write_text(
            response.model_dump_json(indent=2),
            encoding="utf-8"
        )

        return response

    except DocumentProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc)
        ) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("Layout detection failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Layout detection failed."
        ) from exc


def _find_source_path(document_id: str) -> Path | None:
    if not _valid_document_id(document_id):
        return None
    candidates = [path for path in UPLOAD_DIR.glob(f"{document_id}.*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".pdf"}]
    return candidates[0] if len(candidates) == 1 else None


def _valid_document_id(document_id: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{8,64}", document_id))


def _load_or_create_report(document_id: str, source_path: Path, image) -> DocumentAnalysisResponse:
    report_path = PROCESSED_DIR / f"{document_id}.json"
    if report_path.is_file():
        return DocumentAnalysisResponse.model_validate_json(report_path.read_text(encoding="utf-8"))
    metrics = analyze_image(image)
    score, classification = calculate_overall_score(metrics)
    return DocumentAnalysisResponse(
        document_id=document_id,
        filename=source_path.name,
        file_type=_mime_type_for_extension(source_path.suffix.lower()),
        status="analyzed",
        overall_score=score,
        classification=classification,
        metrics=metrics,
    )


def _comparison(report: DocumentAnalysisResponse) -> QualityComparison:
    metrics = report.metrics
    return QualityComparison(
        quality=report.overall_score,
        sharpness=metrics.blur.raw_score,
        brightness=metrics.brightness.mean_brightness,
        contrast=metrics.contrast.raw_score,
        noise=metrics.noise.raw_score,
        skew=abs(metrics.skew.angle),
    )


def _mime_type_for_extension(extension: str) -> str:
    return "application/pdf" if extension == ".pdf" else "image/png" if extension == ".png" else "image/jpeg"
