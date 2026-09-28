import json
import secrets
import re
from pathlib import Path

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from app.core.config import ENHANCED_DIR, PROCESSED_DIR, REPORTS_DIR, UPLOAD_DIR
from app.schemas.enhancement import (
    EnhancedFile,
    EnhancementOperation,
    EnhancementResponse,
    ImprovementReport,
    QualityComparison,
)
from app.schemas.document import DocumentAnalysisResponse
from app.services.document_preprocessing import DocumentProcessingError, image_from_upload
from app.services.enhancement.pipeline import enhance_document
from app.services.quality_analysis import analyze_image
from app.services.quality_scoring import calculate_overall_score
from app.services.ocr.metrics import compare_ocr
from app.services.ocr.paddle_engine import OCREngineUnavailableError
from app.services.ocr.service import OCRService
from app.utils.file_validation import FileValidationError, validate_file_signature, validate_upload_metadata


router = APIRouter(prefix="/api/documents", tags=["documents"])
ocr_service = OCRService()


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
        report = DocumentAnalysisResponse(
            document_id=document_id,
            filename=safe_original_name,
            file_type=file.content_type or _mime_type_for_extension(extension),
            status="analyzed",
            overall_score=overall_score,
            classification=classification,
            metrics=metrics,
        )
        (PROCESSED_DIR / f"{document_id}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return report
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Document analysis failed.") from exc


@router.post("/{document_id}/enhance", response_model=EnhancementResponse)
async def enhance_document_endpoint(document_id: str) -> EnhancementResponse:
    source_path = _find_source_path(document_id)
    if source_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    source_content = source_path.read_bytes()
    try:
        source_image = image_from_upload(source_content, source_path.suffix.lower())
        original_report = _load_or_create_report(document_id, source_path, source_image)
        original_ocr = None
        ocr_error = None
        try:
            original_ocr = ocr_service.recognize(source_image)
        except (OCREngineUnavailableError, RuntimeError):
            ocr_error = "OCR validation unavailable."
        enhancement_result = enhance_document(source_image, original_report.metrics)
        enhanced_image = enhancement_result.image
        enhanced_metrics = analyze_image(enhanced_image)
        enhanced_score, enhanced_classification = calculate_overall_score(enhanced_metrics)
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
            improved=enhanced_score > original_report.overall_score,
        )
        is_improved = improvement.improved
        if original_ocr is None:
            ocr_validation = _unavailable_readability(ocr_error or "OCR validation unavailable.")
            final_recommendation = "ocr_validation_unavailable"
        else:
            try:
                enhanced_ocr = ocr_service.recognize(enhanced_image)
                ocr_validation = compare_ocr(original_ocr, enhanced_ocr)
                final_recommendation = "recommended" if is_improved and ocr_validation.improvement and ocr_validation.improvement.improved else "review_required"
            except (OCREngineUnavailableError, RuntimeError):
                ocr_validation = _unavailable_readability("OCR validation unavailable.")
                final_recommendation = "ocr_validation_unavailable"
        recommended = final_recommendation == "recommended"
        response = EnhancementResponse(
            document_id=document_id,
            status="enhanced" if is_improved else "enhancement_not_beneficial",
            original_quality=original_report,
            enhanced_quality=enhanced_report,
            before=before,
            after=after,
            improvement=improvement,
            operations=[EnhancementOperation.model_validate(operation) for operation in enhancement_result.operations],
            enhanced_file=EnhancedFile(
                available=recommended,
                download_url=f"/api/documents/{document_id}/enhanced" if recommended else None,
                filename=enhanced_filename if recommended else None,
            ),
            ocr_validation=ocr_validation,
            final_recommendation=final_recommendation,
        )
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        persisted = response.model_dump()
        persisted["ocr_validation"]["original_text"] = None
        persisted["ocr_validation"]["enhanced_text"] = None
        (REPORTS_DIR / f"{document_id}_enhancement.json").write_text(json.dumps(persisted, indent=2), encoding="utf-8")
        return response
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Document enhancement failed.") from exc


@router.get("/{document_id}/enhanced")
async def download_enhanced_document(document_id: str) -> FileResponse:
    if not _valid_document_id(document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Enhanced document not found.")
    enhanced_path = ENHANCED_DIR / f"{document_id}_enhanced.png"
    if not enhanced_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Enhanced document not found.")
    return FileResponse(enhanced_path, media_type="image/png", filename=enhanced_path.name)


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


def _unavailable_readability(error: str):
    from app.schemas.readability import ReadabilityValidation

    return ReadabilityValidation(available=False, error=error)


def _mime_type_for_extension(extension: str) -> str:
    return "application/pdf" if extension == ".pdf" else "image/png" if extension == ".png" else "image/jpeg"
