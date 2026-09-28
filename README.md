# DocClear AI

DocClear AI is a full-stack deep learning project for intelligent document quality assessment and enhancement.

## Initial architecture

- `frontend/`: React + Vite frontend with Tailwind CSS configured.
- `backend/`: FastAPI service foundation.
- `ml/`: Modular dataset, experimentation, training, evaluation, and inference workspace.
- `docs/`: Architecture, API, model, and project documentation.

## Phase 2 functionality

Phase 2 provides a functional document upload and baseline quality assessment pipeline:

- `POST /api/documents/analyze` accepts JPG, JPEG, PNG, and PDF files up to 10 MB.
- PDFs are preserved unchanged and only their first page is converted for analysis.
- Quality metrics include blur, brightness, contrast, noise, skew, document boundary/cropping, and resolution.
- Results are returned as a structured quality report with an overall score and classification.
- Uploaded originals and generated reports are kept in separate directories and are excluded from version control.

Phase 2 uses classical computer-vision quality metrics as a baseline. A learned deep-learning quality assessment model will be introduced in a later phase.

The Phase 2 baseline does not perform enhancement, deblurring, super-resolution, OCR, authentication, database persistence, or multi-page PDF analysis. Its thresholds and scoring weights are initial engineering values and must be tuned against a validation dataset before production use.

## Phase 3 functionality

Phase 3 adds a conservative enhancement workflow through `POST /api/documents/{document_id}/enhance`:

1. Reuse the Phase 2 quality report.
2. Select operations based on measured degradation.
3. Generate an enhanced derivative without changing the original.
4. Re-analyze the derivative.
5. Compare quality metrics before and after.
6. Recommend the derivative only when its overall score strictly improves.

The deterministic baseline supports illumination correction, CLAHE contrast enhancement, conservative denoising, deskew, reliable-quadrilateral perspective correction, Lanczos upscaling, and unsharp-mask sharpening. Each operation records whether it ran, its parameters, or why it was skipped.

Phase 3 uses deterministic image restoration/enhancement techniques as a safe baseline. Learned restoration models can be introduced in a later research iteration.

The original uploaded document is never overwritten. Enhanced derivatives are stored separately under `backend/processed/enhanced/`, with explainability reports under `backend/processed/reports/`. The download endpoint is `GET /api/documents/{document_id}/enhanced` and is only advertised when the safety gate confirms improvement.

Example enhancement response:

```json
{
  "document_id": "generated-id",
  "status": "enhanced",
  "original_quality": {"overall_score": 43.2},
  "enhanced_quality": {"overall_score": 81.7},
  "improvement": {"absolute_score_change": 38.5, "improved": true},
  "operations": [{"name": "contrast_enhancement", "applied": true}],
  "enhanced_file": {"available": true, "download_url": "/api/documents/generated-id/enhanced"}
}
```

Quality deltas measure image-quality signals only; they do not prove semantic correctness or that document content has been reconstructed.

## Phase 4 OCR readability validation

Phase 4 adds a pluggable local OCR service, preferring PaddleOCR on CPU. OCR runs once on the original first-page image and once on the enhanced first-page image. The service returns confidence, detected word count, line count, and detections behind an engine interface so another OCR implementation can be substituted later.

The enhancement response now includes `ocr_validation` with confidence, word-count, line-count, readability-score, text-similarity, and content-preservation signals. The readability score is an initial engineering metric using 60% confidence improvement, 25% word-count improvement, and 15% line-count improvement. These weights are configurable and are not a universal readability standard.

There are two recommendation gates: image quality must improve, and OCR readability must improve while normalized text similarity remains above the configured preservation threshold. If OCR is unavailable or fails, the response is marked `ocr_validation_unavailable` and the enhanced derivative is not recommended. OCR text is returned only for the optional UI disclosure and is removed from persisted enhancement reports; it is never logged.

OCR confidence is used as an operational readability signal and does not establish document authenticity, validity, or factual correctness.

PaddleOCR is an optional local CPU dependency. Automated tests use a mocked engine when the local OCR runtime is unavailable. PDFs continue to support first-page-only analysis, enhancement, and OCR; multi-page OCR is not implemented.

### Local OCR Runtime

The main FastAPI backend remains on Python 3.14. PaddleOCR runs in the isolated Python 3.12 environment at `backend/.venv-ocr/` because PaddlePaddle CPU wheels are not available for Python 3.14 on this Windows setup.

Create the OCR environment with:

```powershell
py -3.12 -m venv backend\.venv-ocr
backend\.venv-ocr\Scripts\Activate.ps1
python --version
backend\.venv-ocr\Scripts\python.exe -m pip install --upgrade pip
backend\.venv-ocr\Scripts\python.exe -m pip install -r backend\ocr-requirements.txt
# Leave the isolated environment when finished:
deactivate
```

Verify the CPU runtime:

```powershell
backend\.venv-ocr\Scripts\python.exe -c "import paddle; import paddleocr; print(paddle.__version__); print(paddle.device.get_device()); print(paddleocr.__version__)"
```

The Python 3.14 backend communicates with this runtime through a fixed local subprocess boundary at `backend/app/services/ocr/paddle_runner.py`. It passes only generated temporary image paths under `backend/processed/ocr-input/`, invokes the known `backend/.venv-ocr/Scripts/python.exe` executable with `shell=False`, captures structured JSON, enforces a timeout, and removes the temporary input afterward. The OCR runner is not exposed as a public service.

Example response:

```json
{
  "document_id": "generated-id",
  "filename": "marksheet.jpg",
  "file_type": "image/jpeg",
  "status": "analyzed",
  "overall_score": 78.5,
  "classification": "acceptable",
  "metrics": {
    "blur": {"score": 72.4, "raw_score": 184.2, "status": "good"},
    "brightness": {"mean_brightness": 126.1, "normalized_score": 98.5, "status": "good", "condition": "acceptable"},
    "contrast": {"score": 68.0, "raw_score": 61.2, "status": "good"},
    "noise": {"score": 87.0, "raw_score": 5.2, "status": "good"},
    "skew": {"angle": 0.8, "status": "good"},
    "cropping": {"document_detected": true, "centered": true, "appears_cropped": false, "status": "good"},
    "resolution": {"width": 1280, "height": 920, "megapixels": 1.18, "score": 100, "status": "good"}
  }
}
```

## Development commands

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The backend health check is available at `http://127.0.0.1:8000/health`.

To run the backend tests:

```powershell
cd backend
pytest
```
