from pathlib import Path

from app.core.config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_IMAGE_MIME_TYPES,
    ALLOWED_PDF_MIME_TYPES,
    MAX_UPLOAD_SIZE_BYTES,
)


class FileValidationError(ValueError):
    pass


def validate_upload_metadata(filename: str | None, content_type: str | None, size: int) -> str:
    if not filename or not filename.strip():
        raise FileValidationError("An upload is required.")

    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise FileValidationError("Unsupported file type. Use JPG, JPEG, PNG, or PDF.")

    if size == 0:
        raise FileValidationError("The uploaded file is empty.")
    if size > MAX_UPLOAD_SIZE_BYTES:
        raise FileValidationError("The uploaded file exceeds the 10 MB size limit.")

    if content_type:
        allowed_mime_types = ALLOWED_PDF_MIME_TYPES if extension == ".pdf" else ALLOWED_IMAGE_MIME_TYPES
        if content_type.lower() not in allowed_mime_types:
            raise FileValidationError("The file content type does not match its extension.")

    return extension


def validate_file_signature(extension: str, content: bytes) -> None:
    if extension == ".pdf" and not content.startswith(b"%PDF"):
        raise FileValidationError("The uploaded PDF is invalid or corrupted.")
    if extension in {".jpg", ".jpeg"} and not content.startswith(b"\xff\xd8\xff"):
        raise FileValidationError("The uploaded JPEG is invalid or corrupted.")
    if extension == ".png" and not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise FileValidationError("The uploaded PNG is invalid or corrupted.")
