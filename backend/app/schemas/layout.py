"""Pydantic schemas for document layout detection."""

from typing import Literal

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates in xyxy format."""
    x1: float = Field(ge=0, description="Left x coordinate")
    y1: float = Field(ge=0, description="Top y coordinate")
    x2: float = Field(ge=0, description="Right x coordinate")
    y2: float = Field(ge=0, description="Bottom y coordinate")


class LayoutDetection(BaseModel):
    """Single layout region detection."""
    label: str = Field(description="Layout element class name")
    label_id: int = Field(ge=1, le=11, description="Class ID (1-11)")
    confidence: float = Field(ge=0, le=1, description="Detection confidence score")
    bbox: BoundingBox = Field(description="Bounding box coordinates")


class PageLayoutDetections(BaseModel):
    """Layout detections for a single page."""
    page_number: int = Field(ge=1, description="Page number (1-indexed)")
    image_width: int = Field(gt=0, description="Original image width in pixels")
    image_height: int = Field(gt=0, description="Original image height in pixels")
    detections: list[LayoutDetection] = Field(default_factory=list, description="List of detected regions")
    detection_count: int = Field(ge=0, description="Number of detections after filtering")


class LayoutDetectionResponse(BaseModel):
    """Complete layout detection response for a document."""
    model_config = {"protected_namespaces": ()}  # Allow 'model_' prefix

    document_id: str = Field(description="Document identifier")
    filename: str = Field(description="Original filename")
    status: Literal["detected", "no_detections", "model_unavailable"] = Field(
        description="Detection status"
    )
    model_available: bool = Field(description="Whether the layout model was available")
    confidence_threshold: float = Field(ge=0, le=1, description="Applied confidence threshold")
    nms_threshold: float = Field(ge=0, le=1, description="Applied NMS IoU threshold")
    pages: list[PageLayoutDetections] = Field(default_factory=list, description="Per-page detections")
    total_detections: int = Field(ge=0, description="Total detections across all pages")
    error: str | None = Field(default=None, description="Error message if detection failed")


class LayoutDetectionError(RuntimeError):
    """Raised when layout detection model cannot be loaded or used."""
    pass
