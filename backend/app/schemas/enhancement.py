from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.quality import QualityReport


class EnhancementOperation(BaseModel):
    name: str
    applied: bool
    parameters: dict[str, Any] | None = None
    reason: str | None = None


class QualityComparison(BaseModel):
    quality: float = Field(ge=0, le=100)
    sharpness: float = Field(ge=0)
    brightness: float = Field(ge=0, le=255)
    contrast: float = Field(ge=0)
    noise: float = Field(ge=0)
    skew: float


class ImprovementReport(BaseModel):
    absolute_score_change: float
    sharpness_change: float
    brightness_change: float
    contrast_change: float
    noise_change: float
    skew_change: float
    improved: bool


class EnhancedFile(BaseModel):
    available: bool
    download_url: str | None = None
    preview_url: str | None = None
    filename: str | None = None


class EnhancementResponse(BaseModel):
    """Complete enhancement response for a document."""
    document_id: str
    status: Literal["enhanced", "enhancement_not_beneficial", "enhancement_rejected"] = Field(
        description="Enhancement status: enhanced=accepted, enhancement_rejected=validation failed"
    )
    original_quality: QualityReport
    enhanced_quality: QualityReport
    before: QualityComparison
    after: QualityComparison
    improvement: ImprovementReport
    operations: list[EnhancementOperation]
    enhanced_file: EnhancedFile
    final_recommendation: Literal["recommended", "review_required"]
