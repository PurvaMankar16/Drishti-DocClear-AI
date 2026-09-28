from typing import Literal

from pydantic import BaseModel, Field


MetricStatus = Literal["good", "warning", "poor"]


class BlurMetric(BaseModel):
    score: float = Field(ge=0, le=100)
    raw_score: float = Field(ge=0)
    status: MetricStatus


class BrightnessMetric(BaseModel):
    mean_brightness: float = Field(ge=0, le=255)
    normalized_score: float = Field(ge=0, le=100)
    status: MetricStatus
    condition: Literal["too_dark", "acceptable", "too_bright"]


class ContrastMetric(BaseModel):
    score: float = Field(ge=0, le=100)
    raw_score: float = Field(ge=0)
    status: MetricStatus


class NoiseMetric(BaseModel):
    score: float = Field(ge=0, le=100)
    raw_score: float = Field(ge=0)
    status: MetricStatus


class SkewMetric(BaseModel):
    angle: float
    status: MetricStatus


class CroppingMetric(BaseModel):
    document_detected: bool
    centered: bool
    appears_cropped: bool
    status: MetricStatus


class ResolutionMetric(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    megapixels: float = Field(gt=0)
    score: float = Field(ge=0, le=100)
    status: MetricStatus


class QualityMetrics(BaseModel):
    blur: BlurMetric
    brightness: BrightnessMetric
    contrast: ContrastMetric
    noise: NoiseMetric
    skew: SkewMetric
    cropping: CroppingMetric
    resolution: ResolutionMetric


class QualityReport(BaseModel):
    document_id: str
    filename: str
    file_type: str
    status: Literal["analyzed"]
    overall_score: float = Field(ge=0, le=100)
    classification: Literal["poor", "needs_enhancement", "acceptable"]
    metrics: QualityMetrics
