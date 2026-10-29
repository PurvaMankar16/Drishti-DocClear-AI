from pydantic import BaseModel, Field


class OCRDetection(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1)
    bbox: list[list[float]]


class OCRResult(BaseModel):
    engine: str
    text: str
    confidence: float = Field(ge=0, le=1)
    word_count: int = Field(ge=0)
    line_count: int = Field(ge=0)
    detections: list[OCRDetection] = Field(default_factory=list)
