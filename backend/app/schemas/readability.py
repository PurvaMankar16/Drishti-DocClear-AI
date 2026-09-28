from pydantic import BaseModel, Field


class OCRSnapshot(BaseModel):
    confidence: float = Field(ge=0, le=1)
    word_count: int = Field(ge=0)
    line_count: int = Field(ge=0)


class ReadabilityImprovement(BaseModel):
    confidence_delta: float
    word_count_delta: int
    line_count_delta: int
    readability_score: float = Field(ge=0, le=100)
    improved: bool


class ReadabilityValidation(BaseModel):
    available: bool
    original: OCRSnapshot | None = None
    enhanced: OCRSnapshot | None = None
    improvement: ReadabilityImprovement | None = None
    text_similarity: float | None = Field(default=None, ge=0, le=1)
    content_preservation_warning: bool = False
    original_text: str | None = None
    enhanced_text: str | None = None
    error: str | None = None
