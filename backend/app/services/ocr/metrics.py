from difflib import SequenceMatcher

from app.core.config import OCR_CONTENT_SIMILARITY_THRESHOLD, OCR_READABILITY_WEIGHTS
from app.schemas.ocr import OCRResult
from app.schemas.readability import OCRSnapshot, ReadabilityImprovement, ReadabilityValidation


def compare_ocr(original: OCRResult, enhanced: OCRResult) -> ReadabilityValidation:
    confidence_delta = enhanced.confidence - original.confidence
    word_delta = enhanced.word_count - original.word_count
    line_delta = enhanced.line_count - original.line_count
    confidence_component = max(0.0, min(1.0, confidence_delta)) * 100
    word_component = _relative_gain(original.word_count, enhanced.word_count)
    line_component = _relative_gain(original.line_count, enhanced.line_count)
    readability_score = (
        confidence_component * OCR_READABILITY_WEIGHTS["confidence"]
        + word_component * OCR_READABILITY_WEIGHTS["words"]
        + line_component * OCR_READABILITY_WEIGHTS["lines"]
    )
    similarity = text_similarity(original.text, enhanced.text)
    preservation_warning = similarity < OCR_CONTENT_SIMILARITY_THRESHOLD
    improved = (confidence_delta > 0 or word_delta > 0 or line_delta > 0) and not preservation_warning
    return ReadabilityValidation(
        available=True,
        original=OCRSnapshot(confidence=original.confidence, word_count=original.word_count, line_count=original.line_count),
        enhanced=OCRSnapshot(confidence=enhanced.confidence, word_count=enhanced.word_count, line_count=enhanced.line_count),
        improvement=ReadabilityImprovement(
            confidence_delta=round(confidence_delta, 4),
            word_count_delta=word_delta,
            line_count_delta=line_delta,
            readability_score=round(max(0.0, min(100.0, readability_score)), 2),
            improved=improved,
        ),
        text_similarity=round(similarity, 4),
        content_preservation_warning=preservation_warning,
        original_text=original.text,
        enhanced_text=enhanced.text,
    )


def text_similarity(original_text: str, enhanced_text: str) -> float:
    normalized_original = " ".join(original_text.lower().split())
    normalized_enhanced = " ".join(enhanced_text.lower().split())
    return SequenceMatcher(None, normalized_original, normalized_enhanced).ratio()


def _relative_gain(before: int, after: int) -> float:
    if before <= 0:
        return 100.0 if after > 0 else 0.0
    return max(0.0, min(100.0, (after - before) / before * 100.0))
