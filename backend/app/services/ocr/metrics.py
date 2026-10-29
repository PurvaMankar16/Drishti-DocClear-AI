from difflib import SequenceMatcher
from typing import Literal

from app.core.config import OCR_CONTENT_SIMILARITY_THRESHOLD, OCR_READABILITY_WEIGHTS
from app.schemas.ocr import OCRResult
from app.schemas.readability import OCRSnapshot, ReadabilityImprovement, ReadabilityValidation


def compare_ocr(original: OCRResult, enhanced: OCRResult) -> ReadabilityValidation:
    """Compare OCR results between original and enhanced documents.

    Considers text similarity, word count / coverage, and recognition confidence
    to distinguish between:
    - validated: Enhancement improved readability while preserving text content.
    - rejected: Enhancement degraded text readability or altered text content.
    - inconclusive: Insufficient text recognized in both versions to measure readability reliably.
    """
    confidence_delta = enhanced.confidence - original.confidence
    word_delta = enhanced.word_count - original.word_count
    line_delta = enhanced.line_count - original.line_count

    # Empty or nearly empty recognition results:
    # If both images produced no or almost no recognizable text, OCR cannot form a reliable basis.
    if original.word_count <= 1 and enhanced.word_count <= 1:
        return ReadabilityValidation(
            available=True,
            status="inconclusive",
            original=OCRSnapshot(
                confidence=original.confidence,
                word_count=original.word_count,
                line_count=original.line_count,
            ),
            enhanced=OCRSnapshot(
                confidence=enhanced.confidence,
                word_count=enhanced.word_count,
                line_count=enhanced.line_count,
            ),
            improvement=ReadabilityImprovement(
                confidence_delta=round(confidence_delta, 4),
                word_count_delta=word_delta,
                line_count_delta=line_delta,
                readability_score=0.0,
                improved=False,
            ),
            text_similarity=None,
            content_preservation_warning=False,
            original_text=original.text,
            enhanced_text=enhanced.text,
            error="OCR could not reliably validate this document due to insufficient recognizable text.",
        )

    # Special case: original was unreadable (<= 1 word) but enhanced revealed substantial confident text
    recovered_from_unreadable = (
        original.word_count <= 1
        and enhanced.word_count >= 3
        and enhanced.confidence >= 0.50
    )

    similarity = text_similarity(original.text, enhanced.text)
    preservation_warning = (
        False if recovered_from_unreadable else (similarity < OCR_CONTENT_SIMILARITY_THRESHOLD)
    )

    # Text loss check: enhancement caused significant reduction in detected words
    text_lost = (
        (original.word_count >= 3 and enhanced.word_count < original.word_count * 0.8)
        or (original.word_count > 0 and enhanced.word_count == 0)
    )

    # Confidence drop: confidence significantly decreased without text count gain
    confidence_dropped = confidence_delta < -0.05 and word_delta <= 0

    confidence_component = max(0.0, min(1.0, confidence_delta)) * 100
    word_component = _relative_gain(original.word_count, enhanced.word_count)
    line_component = _relative_gain(original.line_count, enhanced.line_count)
    raw_readability_score = (
        confidence_component * OCR_READABILITY_WEIGHTS["confidence"]
        + word_component * OCR_READABILITY_WEIGHTS["words"]
        + line_component * OCR_READABILITY_WEIGHTS["lines"]
    )

    status: Literal["validated", "rejected", "inconclusive", "unavailable"]
    improved: bool

    if preservation_warning or text_lost or confidence_dropped:
        status = "rejected"
        improved = False
        readability_score = 0.0
    elif (confidence_delta > 0 or word_delta > 0 or line_delta > 0 or recovered_from_unreadable) and not preservation_warning:
        status = "validated"
        improved = True
        readability_score = round(max(0.0, min(100.0, raw_readability_score)), 2)
    else:
        # Text preserved, but no measurable readability gain
        status = "rejected"
        improved = False
        readability_score = 0.0

    return ReadabilityValidation(
        available=True,
        status=status,
        original=OCRSnapshot(
            confidence=original.confidence,
            word_count=original.word_count,
            line_count=original.line_count,
        ),
        enhanced=OCRSnapshot(
            confidence=enhanced.confidence,
            word_count=enhanced.word_count,
            line_count=enhanced.line_count,
        ),
        improvement=ReadabilityImprovement(
            confidence_delta=round(confidence_delta, 4),
            word_count_delta=word_delta,
            line_count_delta=line_delta,
            readability_score=readability_score,
            improved=improved,
        ),
        text_similarity=round(similarity, 4) if not recovered_from_unreadable else 1.0,
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
