from app.core.config import OVERALL_CLASSIFICATION_THRESHOLDS, QUALITY_WEIGHTS
from app.schemas.quality import QualityMetrics


def calculate_overall_score(metrics: QualityMetrics) -> tuple[float, str]:
    scores = {
        "blur": metrics.blur.score,
        "brightness": metrics.brightness.normalized_score,
        "contrast": metrics.contrast.score,
        "noise": metrics.noise.score,
        "skew": _skew_score(metrics.skew.angle),
        "cropping": _cropping_score(metrics.cropping.status),
        "resolution": metrics.resolution.score,
    }
    score = sum(scores[name] * QUALITY_WEIGHTS[name] for name in QUALITY_WEIGHTS)
    score = round(max(0.0, min(100.0, score)), 2)
    thresholds = OVERALL_CLASSIFICATION_THRESHOLDS
    classification = "acceptable" if score >= thresholds["acceptable"] else "needs_enhancement" if score >= thresholds["needs_enhancement"] else "poor"
    return score, classification


def _skew_score(angle: float) -> float:
    return max(0.0, 100.0 - min(abs(angle), 15.0) / 15.0 * 100.0)


def _cropping_score(status: str) -> float:
    return {"good": 100.0, "warning": 60.0, "poor": 20.0}.get(status, 0.0)
