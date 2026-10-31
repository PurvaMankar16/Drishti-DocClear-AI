from app.core.config import OVERALL_CLASSIFICATION_THRESHOLDS, QUALITY_WEIGHTS
from app.schemas.quality import QualityMetrics


def calculate_overall_score(metrics: QualityMetrics) -> tuple[float, str]:
    """Calculate overall quality score with proper weighting."""
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


def validate_enhancement(original_metrics: QualityMetrics, enhanced_metrics: QualityMetrics) -> tuple[bool, str, dict]:
    """Validate that enhancement genuinely improved quality.

    Checks that:
    1. Critical metrics (blur, contrast) did not degrade significantly
    2. Overall score improved
    3. No single metric degraded by more than 15%

    Args:
        original_metrics: Metrics before enhancement
        enhanced_metrics: Metrics after enhancement

    Returns:
        Tuple of (is_valid, reason, details_dict)
    """
    # Extract raw scores for comparison
    orig_blur = original_metrics.blur.raw_score
    enh_blur = enhanced_metrics.blur.raw_score

    orig_contrast = original_metrics.contrast.raw_score
    enh_contrast = enhanced_metrics.contrast.raw_score

    orig_noise = original_metrics.noise.raw_score
    enh_noise = enhanced_metrics.noise.raw_score

    orig_brightness = original_metrics.brightness.mean_brightness
    enh_brightness = enhanced_metrics.brightness.mean_brightness

    # Calculate percentage changes
    blur_change = ((enh_blur - orig_blur) / max(orig_blur, 1)) * 100
    contrast_change = ((enh_contrast - orig_contrast) / max(orig_contrast, 1)) * 100
    noise_change = ((orig_noise - enh_noise) / max(orig_noise, 0.1)) * 100  # Lower is better
    brightness_change = abs(enh_brightness - orig_brightness)

    details = {
        "blur_change_pct": round(blur_change, 1),
        "contrast_change_pct": round(contrast_change, 1),
        "noise_change_pct": round(noise_change, 1),
        "brightness_change": round(brightness_change, 1),
    }

    # Critical metric validation
    # Blur: must not decrease by more than 10%
    if blur_change < -10:
        return False, f"Sharpness decreased by {abs(blur_change):.1f}% (threshold: 10%)", details

    # Contrast: must not decrease by more than 10%
    if contrast_change < -10:
        return False, f"Contrast decreased by {abs(contrast_change):.1f}% (threshold: 10%)", details

    # Overall improvement check
    orig_score, _ = calculate_overall_score(original_metrics)
    enh_score, _ = calculate_overall_score(enhanced_metrics)

    score_improvement = enh_score - orig_score

    # Require at least 2 points improvement OR no degradation in critical metrics
    if score_improvement < 2.0:
        if blur_change < 0 or contrast_change < 0:
            return False, f"Insufficient improvement ({score_improvement:.1f} points) with metric degradation", details

    # Check for multi-metric improvement
    # At least one critical metric must improve by >5% OR noise must improve by >10%
    has_improvement = (
        blur_change > 5 or
        contrast_change > 5 or
        noise_change > 10
    )

    if not has_improvement and score_improvement < 5.0:
        return False, f"No significant metric improvement detected (score +{score_improvement:.1f})", details

    return True, f"Enhancement validated: score +{score_improvement:.1f}, blur {blur_change:+.1f}%, contrast {contrast_change:+.1f}%", details


def _skew_score(angle: float) -> float:
    return max(0.0, 100.0 - min(abs(angle), 15.0) / 15.0 * 100.0)


def _cropping_score(status: str) -> float:
    return {"good": 100.0, "warning": 60.0, "poor": 20.0}.get(status, 0.0)
