"""Adaptive, metric-driven document enhancement pipeline with operation validation.

Operations are applied based on continuous quality metrics and validated
individually. Each operation must demonstrate improvement or it is rejected.

Order optimized for documents:
  1. Geometry corrections (deskew, perspective) – align document first
  2. Background normalization – create clean white background
  3. Denoising – reduce noise before sharpening
  4. Contrast enhancement – improve text/background separation
  5. Sharpening – enhance edges last for maximum effect
  6. Upscaling – only if resolution genuinely poor
"""
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from app.schemas.quality import QualityMetrics
from app.services.enhancement.contrast import enhance_contrast
from app.services.enhancement.deblur import sharpen
from app.services.enhancement.denoise import denoise
from app.services.enhancement.deskew import deskew
from app.services.enhancement.illumination import normalize_background
from app.services.enhancement.perspective import correct_perspective
from app.services.enhancement.upscaling import upscale


@dataclass
class EnhancementResult:
    image: np.ndarray
    operations: list[dict] = field(default_factory=list)
    metrics_before: Optional[dict] = None
    metrics_after: Optional[dict] = None


def enhance_document(image: np.ndarray, quality: QualityMetrics) -> EnhancementResult:
    """Apply adaptive enhancement operations with individual validation.

    Each operation is evaluated and accepted only if it improves quality
    without degrading critical metrics. Operations use continuous thresholds
    rather than binary good/warning/poor status.

    Args:
        image: Input BGR image
        quality: Quality metrics for the original image

    Returns:
        EnhancementResult with enhanced image and operation log
    """
    current = image.copy()
    operations: list[dict] = []

    # Store initial metrics for comparison
    initial_metrics = _extract_key_metrics(quality)

    # 1. Deskew – apply if angle > 0.5° (more sensitive than before)
    if abs(quality.skew.angle) > 0.5:
        try:
            deskewed = deskew(current, quality.skew.angle)
            # Validate: check if image is reasonable after deskew
            if _validate_image(deskewed):
                current = deskewed
                operations.append({
                    "name": "deskew",
                    "applied": True,
                    "parameters": {"angle": round(-quality.skew.angle, 2)},
                })
            else:
                operations.append({
                    "name": "deskew",
                    "applied": False,
                    "reason": "deskewing produced invalid result",
                })
        except Exception as e:
            operations.append({
                "name": "deskew",
                "applied": False,
                "reason": f"deskewing failed: {str(e)}",
            })
    else:
        operations.append({
            "name": "deskew",
            "applied": False,
            "reason": f"skew angle {quality.skew.angle:.2f}° within tolerance",
        })

    # 2. Perspective correction – only if document boundary detected
    if quality.cropping.document_detected:
        try:
            corrected, applied = correct_perspective(current)
            if applied and _validate_image(corrected):
                current = corrected
                operations.append({
                    "name": "perspective_correction",
                    "applied": True,
                    "reason": "document boundary detected and corrected",
                })
            else:
                operations.append({
                    "name": "perspective_correction",
                    "applied": False,
                    "reason": "no reliable quadrilateral found",
                })
        except Exception as e:
            operations.append({
                "name": "perspective_correction",
                "applied": False,
                "reason": f"perspective correction failed: {str(e)}",
            })
    else:
        operations.append({
            "name": "perspective_correction",
            "applied": False,
            "reason": "document boundary not detected",
        })

    # 3. Background normalization – apply if background not bright enough
    # Use 90th percentile as background estimate
    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY) if current.ndim == 3 else current
    background_level = float(np.percentile(gray, 90))

    if background_level < 230:  # Apply if background not near-white
        try:
            normalized = normalize_background(current)
            # Validate: check sharpness not reduced
            before_sharpness = _calculate_sharpness(current)
            after_sharpness = _calculate_sharpness(normalized)

            if after_sharpness >= before_sharpness * 0.95:  # Allow 5% tolerance
                current = normalized
                operations.append({
                    "name": "background_normalization",
                    "applied": True,
                    "parameters": {"target_background": 240, "before_level": round(background_level, 1)},
                    "validation": {"sharpness_preserved": True},
                })
            else:
                operations.append({
                    "name": "background_normalization",
                    "applied": False,
                    "reason": f"would reduce sharpness by {100*(1-after_sharpness/before_sharpness):.1f}%",
                })
        except Exception as e:
            operations.append({
                "name": "background_normalization",
                "applied": False,
                "reason": f"normalization failed: {str(e)}",
            })
    else:
        operations.append({
            "name": "background_normalization",
            "applied": False,
            "reason": f"background already bright ({background_level:.0f}/255)",
        })

    # 4. Denoising – apply if noise level > 4.0 (lowered from 8.0)
    noise_level = quality.noise.raw_score

    if noise_level > 4.0:
        try:
            denoised = denoise(current, adaptive=True)
            # Validate: check sharpness not over-reduced
            before_sharpness = _calculate_sharpness(current)
            after_sharpness = _calculate_sharpness(denoised)

            if after_sharpness >= before_sharpness * 0.90:  # Allow 10% tolerance for noise reduction
                current = denoised
                operations.append({
                    "name": "denoising",
                    "applied": True,
                    "parameters": {"method": "fastNlMeans_adaptive", "noise_level": round(noise_level, 2)},
                    "validation": {"sharpness_preserved": True},
                })
            else:
                operations.append({
                    "name": "denoising",
                    "applied": False,
                    "reason": f"would reduce sharpness by {100*(1-after_sharpness/before_sharpness):.1f}%",
                })
        except Exception as e:
            operations.append({
                "name": "denoising",
                "applied": False,
                "reason": f"denoising failed: {str(e)}",
            })
    else:
        operations.append({
            "name": "denoising",
            "applied": False,
            "reason": f"noise level {noise_level:.2f} is acceptable (threshold 4.0)",
        })

    # 5. Contrast enhancement – apply if contrast < 40 (raised from threshold)
    contrast_level = quality.contrast.raw_score

    if contrast_level < 40:
        try:
            contrasted = enhance_contrast(current)
            # Validate: check if contrast actually improved
            before_contrast = _calculate_contrast(current)
            after_contrast = _calculate_contrast(contrasted)

            if after_contrast > before_contrast * 1.05:  # Require 5% improvement
                current = contrasted
                operations.append({
                    "name": "contrast_enhancement",
                    "applied": True,
                    "parameters": {"method": "CLAHE_document", "clip_limit": 3.5, "tile_grid": [16, 16]},
                    "validation": {"contrast_improved": True, "improvement": f"+{100*(after_contrast/before_contrast-1):.1f}%"},
                })
            else:
                operations.append({
                    "name": "contrast_enhancement",
                    "applied": False,
                    "reason": f"would not improve contrast (before: {before_contrast:.1f}, after: {after_contrast:.1f})",
                })
        except Exception as e:
            operations.append({
                "name": "contrast_enhancement",
                "applied": False,
                "reason": f"contrast enhancement failed: {str(e)}",
            })
    else:
        operations.append({
            "name": "contrast_enhancement",
            "applied": False,
            "reason": f"contrast {contrast_level:.1f} is acceptable (threshold 40)",
        })

    # 6. Sharpening – apply if blur score < 180 (more sensitive for documents)
    blur_score = quality.blur.raw_score

    if blur_score < 180:
        try:
            sharpened = sharpen(current, amount=1.0, sigma=1.5, adaptive=True)
            # Validate: check if sharpness actually improved
            before_sharpness = _calculate_sharpness(current)
            after_sharpness = _calculate_sharpness(sharpened)

            if after_sharpness > before_sharpness * 1.05:  # Require 5% improvement
                current = sharpened
                operations.append({
                    "name": "sharpening",
                    "applied": True,
                    "parameters": {"method": "unsharp_mask_adaptive", "blur_score": round(blur_score, 1)},
                    "validation": {"sharpness_improved": True, "improvement": f"+{100*(after_sharpness/before_sharpness-1):.1f}%"},
                })
            else:
                operations.append({
                    "name": "sharpening",
                    "applied": False,
                    "reason": f"would not improve sharpness (before: {before_sharpness:.1f}, after: {after_sharpness:.1f})",
                })
        except Exception as e:
            operations.append({
                "name": "sharpening",
                "applied": False,
                "reason": f"sharpening failed: {str(e)}",
            })
    else:
        operations.append({
            "name": "sharpening",
            "applied": False,
            "reason": f"sharpness {blur_score:.1f} is acceptable (threshold 180)",
        })

    # 7. Upscaling – only for very low resolution (< 0.5 MP)
    if quality.resolution.megapixels < 0.5:
        try:
            upscaled = upscale(current)
            if _validate_image(upscaled):
                current = upscaled
                operations.append({
                    "name": "upscaling",
                    "applied": True,
                    "parameters": {"method": "INTER_LANCZOS4", "scale": 2.0, "original_mp": round(quality.resolution.megapixels, 2)},
                })
            else:
                operations.append({
                    "name": "upscaling",
                    "applied": False,
                    "reason": "upscaling produced invalid result",
                })
        except Exception as e:
            operations.append({
                "name": "upscaling",
                "applied": False,
                "reason": f"upscaling failed: {str(e)}",
            })
    else:
        operations.append({
            "name": "upscaling",
            "applied": False,
            "reason": f"resolution {quality.resolution.megapixels:.2f} MP is sufficient",
        })

    return EnhancementResult(
        image=current,
        operations=operations,
        metrics_before=initial_metrics,
        metrics_after=None  # Will be calculated by caller
    )


def _calculate_sharpness(image: np.ndarray) -> float:
    """Calculate sharpness using Laplacian variance."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _calculate_contrast(image: np.ndarray) -> float:
    """Calculate contrast using standard deviation."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(np.std(gray))


def _validate_image(image: np.ndarray) -> bool:
    """Validate that image is reasonable (not all black/white, reasonable size)."""
    if image is None or image.size == 0:
        return False
    if image.shape[0] < 100 or image.shape[1] < 100:
        return False
    mean = np.mean(image)
    if mean < 5 or mean > 250:  # All black or all white
        return False
    return True


def _extract_key_metrics(quality: QualityMetrics) -> dict:
    """Extract key metrics for before/after comparison."""
    return {
        "blur": quality.blur.raw_score,
        "contrast": quality.contrast.raw_score,
        "noise": quality.noise.raw_score,
        "brightness": quality.brightness.mean_brightness,
        "skew": abs(quality.skew.angle),
    }
