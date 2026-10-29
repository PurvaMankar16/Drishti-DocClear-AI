"""Conservative, metric-driven document enhancement pipeline.

Operations are applied only when the corresponding quality metric indicates a
problem.  Strengths are deliberately mild to avoid artefacts on handwritten
and photographic content.

Order matters:
  1. Geometry corrections first (deskew, perspective) – so subsequent pixel
     operations work on a properly-oriented image.
  2. Illumination and contrast – tonal corrections before sharpening.
  3. Denoising – before sharpening so noise is not amplified.
  4. Upscaling – before sharpening to benefit from higher resolution.
  5. Sharpening last – so it enhances the final composite result.
"""
from dataclasses import dataclass, field

import numpy as np

from app.schemas.quality import QualityMetrics
from app.services.enhancement.contrast import enhance_contrast
from app.services.enhancement.deblur import sharpen
from app.services.enhancement.denoise import denoise
from app.services.enhancement.deskew import deskew
from app.services.enhancement.illumination import correct_illumination
from app.services.enhancement.perspective import correct_perspective
from app.services.enhancement.upscaling import upscale

# Conservative sharpening defaults – keep in sync with sharpen() defaults.
_SHARPEN_AMOUNT = 0.45
_SHARPEN_SIGMA = 1.1


@dataclass
class EnhancementResult:
    image: np.ndarray
    operations: list[dict] = field(default_factory=list)


def enhance_document(image: np.ndarray, quality: QualityMetrics) -> EnhancementResult:
    """Apply conservative enhancement operations driven by *quality* metrics.

    Each operation is attempted only when the relevant metric is not "good".
    The original array is never mutated; a copy is made at the start.
    """
    current = image.copy()
    operations: list[dict] = []

    # 1. Deskew – only when the detected angle exceeds 1° (below that the
    #    interpolation artefacts outweigh the benefit).
    if quality.skew.status != "good" and abs(quality.skew.angle) > 1.0:
        current = deskew(current, quality.skew.angle)
        operations.append({
            "name": "deskew",
            "applied": True,
            "parameters": {"angle": round(-quality.skew.angle, 2)},
        })
    else:
        operations.append({
            "name": "deskew",
            "applied": False,
            "reason": "skew is within the configured tolerance",
        })

    # 2. Perspective correction – only when a reliable document boundary was
    #    detected by the cropping metric.  For photographs without a clear
    #    quadrilateral border (the common case for handwritten notes), this
    #    step is skipped rather than distorting the image.
    if quality.cropping.document_detected:
        current, applied = correct_perspective(current)
        operations.append({
            "name": "perspective_correction",
            "applied": applied,
            "reason": None if applied else "no reliable quadrilateral found",
        })
    else:
        operations.append({
            "name": "perspective_correction",
            "applied": False,
            "reason": "document boundary not detected; skipping to preserve framing",
        })

    # 3. Illumination correction – caps the gain at ±25% to avoid clipping.
    if quality.brightness.status != "good":
        current = correct_illumination(current)
        operations.append({
            "name": "illumination_correction",
            "applied": True,
            "parameters": {"target_brightness": 128, "max_gain": 1.25},
        })
    else:
        operations.append({
            "name": "illumination_correction",
            "applied": False,
            "reason": "brightness is acceptable",
        })

    # 4. Contrast enhancement via CLAHE on the L channel.
    if quality.contrast.status != "good":
        current = enhance_contrast(current)
        operations.append({
            "name": "contrast_enhancement",
            "applied": True,
            "parameters": {"method": "CLAHE", "clip_limit": 2.0, "tile_grid": [8, 8]},
        })
    else:
        operations.append({
            "name": "contrast_enhancement",
            "applied": False,
            "reason": "contrast is acceptable",
        })

    # 5. Denoising – only when noise measurement clearly exceeds the threshold.
    if quality.noise.status != "good":
        current = denoise(current)
        operations.append({
            "name": "denoising",
            "applied": True,
            "parameters": {"method": "fastNlMeansDenoisingColored", "strength": 7},
        })
    else:
        operations.append({
            "name": "denoising",
            "applied": False,
            "reason": "noise level is acceptable",
        })

    # 6. Upscaling – only for genuinely low-resolution images.
    if quality.resolution.status != "good":
        current = upscale(current)
        operations.append({
            "name": "upscaling",
            "applied": True,
            "parameters": {"method": "INTER_LANCZOS4", "scale": 2.0},
        })
    else:
        operations.append({
            "name": "upscaling",
            "applied": False,
            "reason": "resolution is sufficient",
        })

    # 7. Sharpening – mild unsharp mask applied last.
    if quality.blur.status != "good":
        current = sharpen(current, amount=_SHARPEN_AMOUNT, sigma=_SHARPEN_SIGMA)
        operations.append({
            "name": "sharpening",
            "applied": True,
            "parameters": {
                "method": "unsharp_mask",
                "amount": _SHARPEN_AMOUNT,
                "sigma": _SHARPEN_SIGMA,
            },
        })
    else:
        operations.append({
            "name": "sharpening",
            "applied": False,
            "reason": "sharpness is acceptable",
        })

    return EnhancementResult(image=current, operations=operations)
