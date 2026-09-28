from dataclasses import dataclass

import numpy as np

from app.schemas.quality import QualityMetrics
from app.services.enhancement.contrast import enhance_contrast
from app.services.enhancement.deblur import sharpen
from app.services.enhancement.denoise import denoise
from app.services.enhancement.deskew import deskew
from app.services.enhancement.illumination import correct_illumination
from app.services.enhancement.perspective import correct_perspective
from app.services.enhancement.upscaling import upscale


@dataclass
class EnhancementResult:
    image: np.ndarray
    operations: list[dict]


def enhance_document(image: np.ndarray, quality: QualityMetrics) -> EnhancementResult:
    current = image.copy()
    operations: list[dict] = []

    if quality.skew.status != "good" and abs(quality.skew.angle) > 1.0:
        current = deskew(current, quality.skew.angle)
        operations.append({"name": "deskew", "applied": True, "parameters": {"angle": round(-quality.skew.angle, 2)}})
    else:
        operations.append({"name": "deskew", "applied": False, "reason": "skew is within the configured tolerance"})

    if quality.cropping.document_detected:
        current, applied = correct_perspective(current)
        operations.append({"name": "perspective_correction", "applied": applied, "reason": None if applied else "no reliable quadrilateral found"})
    else:
        operations.append({"name": "perspective_correction", "applied": False, "reason": "document boundary is not reliable"})

    if quality.brightness.status != "good":
        current = correct_illumination(current)
        operations.append({"name": "illumination_correction", "applied": True, "parameters": {"target_brightness": 128}})
    else:
        operations.append({"name": "illumination_correction", "applied": False, "reason": "brightness is acceptable"})

    if quality.contrast.status != "good":
        current = enhance_contrast(current)
        operations.append({"name": "contrast_enhancement", "applied": True, "parameters": {"method": "CLAHE", "clip_limit": 2.0}})
    else:
        operations.append({"name": "contrast_enhancement", "applied": False, "reason": "contrast is acceptable"})

    if quality.noise.status != "good":
        current = denoise(current)
        operations.append({"name": "denoising", "applied": True, "parameters": {"method": "fastNlMeans", "strength": 7}})
    else:
        operations.append({"name": "denoising", "applied": False, "reason": "noise level is acceptable"})

    if quality.resolution.status != "good":
        current = upscale(current)
        operations.append({"name": "upscaling", "applied": True, "parameters": {"method": "INTER_LANCZOS4", "scale": 2.0}})
    else:
        operations.append({"name": "upscaling", "applied": False, "reason": "resolution is sufficient"})

    if quality.blur.status != "good":
        current = sharpen(current)
        operations.append({"name": "sharpening", "applied": True, "parameters": {"method": "unsharp_mask", "amount": 0.55}})
    else:
        operations.append({"name": "sharpening", "applied": False, "reason": "sharpness is acceptable"})

    return EnhancementResult(image=current, operations=operations)
