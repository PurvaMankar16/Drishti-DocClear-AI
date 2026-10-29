import math

import cv2
import numpy as np

from app.core.config import (
    BLUR_THRESHOLDS,
    BRIGHTNESS_THRESHOLDS,
    CONTRAST_THRESHOLDS,
    NOISE_THRESHOLDS,
    RESOLUTION_THRESHOLDS,
    SKEW_THRESHOLDS,
)
from app.schemas.quality import (
    BlurMetric,
    BrightnessMetric,
    ContrastMetric,
    CroppingMetric,
    NoiseMetric,
    QualityMetrics,
    ResolutionMetric,
    SkewMetric,
)


def analyze_image(image: np.ndarray) -> QualityMetrics:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return QualityMetrics(
        blur=calculate_blur(gray),
        brightness=calculate_brightness(gray),
        contrast=calculate_contrast(gray),
        noise=calculate_noise(gray),
        skew=calculate_skew(gray),
        cropping=calculate_cropping(gray),
        resolution=calculate_resolution(gray),
    )


def calculate_blur(gray: np.ndarray) -> BlurMetric:
    raw_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if raw_score >= BLUR_THRESHOLDS["good"]:
        status = "good"
    elif raw_score >= BLUR_THRESHOLDS["warning"]:
        status = "warning"
    else:
        status = "poor"
    normalized = _scale(raw_score, BLUR_THRESHOLDS["warning"], BLUR_THRESHOLDS["good"])
    return BlurMetric(score=normalized, raw_score=raw_score, status=status)


def calculate_brightness(gray: np.ndarray) -> BrightnessMetric:
    mean = float(np.mean(gray))
    thresholds = BRIGHTNESS_THRESHOLDS
    if mean < thresholds["warning_min"]:
        status, condition = "poor", "too_dark"
    elif mean < thresholds["good_min"]:
        status, condition = "warning", "too_dark"
    elif mean <= thresholds["good_max"]:
        status, condition = "good", "acceptable"
    elif mean <= thresholds["warning_max"]:
        status, condition = "warning", "too_bright"
    else:
        status, condition = "poor", "too_bright"
    normalized = max(0.0, 100.0 - abs(mean - 128.0) / 128.0 * 100.0)
    return BrightnessMetric(mean_brightness=mean, normalized_score=normalized, status=status, condition=condition)


def calculate_contrast(gray: np.ndarray) -> ContrastMetric:
    raw_score = float(np.std(gray))
    if raw_score >= CONTRAST_THRESHOLDS["good"]:
        status = "good"
    elif raw_score >= CONTRAST_THRESHOLDS["warning"]:
        status = "warning"
    else:
        status = "poor"
    return ContrastMetric(score=_scale(raw_score, 0.0, 90.0), raw_score=raw_score, status=status)


def calculate_noise(gray: np.ndarray) -> NoiseMetric:
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    raw_score = float(np.std(gray.astype(np.float32) - smooth.astype(np.float32)))
    if raw_score <= NOISE_THRESHOLDS["good"]:
        status = "good"
    elif raw_score <= NOISE_THRESHOLDS["warning"]:
        status = "warning"
    else:
        status = "poor"
    quality_score = 100.0 - _scale(raw_score, 0.0, 40.0)
    return NoiseMetric(score=quality_score, raw_score=raw_score, status=status)


def calculate_skew(gray: np.ndarray) -> SkewMetric:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(30, min(gray.shape) // 8), minLineLength=max(30, gray.shape[1] // 8), maxLineGap=12)
    angles: list[float] = []
    if lines is not None:
        for line in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = map(int, line)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) <= 15 or abs(abs(angle) - 180) <= 15:
                normalized_angle = angle if abs(angle) <= 90 else angle - math.copysign(180, angle)
                angles.append(normalized_angle)
    angle = float(np.median(angles)) if angles else 0.0
    absolute_angle = abs(angle)
    status = "good" if absolute_angle <= SKEW_THRESHOLDS["good"] else "warning" if absolute_angle <= SKEW_THRESHOLDS["warning"] else "poor"
    return SkewMetric(angle=angle, status=status)


def calculate_cropping(gray: np.ndarray) -> CroppingMetric:
    height, width = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        area = cv2.contourArea(polygon)
        if len(polygon) == 4 and area >= 0.15 * width * height:
            candidates.append((area, cv2.boundingRect(polygon)))
    if not candidates:
        return CroppingMetric(document_detected=False, centered=False, appears_cropped=False, status="warning")
    _, (x, y, w, h) = max(candidates, key=lambda item: item[0])
    margin = 0.05
    centered = abs((x + w / 2) - width / 2) <= width * 0.12 and abs((y + h / 2) - height / 2) <= height * 0.12
    appears_cropped = x <= width * margin or y <= height * margin or x + w >= width * (1 - margin) or y + h >= height * (1 - margin)
    status = "poor" if appears_cropped else "good" if centered else "warning"
    return CroppingMetric(document_detected=True, centered=centered, appears_cropped=appears_cropped, status=status)


def calculate_resolution(gray: np.ndarray) -> ResolutionMetric:
    height, width = gray.shape
    megapixels = width * height / 1_000_000
    thresholds = RESOLUTION_THRESHOLDS
    status = "good" if megapixels >= thresholds["good_megapixels"] else "warning" if megapixels >= thresholds["warning_megapixels"] else "poor"
    score = _scale(megapixels, 0.0, thresholds["good_megapixels"])
    return ResolutionMetric(width=width, height=height, megapixels=megapixels, score=score, status=status)


def _scale(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(100.0, (value - low) / (high - low) * 100.0))
