"""Faster R-CNN document layout detection service.

This module provides document layout detection using a Faster R-CNN model
with ResNet50-FPN backbone, trained on DocLayNet for 11 document element classes.

The model was trained with images resized to 1025×1025 and achieves a validation
mAP@50 of approximately 0.23. This is an experimental prototype and predictions
should be treated as estimates requiring visual verification.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms

from app.core.config import (
    LAYOUT_CHECKPOINT_PATH,
    LAYOUT_CONFIDENCE_THRESHOLD,
    LAYOUT_NMS_IOU_THRESHOLD,
    LAYOUT_INPUT_SIZE,
)
from app.schemas.layout import LayoutDetectionError

# Class names for DocLayNet layout elements (11 classes + background at index 0)
LAYOUT_CLASS_NAMES = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]

# Total classes including background (index 0)
NUM_CLASSES = len(LAYOUT_CLASS_NAMES) + 1  # 12 classes


@lru_cache(maxsize=1)
def _load_layout_model() -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load the Faster R-CNN layout detection model.

    The model is loaded once and cached for subsequent calls. The checkpoint
    contains the model state dict, class mappings, and training metadata.

    Returns:
        Tuple of (model, metadata) where model is the loaded Faster R-CNN
        and metadata contains class names, training info, etc.

    Raises:
        LayoutDetectionError: If checkpoint is missing or incompatible.
    """
    if not LAYOUT_CHECKPOINT_PATH.is_file():
        raise LayoutDetectionError(
            f"Layout detection model checkpoint not found at {LAYOUT_CHECKPOINT_PATH}. "
            "Place best_fasterrcnn_doclaynet.pth in backend/models/."
        )

    try:
        # Load checkpoint with weights_only=False for backward compatibility
        checkpoint = torch.load(
            LAYOUT_CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False
        )
    except Exception as exc:
        raise LayoutDetectionError(
            f"Failed to load layout detection checkpoint: {exc}"
        ) from exc

    # Validate checkpoint structure
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise LayoutDetectionError(
            "Invalid checkpoint format: expected dictionary with 'model_state_dict' key"
        )

    # Validate class names match expected layout classes
    checkpoint_classes = checkpoint.get("class_names", [])
    if checkpoint_classes != LAYOUT_CLASS_NAMES:
        raise LayoutDetectionError(
            f"Checkpoint class names do not match expected layout classes. "
            f"Expected: {LAYOUT_CLASS_NAMES}, Got: {checkpoint_classes}"
        )

    # Create Faster R-CNN model with ResNet50-FPN backbone
    model = fasterrcnn_resnet50_fpn(weights=None)

    # Replace the prediction head with correct number of classes (11 + background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    # Load trained weights
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except (KeyError, RuntimeError) as exc:
        raise LayoutDetectionError(
            f"Could not load Faster R-CNN checkpoint weights: {exc}"
        ) from exc

    # Set to evaluation mode
    model.eval()

    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Extract metadata
    metadata = {
        "class_names": checkpoint_classes,
        "class_to_id": checkpoint.get("class_to_id", {}),
        "num_classes": NUM_CLASSES,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_map50": checkpoint.get("best_map50"),
        "device": device,
    }

    return model, metadata


def detect_layout(
    image_bgr: np.ndarray,
    confidence_threshold: float = LAYOUT_CONFIDENCE_THRESHOLD,
    nms_threshold: float = LAYOUT_NMS_IOU_THRESHOLD,
) -> list[dict[str, Any]]:
    """Detect document layout elements in an image.

    Args:
        image_bgr: Input image in BGR format (OpenCV convention)
        confidence_threshold: Minimum confidence score for detections (0-1)
        nms_threshold: IoU threshold for Non-Maximum Suppression (0-1)

    Returns:
        List of detections, each containing:
            - label: Class name (e.g., "Text", "Table")
            - label_id: Class ID (1-11, background is 0)
            - confidence: Detection confidence score (0-1)
            - bbox: Bounding box dict with x1, y1, x2, y2 coordinates

    Raises:
        LayoutDetectionError: If model cannot be loaded or inference fails
    """
    if image_bgr is None or image_bgr.size == 0:
        raise LayoutDetectionError("Cannot run layout detection on an empty image.")

    # Load model (cached after first call)
    model, metadata = _load_layout_model()
    device = metadata["device"]

    # Get original image dimensions
    original_height, original_width = image_bgr.shape[:2]

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Resize to model input size while maintaining aspect ratio
    scale = min(LAYOUT_INPUT_SIZE / original_width, LAYOUT_INPUT_SIZE / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Convert to tensor and normalize to [0, 1]
    image_tensor = torch.from_numpy(resized).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension

    # Run inference
    with torch.inference_mode():
        predictions = model(image_tensor)

    # Extract predictions from first (and only) image in batch
    pred = predictions[0]
    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()

    # Scale boxes back to original image size
    scale_x = original_width / new_width
    scale_y = original_height / new_height
    boxes[:, [0, 2]] *= scale_x  # x1, x2
    boxes[:, [1, 3]] *= scale_y  # y1, y2

    # Filter by confidence threshold
    confident_mask = scores >= confidence_threshold
    boxes = boxes[confident_mask]
    scores = scores[confident_mask]
    labels = labels[confident_mask]

    if len(boxes) == 0:
        return []

    # Apply Non-Maximum Suppression to remove overlapping detections
    boxes_tensor = torch.from_numpy(boxes).float()
    scores_tensor = torch.from_numpy(scores).float()
    keep_indices = nms(boxes_tensor, scores_tensor, nms_threshold)
    keep_indices = keep_indices.cpu().numpy()

    # Build detection results
    detections = []
    for idx in keep_indices:
        label_id = int(labels[idx])

        # Skip background class (id=0)
        if label_id == 0:
            continue

        # Map label_id to class name (subtract 1 because background is at index 0)
        class_idx = label_id - 1
        if 0 <= class_idx < len(LAYOUT_CLASS_NAMES):
            label_name = LAYOUT_CLASS_NAMES[class_idx]
        else:
            label_name = f"Unknown-{label_id}"

        x1, y1, x2, y2 = boxes[idx]

        detections.append({
            "label": label_name,
            "label_id": label_id,
            "confidence": round(float(scores[idx]), 6),
            "bbox": {
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "x2": round(float(x2), 2),
                "y2": round(float(y2), 2),
            }
        })

    return detections


def model_is_available() -> bool:
    """Check whether the layout detection checkpoint exists.

    This does not force-load the model, only checks file existence.

    Returns:
        True if checkpoint file exists, False otherwise
    """
    return LAYOUT_CHECKPOINT_PATH.is_file()
