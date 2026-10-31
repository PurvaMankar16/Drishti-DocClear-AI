"""ResNet-18 document-defect inference service.

The checkpoint is trained on Drishti's synthetic six-label dataset. Predictions
are returned as supplementary evidence for the user; they do not replace objective
image-quality metrics based on classical computer vision techniques.
"""
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from torch import nn

from app.core.config import ML_CHECKPOINT_PATH

CLASS_NAMES = [
    "blur",
    "dark",
    "low_contrast",
    "noise",
    "skew",
    "low_resolution",
]


class DefectPredictionModelError(RuntimeError):
    """Raised when the configured defect model cannot be loaded or run."""


@lru_cache(maxsize=1)
def _load_model() -> tuple[nn.Module, dict[str, Any]]:
    if not ML_CHECKPOINT_PATH.is_file():
        raise DefectPredictionModelError(
            f"Model checkpoint not found at {ML_CHECKPOINT_PATH}. "
            "Place best_resnet18.pth in backend/models/."
        )

    try:
        checkpoint = torch.load(ML_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(ML_CHECKPOINT_PATH, map_location="cpu")

    class_names = checkpoint.get("class_names", CLASS_NAMES)
    if class_names != CLASS_NAMES:
        raise DefectPredictionModelError(
            f"Checkpoint labels do not match the expected labels: {class_names}"
        )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except (KeyError, RuntimeError) as exc:
        raise DefectPredictionModelError("Could not load ResNet-18 checkpoint weights.") from exc

    model.eval()
    mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("std", [0.229, 0.224, 0.225])
    image_size = int(checkpoint.get("image_size", 224))
    threshold_map = checkpoint.get("thresholds", {name: 0.1 for name in CLASS_NAMES})

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(tuple(mean), tuple(std)),
    ])
    metadata = {
        "preprocess": preprocess,
        "thresholds": threshold_map,
        "class_names": class_names,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "evaluation_dataset": checkpoint.get("synthetic_test_metrics", {}).get("evaluation_dataset"),
    }
    return model, metadata


def predict_defects(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    """Return per-label probabilities and threshold decisions for a BGR image."""
    model, metadata = _load_model()

    if image_bgr is None or image_bgr.size == 0:
        raise DefectPredictionModelError("Cannot run defect inference on an empty image.")

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    tensor = metadata["preprocess"](pil_image).unsqueeze(0)

    with torch.inference_mode():
        probabilities = torch.sigmoid(model(tensor))[0].cpu().numpy()

    predictions = []
    for index, label in enumerate(metadata["class_names"]):
        probability = float(probabilities[index])
        threshold = float(metadata["thresholds"].get(label, 0.1))
        predictions.append({
            "label": label,
            "probability": round(probability, 6),
            "threshold": threshold,
            "detected": probability >= threshold,
        })
    return predictions


def model_is_available() -> bool:
    """Report whether the checkpoint exists; does not force-load the model."""
    return ML_CHECKPOINT_PATH.is_file()
