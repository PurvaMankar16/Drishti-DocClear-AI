import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .config import CLASS_NAMES


def classification_metrics(targets, probabilities, thresholds):
    targets = np.asarray(targets).astype(int)
    probabilities = np.asarray(probabilities)
    predictions = probabilities >= np.asarray(thresholds)
    per_label = {}
    for index, name in enumerate(CLASS_NAMES):
        per_label[name] = {
            "precision": float(precision_score(targets[:, index], predictions[:, index], zero_division=0)),
            "recall": float(recall_score(targets[:, index], predictions[:, index], zero_division=0)),
            "f1": float(f1_score(targets[:, index], predictions[:, index], zero_division=0)),
        }
        try:
            per_label[name]["auroc"] = float(roc_auc_score(targets[:, index], probabilities[:, index]))
        except ValueError:
            per_label[name]["auroc"] = None
    return {
        "per_label": per_label,
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "exact_match_accuracy": float(np.all(targets == predictions, axis=1).mean()),
    }


def find_thresholds(targets, probabilities):
    thresholds = []
    for index in range(len(CLASS_NAMES)):
        best_threshold, best_f1 = 0.5, -1.0
        for threshold in np.arange(0.1, 0.91, 0.05):
            score = f1_score(targets[:, index], probabilities[:, index] >= threshold, zero_division=0)
            if score > best_f1:
                best_threshold, best_f1 = float(threshold), float(score)
        thresholds.append(best_threshold)
    return thresholds
