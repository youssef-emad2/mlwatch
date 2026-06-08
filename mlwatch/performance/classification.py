import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def track_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    thresholds = thresholds or {"accuracy": 0.8, "f1": 0.8}

    accuracy = round(float(accuracy_score(y_true, y_pred)), 4)
    f1 = round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4)

    result = {
        "accuracy": accuracy,
        "f1": f1,
        "degraded": False,
        "issues": [],
    }

    if y_prob is not None:
        try:
            auc = round(float(roc_auc_score(y_true, y_prob, multi_class="ovr")), 4)
            result["auc"] = auc
        except Exception:
            pass

    if accuracy < thresholds.get("accuracy", 0.8):
        result["degraded"] = True
        result["issues"].append(f"accuracy {accuracy} below threshold {thresholds['accuracy']}")

    if f1 < thresholds.get("f1", 0.8):
        result["degraded"] = True
        result["issues"].append(f"f1 {f1} below threshold {thresholds['f1']}")

    return result
