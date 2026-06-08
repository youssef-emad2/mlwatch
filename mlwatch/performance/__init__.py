from .classification import track_classification
from .regression import track_regression
import numpy as np
from typing import Dict, Any, Optional


def track(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "classification",
    y_prob: Optional[np.ndarray] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    if task == "classification":
        return track_classification(y_true, y_pred, y_prob, thresholds)
    elif task == "regression":
        return track_regression(y_true, y_pred, thresholds)
    else:
        raise ValueError(f"task must be 'classification' or 'regression', got '{task}'")
