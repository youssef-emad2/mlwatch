import numpy as np
from typing import Dict, Any, Optional


def track_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    thresholds = thresholds or {"mae": 0.1, "rmse": 0.15}

    mae = round(float(np.mean(np.abs(y_true - y_pred))), 4)
    rmse = round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4)
    r2 = round(float(1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-10)), 4)

    result = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "degraded": False,
        "issues": [],
    }

    if mae > thresholds.get("mae", 0.1):
        result["degraded"] = True
        result["issues"].append(f"mae {mae} above threshold {thresholds['mae']}")

    if rmse > thresholds.get("rmse", 0.15):
        result["degraded"] = True
        result["issues"].append(f"rmse {rmse} above threshold {thresholds['rmse']}")

    return result
