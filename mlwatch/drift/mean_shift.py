import numpy as np
from typing import Dict, Any


def mean_shift_test(reference: np.ndarray, current: np.ndarray, threshold: float = 0.05) -> Dict[str, Any]:
    ref_mean = np.mean(reference)
    cur_mean = np.mean(current)
    ref_std = np.std(reference)

    shift = abs(cur_mean - ref_mean)
    normalized_shift = shift / (ref_std + 1e-10)
    drifted = normalized_shift > threshold

    return {
        "method": "MeanShift",
        "reference_mean": round(float(ref_mean), 4),
        "current_mean": round(float(cur_mean), 4),
        "shift": round(float(normalized_shift), 4),
        "drifted": drifted,
    }
