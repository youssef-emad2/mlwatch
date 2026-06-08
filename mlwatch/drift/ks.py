import numpy as np
from scipy import stats
from typing import Dict, Any


def ks_test(reference: np.ndarray, current: np.ndarray) -> Dict[str, Any]:
    stat, p_value = stats.ks_2samp(reference, current)
    drifted = p_value < 0.05
    return {
        "method": "KS",
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 4),
        "drifted": drifted,
    }
