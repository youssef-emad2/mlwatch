import numpy as np
from typing import Dict, Any


def psi_test(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> Dict[str, Any]:
    ref_hist, bin_edges = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bin_edges)

    ref_hist = np.where(ref_hist == 0, 1e-6, ref_hist) / len(reference)
    cur_hist = np.where(cur_hist == 0, 1e-6, cur_hist) / len(current)

    psi_value = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
    psi_value = round(float(psi_value), 4)

    if psi_value < 0.1:
        severity = "none"
    elif psi_value < 0.2:
        severity = "moderate"
    else:
        severity = "severe"

    return {
        "method": "PSI",
        "score": psi_value,
        "severity": severity,
        "drifted": psi_value >= 0.1,
    }
