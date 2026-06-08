import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from .ks import ks_test
from .psi import psi_test
from .mean_shift import mean_shift_test


def detect(
    reference: Union[pd.DataFrame, np.ndarray],
    current: Union[pd.DataFrame, np.ndarray],
    method: str = "all",
) -> Dict[str, Any]:
    if isinstance(reference, pd.DataFrame):
        results = {}
        any_drift = False
        for col in reference.columns:
            ref_col = reference[col].dropna().values
            cur_col = current[col].dropna().values
            col_result = _run_method(ref_col, cur_col, method)
            col_drifted = any(v.get("drifted", False) for v in col_result.values())
            results[col] = {**col_result, "drifted": col_drifted}
            if col_drifted:
                any_drift = True
        return {"drifted": any_drift, "features": results}
    else:
        ref = np.array(reference).flatten()
        cur = np.array(current).flatten()
        result = _run_method(ref, cur, method)
        drifted = any(v.get("drifted", False) for v in result.values())
        return {"drifted": drifted, **result}


def _run_method(ref: np.ndarray, cur: np.ndarray, method: str) -> Dict[str, Any]:
    if method == "ks":
        return {"ks": ks_test(ref, cur)}
    elif method == "psi":
        return {"psi": psi_test(ref, cur)}
    elif method == "mean_shift":
        return {"mean_shift": mean_shift_test(ref, cur)}
    else:
        return {
            "ks": ks_test(ref, cur),
            "psi": psi_test(ref, cur),
            "mean_shift": mean_shift_test(ref, cur),
        }
