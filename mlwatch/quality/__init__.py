import pandas as pd
import numpy as np
from typing import Dict, Any


def check(df: pd.DataFrame) -> Dict[str, Any]:
    issues = []
    result = {}

    # Nulls
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0].to_dict()
    result["nulls"] = {col: int(count) for col, count in null_cols.items()}
    if null_cols:
        issues.append(f"nulls found in columns: {list(null_cols.keys())}")

    # Outliers (IQR)
    outlier_cols = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
        if len(outliers) > 0:
            outlier_cols[col] = len(outliers)
    result["outliers"] = outlier_cols
    if outlier_cols:
        issues.append(f"outliers found in columns: {list(outlier_cols.keys())}")

    # Schema
    result["schema"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    result["issues"] = issues
    result["passed"] = len(issues) == 0

    return result
