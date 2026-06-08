import numpy as np
import pandas as pd
import pytest
from mlwatch import Monitor, detect, track, check


def test_drift_no_drift():
    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(0, 1, 1000)
    result = detect(ref, cur)
    assert "drifted" in result


def test_drift_detected():
    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(5, 1, 1000)
    result = detect(ref, cur)
    assert result["drifted"] is True


def test_drift_dataframe():
    ref = pd.DataFrame({"age": np.random.normal(30, 5, 500), "income": np.random.normal(50000, 10000, 500)})
    cur = pd.DataFrame({"age": np.random.normal(30, 5, 500), "income": np.random.normal(50000, 10000, 500)})
    result = detect(ref, cur)
    assert "features" in result


def test_performance_classification():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0])
    result = track(y_true, y_pred, task="classification")
    assert "accuracy" in result
    assert "f1" in result


def test_quality_check():
    df = pd.DataFrame({
        "age": [25, 30, None, 40],
        "income": [50000, 60000, 70000, 1000000],
    })
    result = check(df)
    assert "nulls" in result
    assert "outliers" in result
    assert result["passed"] is False


def test_monitor_full():
    monitor = Monitor(name="test_model", storage=":memory:")
    ref = pd.DataFrame({"a": np.random.normal(0, 1, 200), "b": np.random.normal(0, 1, 200)})
    cur = pd.DataFrame({"a": np.random.normal(0, 1, 200), "b": np.random.normal(0, 1, 200)})
    y_true = np.random.randint(0, 2, 200)
    y_pred = np.random.randint(0, 2, 200)
    result = monitor.log(reference=ref, current=cur, y_true=y_true, y_pred=y_pred)
    assert result.drift is not None
    assert result.performance is not None
    assert result.quality is not None
