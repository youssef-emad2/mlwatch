# mlwatch 🔍

> Lightweight ML monitoring — drift detection, performance tracking, data quality & alerts in one library.

[![PyPI version](https://badge.fury.io/py/mlwatch.svg)](https://badge.fury.io/py/mlwatch)
[![Python](https://img.shields.io/pypi/pyversions/mlwatch)](https://pypi.org/project/mlwatch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Why mlwatch?

| | mlwatch | Evidently | WhyLogs |
|---|---|---|---|
| Simple API | ✅ | ❌ | ❌ |
| Lightweight | ✅ | ❌ | ❌ |
| JSON output | ✅ | ❌ | ❌ |
| No setup needed | ✅ | ❌ | ❌ |
| SQLite history | ✅ | ❌ | ❌ |

---

## Install

```bash
pip install mlwatch
```

---

## Quickstart

```python
import mlwatch
import numpy as np
import pandas as pd

# Your training data
train_data = pd.DataFrame({
    "age":    np.random.normal(30, 5, 1000),
    "income": np.random.normal(50000, 10000, 1000),
})

# New incoming data
new_data = pd.DataFrame({
    "age":    np.random.normal(45, 5, 500),  # shifted!
    "income": np.random.normal(50000, 10000, 500),
})

# Monitor your model
monitor = mlwatch.Monitor(name="my_model")

result = monitor.log(
    reference=train_data,
    current=new_data,
    y_true=actual_labels,
    y_pred=model_predictions,
)

print(result.to_dict())
```

---

## Features

### 1. Drift Detection
Detects when your input data distribution has changed.

```python
from mlwatch import detect

result = detect(reference_data, current_data)
# {
#   "drifted": True,
#   "features": {
#     "age":    { "drifted": True,  "ks": {...}, "psi": {...} },
#     "income": { "drifted": False, "ks": {...}, "psi": {...} }
#   }
# }
```

Supported methods:
- **KS Test** — Kolmogorov-Smirnov statistical test
- **PSI** — Population Stability Index
- **Mean Shift** — normalized mean difference

---

### 2. Performance Monitoring
Tracks model accuracy over time and alerts on degradation.

```python
from mlwatch import track

# Classification
result = track(y_true, y_pred, task="classification")
# { "accuracy": 0.91, "f1": 0.89, "degraded": False }

# Regression
result = track(y_true, y_pred, task="regression")
# { "mae": 0.03, "rmse": 0.05, "r2": 0.97, "degraded": False }
```

---

### 3. Data Quality
Catches nulls, outliers, and schema issues before they break your model.

```python
from mlwatch import check

result = check(dataframe)
# {
#   "nulls":    { "age": 3 },
#   "outliers": { "income": 7 },
#   "passed":   False,
#   "issues":   ["nulls found in columns: ['age']"]
# }
```

---

### 4. Alerts
Get notified when something goes wrong.

```python
from mlwatch import Monitor
from mlwatch.alerts import AlertConfig

monitor = Monitor(
    name="my_model",
    alerts=AlertConfig(
        webhook="https://hooks.slack.com/...",
        on_drift=lambda data: print("Drift detected!", data),
        on_degradation=lambda data: print("Model degraded!", data),
    )
)
```

---

### 5. History
All results are saved to SQLite automatically.

```python
monitor = mlwatch.Monitor(name="my_model", storage="mlwatch.db")

# Get last 50 logs
history = monitor.history.get("my_model", limit=50)
```

---

## Full Example

```python
import mlwatch
import numpy as np
import pandas as pd
from mlwatch.alerts import AlertConfig

monitor = mlwatch.Monitor(
    name="purchase_model",
    storage="mlwatch.db",
    alerts=AlertConfig(
        on_drift=lambda d: print("⚠️  Drift detected!", d),
        on_degradation=lambda d: print("📉 Model degraded!", d),
    )
)

result = monitor.log(
    reference=train_data,
    current=new_data,
    y_true=y_true,
    y_pred=y_pred,
    task="classification",
    thresholds={"accuracy": 0.85, "f1": 0.80},
)

print(result.to_dict())
```

---

## API Reference

### `mlwatch.Monitor`
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Model name |
| `storage` | str | `mlwatch.db` | SQLite path |
| `alerts` | AlertConfig | None | Alert configuration |

### `monitor.log()`
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference` | DataFrame/ndarray | required | Training data |
| `current` | DataFrame/ndarray | required | New data |
| `y_true` | ndarray | None | Ground truth labels |
| `y_pred` | ndarray | None | Model predictions |
| `task` | str | `classification` | `classification` or `regression` |
| `thresholds` | dict | None | Custom degradation thresholds |

---

## License

MIT
