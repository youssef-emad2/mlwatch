# mlwatch

Lightweight ML monitoring — drift detection, performance tracking, data quality, and alerts in one library.

## Install

```bash
pip install mlwatch
```

## Quickstart

```python
import mlwatch
import numpy as np
import pandas as pd

monitor = mlwatch.Monitor(name="my_model")

result = monitor.log(
    reference=train_data,
    current=new_data,
    y_true=y_true,
    y_pred=y_pred,
)

print(result.to_dict())
```

## Features

- Drift Detection (KS test, PSI, Mean Shift)
- Performance Monitoring (accuracy, F1, AUC, MAE, RMSE)
- Data Quality (nulls, outliers, schema)
- Alerts (webhook, custom callbacks)
- History storage (SQLite)

## Why mlwatch?

| | mlwatch | Evidently | WhyLogs |
|---|---|---|---|
| Simple API | ✅ | ❌ | ❌ |
| Lightweight | ✅ | ❌ | ❌ |
| JSON output | ✅ | ❌ | ❌ |
| No setup | ✅ | ❌ | ❌ |
