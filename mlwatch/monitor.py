import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union
from .core.base import MonitorResult
from .drift import detect as detect_drift
from .performance import track as track_performance
from .quality import check as check_quality
from .alerts import AlertConfig
from .storage.history import History


class Monitor:
    def __init__(
        self,
        name: str,
        storage: str = "mlwatch.db",
        alerts: Optional[AlertConfig] = None,
    ):
        self.name = name
        self.history = History(storage)
        self.alerts = alerts

    def log(
        self,
        reference: Union[pd.DataFrame, np.ndarray],
        current: Union[pd.DataFrame, np.ndarray],
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        task: str = "classification",
        thresholds: Optional[Dict[str, float]] = None,
    ) -> MonitorResult:
        result = MonitorResult()

        # Drift
        result.drift = detect_drift(reference, current)

        # Performance
        if y_true is not None and y_pred is not None:
            result.performance = track_performance(y_true, y_pred, task, thresholds=thresholds)

        # Quality
        if isinstance(current, pd.DataFrame):
            result.quality = check_quality(current)

        # Alerts
        if self.alerts:
            if result.drift and result.drift.get("drifted"):
                result.alerts_sent += self.alerts.send("drift", result.drift)
            if result.performance and result.performance.get("degraded"):
                result.alerts_sent += self.alerts.send("degradation", result.performance)
            if result.quality and not result.quality.get("passed"):
                result.alerts_sent += self.alerts.send("quality", result.quality)

        self.history.save(self.name, result.to_dict())
        return result
