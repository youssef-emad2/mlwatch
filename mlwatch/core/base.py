from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time


@dataclass
class MonitorResult:
    timestamp: float = field(default_factory=time.time)
    drift: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None
    quality: Optional[Dict[str, Any]] = None
    alerts_sent: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "drift": self.drift,
            "performance": self.performance,
            "quality": self.quality,
            "alerts_sent": self.alerts_sent,
        }
