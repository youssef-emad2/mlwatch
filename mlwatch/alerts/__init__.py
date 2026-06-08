import httpx
from typing import Optional, Callable, Dict, Any


class AlertConfig:
    def __init__(
        self,
        webhook: Optional[str] = None,
        on_drift: Optional[Callable] = None,
        on_degradation: Optional[Callable] = None,
        on_quality: Optional[Callable] = None,
    ):
        self.webhook = webhook
        self.on_drift = on_drift
        self.on_degradation = on_degradation
        self.on_quality = on_quality

    def send(self, event: str, data: Dict[str, Any]) -> list:
        sent = []

        if event == "drift" and self.on_drift:
            self.on_drift(data)
            sent.append("on_drift callback")

        if event == "degradation" and self.on_degradation:
            self.on_degradation(data)
            sent.append("on_degradation callback")

        if event == "quality" and self.on_quality:
            self.on_quality(data)
            sent.append("on_quality callback")

        if self.webhook:
            try:
                httpx.post(self.webhook, json={"event": event, "data": data}, timeout=5)
                sent.append("webhook")
            except Exception:
                pass

        return sent
