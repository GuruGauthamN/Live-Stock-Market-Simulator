from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class LlmDecision:
    action: str
    confidence: float
    reason: str


class LlmDecisionClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: int = 20,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def enabled(self) -> bool:
        return bool(self.api_key)

    def decide(self, context: dict[str, Any]) -> LlmDecision:
        if not self.enabled():
            raise RuntimeError("LLM API key is not configured.")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a cautious paper-trading decision engine. "
                        "Return only valid JSON with keys action, confidence, reason. "
                        "action must be one of buy, sell, hold. confidence must be 0 to 1. "
                        "Use the provided market/news/model context only. Prefer hold when uncertain."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(context, separators=(",", ":"), default=str),
                },
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"LLM API request failed: {exc.reason}") from exc

        content = response_payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        action = str(parsed.get("action", "hold")).lower()
        if action not in {"buy", "sell", "hold"}:
            action = "hold"
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
        reason = str(parsed.get("reason", "No reason returned."))
        return LlmDecision(action=action, confidence=confidence, reason=reason)
