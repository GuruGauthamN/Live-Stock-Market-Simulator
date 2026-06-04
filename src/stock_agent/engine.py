from __future__ import annotations

from datetime import datetime

from stock_agent.config import Settings
from stock_agent.data.storage import MarketDataStore
from stock_agent.models.features import compute_features, ticks_to_bars
from stock_agent.models.predict import SignalModel
from stock_agent.schemas import SignalResponse


def to_action(label: int) -> str:
    if label > 0:
        return "buy"
    if label < 0:
        return "sell"
    return "hold"


class SignalEngine:
    def __init__(
        self, settings: Settings, store: MarketDataStore, model: SignalModel
    ) -> None:
        self.settings = settings
        self.store = store
        self.model = model

    def signal_for_symbol(self, symbol: str) -> SignalResponse:
        symbol = symbol.upper()
        ticks = self.store.load_ticks(symbol)
        bars = ticks_to_bars(ticks, bar_seconds=60)
        if len(bars) < self.settings.model_min_bars:
            return SignalResponse(
                symbol=symbol,
                action="hold",
                confidence=0.0,
                expected_return=0.0,
                timestamp=datetime.utcnow(),
                reason=f"Insufficient bars. Need at least {self.settings.model_min_bars}.",
            )
        features = compute_features(bars)
        if features.empty:
            return SignalResponse(
                symbol=symbol,
                action="hold",
                confidence=0.0,
                expected_return=0.0,
                timestamp=datetime.utcnow(),
                reason="No valid features yet.",
            )
        if not self.model.is_loaded():
            return SignalResponse(
                symbol=symbol,
                action="hold",
                confidence=0.0,
                expected_return=0.0,
                timestamp=datetime.utcnow(),
                reason="Model not trained.",
            )
        pred = self.model.predict_latest(features)
        action = to_action(pred.label)
        if pred.confidence < self.settings.signal_threshold:
            action = "hold"
            reason = (
                f"Confidence {pred.confidence:.3f} below threshold "
                f"{self.settings.signal_threshold:.3f}."
            )
        else:
            reason = f"Model signal accepted with confidence {pred.confidence:.3f}."
        return SignalResponse(
            symbol=symbol,
            action=action,
            confidence=pred.confidence,
            expected_return=pred.expected_return,
            timestamp=datetime.utcnow(),
            reason=reason,
        )
