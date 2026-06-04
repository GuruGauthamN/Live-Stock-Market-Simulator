from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from stock_agent.models.features import feature_columns


@dataclass
class Prediction:
    label: int
    confidence: float
    expected_return: float


class SignalModel:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._bundle: dict | None = None

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        self._bundle = joblib.load(self.model_path)
        return True

    def is_loaded(self) -> bool:
        return self._bundle is not None

    def predict_latest(self, features: pd.DataFrame) -> Prediction:
        if self._bundle is None:
            raise RuntimeError("Model not loaded.")
        if features.empty:
            raise ValueError("No features available for inference.")
        x = features.tail(1)[feature_columns()]
        model = self._bundle["model"]
        probs = model.predict_proba(x)[0]
        classes = model.classes_
        idx = int(np.argmax(probs))
        label = int(classes[idx])
        confidence = float(probs[idx])
        expected_return = float((probs * classes).sum() * 0.0025)
        return Prediction(label=label, confidence=confidence, expected_return=expected_return)
