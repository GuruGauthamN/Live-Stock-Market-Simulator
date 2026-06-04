from __future__ import annotations

from pathlib import Path
from threading import Lock

import pandas as pd

from stock_agent.schemas import MarketTick


class MarketDataStore:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self._lock = Lock()
        if not self.csv_path.exists():
            pd.DataFrame(columns=["timestamp", "symbol", "price", "volume"]).to_csv(
                self.csv_path, index=False
            )

    def append_tick(self, tick: MarketTick) -> None:
        row = pd.DataFrame(
            [
                {
                    "timestamp": tick.timestamp.isoformat(),
                    "symbol": tick.symbol.upper(),
                    "price": float(tick.price),
                    "volume": int(tick.volume),
                }
            ]
        )
        with self._lock:
            row.to_csv(self.csv_path, mode="a", header=False, index=False)

    def load_ticks(self, symbol: str | None = None) -> pd.DataFrame:
        with self._lock:
            df = pd.read_csv(self.csv_path)
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if symbol:
            df = df[df["symbol"] == symbol.upper()]
        return df.sort_values("timestamp")

    def list_symbols(self) -> list[str]:
        df = self.load_ticks()
        if df.empty:
            return []
        return sorted(df["symbol"].dropna().unique().tolist())
