from __future__ import annotations

from pathlib import Path
from threading import Lock

import pandas as pd

from stock_agent.news.stream import NewsItem


class NewsStore:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self._lock = Lock()
        if not self.csv_path.exists():
            pd.DataFrame(
                columns=["timestamp", "symbol", "headline", "sentiment", "source"]
            ).to_csv(self.csv_path, index=False)

    def append_news(self, item: NewsItem) -> None:
        row = pd.DataFrame(
            [
                {
                    "timestamp": item.timestamp.isoformat(),
                    "symbol": item.symbol.upper(),
                    "headline": item.headline,
                    "sentiment": float(item.sentiment),
                    "source": item.source,
                }
            ]
        )
        with self._lock:
            row.to_csv(self.csv_path, mode="a", header=False, index=False)

    def load_news(self, symbol: str | None = None) -> pd.DataFrame:
        with self._lock:
            df = pd.read_csv(self.csv_path)
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if symbol:
            df = df[df["symbol"] == symbol.upper()]
        return df.sort_values("timestamp")
