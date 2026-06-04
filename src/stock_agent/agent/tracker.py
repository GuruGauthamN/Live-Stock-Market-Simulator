from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import pandas as pd

from stock_agent.data.storage import MarketDataStore


@dataclass
class DecisionRecord:
    decision_id: str
    created_at: str
    symbol: str
    invest: bool
    action: str
    confidence: float
    entry_price: float
    qty: int
    horizon_minutes: int
    target_time: str
    resolved: bool = False
    exit_price: float | None = None
    pnl: float | None = None
    return_pct: float | None = None
    was_correct: bool | None = None


class DecisionTracker:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self._lock = Lock()
        if not self.csv_path.exists():
            pd.DataFrame(
                columns=[
                    "decision_id",
                    "created_at",
                    "symbol",
                    "invest",
                    "action",
                    "confidence",
                    "entry_price",
                    "qty",
                    "horizon_minutes",
                    "target_time",
                    "resolved",
                    "exit_price",
                    "pnl",
                    "return_pct",
                    "was_correct",
                ]
            ).to_csv(self.csv_path, index=False)

    def add(self, record: DecisionRecord) -> None:
        row = pd.DataFrame([record.__dict__])
        with self._lock:
            row.to_csv(self.csv_path, mode="a", header=False, index=False)

    def load(self) -> pd.DataFrame:
        with self._lock:
            df = pd.read_csv(self.csv_path)
        if df.empty:
            return df
        if "qty" not in df.columns:
            df["qty"] = 1
        if "pnl" not in df.columns:
            df["pnl"] = None
        for col in ["created_at", "target_time"]:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        df["resolved"] = df["resolved"].astype(str).str.lower().eq("true")
        return df

    def save(self, df: pd.DataFrame) -> None:
        with self._lock:
            df.to_csv(self.csv_path, index=False)

    def evaluate_pending(self, market_store: MarketDataStore) -> dict:
        df = self.load()
        if df.empty:
            return {
                "updated": 0,
                "total": 0,
                "resolved": 0,
                "accuracy": 0.0,
                "avg_return_pct": 0.0,
                "total_hypothetical_pnl": 0.0,
            }

        updated = 0
        for idx, row in df[~df["resolved"]].iterrows():
            symbol = str(row["symbol"])
            ticks = market_store.load_ticks(symbol)
            if ticks.empty:
                continue
            target_time = pd.to_datetime(row["target_time"], errors="coerce")
            if pd.isna(target_time):
                continue
            future = ticks[ticks["timestamp"] >= target_time]
            if future.empty:
                continue
            exit_price = float(future.iloc[0]["price"])
            entry_price = float(row["entry_price"])
            qty = int(row.get("qty", 1) or 1)
            ret = (exit_price / entry_price) - 1.0
            action = str(row["action"]).lower()
            if action == "buy":
                pnl = (exit_price - entry_price) * qty
                was_correct = ret > 0
            elif action == "sell":
                pnl = (entry_price - exit_price) * qty
                was_correct = ret < 0
            else:
                pnl = 0.0
                was_correct = abs(ret) <= 0.0015

            df.at[idx, "resolved"] = True
            df.at[idx, "exit_price"] = exit_price
            df.at[idx, "pnl"] = pnl
            df.at[idx, "return_pct"] = ret
            df.at[idx, "was_correct"] = bool(was_correct)
            updated += 1

        self.save(df)
        resolved_df = df[df["resolved"] == True]
        total_resolved = len(resolved_df)
        if total_resolved == 0:
            accuracy = 0.0
            avg_ret = 0.0
            total_pnl = 0.0
        else:
            accuracy = float(resolved_df["was_correct"].astype(bool).mean())
            avg_ret = float(resolved_df["return_pct"].astype(float).mean())
            total_pnl = float(resolved_df["pnl"].fillna(0.0).astype(float).sum())

        return {
            "updated": updated,
            "total": int(len(df)),
            "resolved": int(total_resolved),
            "accuracy": accuracy,
            "avg_return_pct": avg_ret,
            "total_hypothetical_pnl": total_pnl,
            "recent": resolved_df.tail(20).to_dict(orient="records"),
        }
