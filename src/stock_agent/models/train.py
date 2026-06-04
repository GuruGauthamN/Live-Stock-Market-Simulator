from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stock_agent.models.features import create_labels, feature_columns


@dataclass
class TrainResult:
    rows_used: int
    classes_seen: list[int]


@dataclass
class BacktestResult:
    train_rows: int
    test_rows: int
    accuracy: float
    total_pnl: float
    trades: int
    winning_trades: int
    losing_trades: int
    test_start: str
    test_end: str
    sample_predictions: list[dict]


def train_model(labeled: pd.DataFrame, model_path: Path) -> TrainResult:
    if labeled.empty:
        raise ValueError("No labeled rows available for training.")
    cols = feature_columns()
    data = labeled.dropna(subset=cols + ["target"])
    if data.empty:
        raise ValueError("Not enough data after cleaning features.")
    x = data[cols]
    y = data["target"].astype(int)
    model = _build_pipeline()
    model.fit(x, y)
    payload = {"model": model, "feature_columns": cols}
    joblib.dump(payload, model_path)
    return TrainResult(rows_used=len(data), classes_seen=sorted(y.unique().tolist()))


def build_training_frame(
    feature_frames: dict[str, pd.DataFrame], horizon_bars: int
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for symbol, frame in feature_frames.items():
        labeled = create_labels(frame, horizon_bars=horizon_bars)
        if labeled.empty:
            continue
        labeled["timestamp"] = labeled.index
        labeled["symbol"] = symbol
        parts.append(labeled)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=0, ignore_index=False)


def run_time_split_backtest(
    labeled: pd.DataFrame, model_path: Path, test_ratio: float = 0.2
) -> BacktestResult:
    if labeled.empty:
        raise ValueError("No labeled data available for backtest.")
    if not 0.05 <= test_ratio <= 0.5:
        raise ValueError("test_ratio must be between 0.05 and 0.5.")

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for symbol, group in labeled.groupby("symbol"):
        ordered = group.sort_values("timestamp").copy()
        if len(ordered) < 40:
            continue
        split_idx = int(len(ordered) * (1.0 - test_ratio))
        split_idx = max(25, min(split_idx, len(ordered) - 10))
        train_parts.append(ordered.iloc[:split_idx].copy())
        test_parts.append(ordered.iloc[split_idx:].copy())

    if not train_parts or not test_parts:
        raise ValueError("Not enough per-symbol rows to build train/test split.")

    train_df = pd.concat(train_parts, axis=0, ignore_index=True)
    test_df = pd.concat(test_parts, axis=0, ignore_index=True)
    cols = feature_columns()
    train_df = train_df.dropna(subset=cols + ["target"])
    test_df = test_df.dropna(subset=cols + ["target"])
    if train_df.empty or test_df.empty:
        raise ValueError("Train or test set is empty after dropping missing values.")

    model = _build_pipeline()
    model.fit(train_df[cols], train_df["target"].astype(int))

    payload = {"model": model, "feature_columns": cols}
    joblib.dump(payload, model_path)

    preds = model.predict(test_df[cols]).astype(int)
    test_df["prediction"] = preds
    test_df["target"] = test_df["target"].astype(int)
    test_df = test_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    accuracy = float(accuracy_score(test_df["target"], test_df["prediction"]))
    pnl, trades, wins, losses, sample_predictions = _simulate_trades(test_df)
    return BacktestResult(
        train_rows=int(len(train_df)),
        test_rows=int(len(test_df)),
        accuracy=accuracy,
        total_pnl=float(pnl),
        trades=int(trades),
        winning_trades=int(wins),
        losing_trades=int(losses),
        test_start=str(test_df["timestamp"].min()),
        test_end=str(test_df["timestamp"].max()),
        sample_predictions=sample_predictions,
    )


def _build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, multi_class="multinomial", class_weight="balanced"
                ),
            ),
        ]
    )


def _simulate_trades(test_df: pd.DataFrame):
    pnl = 0.0
    trades = 0
    wins = 0
    losses = 0
    positions: dict[str, dict[str, float]] = {}
    samples: list[dict] = []

    for _, row in test_df.iterrows():
        symbol = str(row["symbol"])
        price = float(row["close"])
        pred = int(row["prediction"])
        ts = str(row["timestamp"])
        pos = positions.get(symbol, {"side": 0, "entry": 0.0})

        if len(samples) < 25:
            samples.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "close": price,
                    "prediction": pred,
                    "actual": int(row["target"]),
                }
            )

        if pred == 0:
            continue

        if pred == 1:
            if pos["side"] == 1:
                continue
            if pos["side"] == -1:
                trade_pnl = pos["entry"] - price
                pnl += trade_pnl
                trades += 1
                wins += int(trade_pnl > 0)
                losses += int(trade_pnl < 0)
            positions[symbol] = {"side": 1, "entry": price}
            continue

        if pred == -1:
            if pos["side"] == -1:
                continue
            if pos["side"] == 1:
                trade_pnl = price - pos["entry"]
                pnl += trade_pnl
                trades += 1
                wins += int(trade_pnl > 0)
                losses += int(trade_pnl < 0)
            positions[symbol] = {"side": -1, "entry": price}

    for symbol, pos in positions.items():
        symbol_rows = test_df[test_df["symbol"] == symbol]
        if symbol_rows.empty:
            continue
        final_price = float(symbol_rows.iloc[-1]["close"])
        if pos["side"] == 1:
            trade_pnl = final_price - pos["entry"]
        elif pos["side"] == -1:
            trade_pnl = pos["entry"] - final_price
        else:
            continue
        pnl += trade_pnl
        trades += 1
        wins += int(trade_pnl > 0)
        losses += int(trade_pnl < 0)

    return pnl, trades, wins, losses, samples
