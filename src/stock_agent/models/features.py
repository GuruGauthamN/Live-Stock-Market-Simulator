from __future__ import annotations

import numpy as np
import pandas as pd


def ticks_to_bars(df_ticks: pd.DataFrame, bar_seconds: int = 60) -> pd.DataFrame:
    if df_ticks.empty:
        return pd.DataFrame()
    frame = df_ticks.copy()
    frame = frame.set_index("timestamp")
    rule = f"{bar_seconds}s"
    bars = frame.resample(rule).agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("volume", "sum"),
    )
    bars = bars.dropna()
    return bars


def compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()
    df = bars.copy()
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["vwap_20"] = (df["close"] * df["volume"]).rolling(20).sum() / (
        df["volume"].rolling(20).sum() + 1e-9
    )
    df["rsi_14"] = _rsi(df["close"], 14)
    df["sma_gap"] = (df["sma_5"] - df["sma_20"]) / (df["sma_20"] + 1e-9)
    return df.dropna()


def merge_news_features(features: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    if df.empty:
        return df
    if news_df.empty:
        df["news_sentiment_5"] = 0.0
        df["news_volume_10"] = 0.0
        return df

    news = news_df.copy()
    news = news.set_index("timestamp")
    minute = news.resample("60s").agg(
        sentiment=("sentiment", "mean"),
        news_count=("sentiment", "count"),
    )
    minute["news_sentiment_5"] = minute["sentiment"].rolling(5).mean().fillna(0.0)
    minute["news_volume_10"] = minute["news_count"].rolling(10).sum().fillna(0.0)

    joined = df.join(minute[["news_sentiment_5", "news_volume_10"]], how="left")
    joined["news_sentiment_5"] = joined["news_sentiment_5"].fillna(0.0)
    joined["news_volume_10"] = joined["news_volume_10"].fillna(0.0)
    return joined


def create_labels(features: pd.DataFrame, horizon_bars: int = 5) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    df = features.copy()
    df["future_return"] = df["close"].shift(-horizon_bars) / df["close"] - 1.0
    up = 0.0015
    down = -0.0015
    df["target"] = np.where(
        df["future_return"] > up, 1, np.where(df["future_return"] < down, -1, 0)
    )
    return df.dropna()


def feature_columns() -> list[str]:
    return [
        "ret_1",
        "ret_5",
        "sma_gap",
        "vol_20",
        "vwap_20",
        "rsi_14",
        "volume",
        "news_sentiment_5",
        "news_volume_10",
    ]


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.rolling(window).mean()
    avg_down = down.rolling(window).mean()
    rs = avg_up / (avg_down + 1e-9)
    return 100 - (100 / (1 + rs))
