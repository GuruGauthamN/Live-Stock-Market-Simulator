from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class MarketTick(BaseModel):
    timestamp: datetime
    symbol: str
    price: float
    volume: int


class TrainResponse(BaseModel):
    status: str
    symbols_trained: list[str]
    rows_used: int


class BacktestPrediction(BaseModel):
    timestamp: str
    symbol: str
    close: float
    prediction: int
    actual: int


class BacktestResponse(BaseModel):
    status: str
    train_rows: int
    test_rows: int
    accuracy: float
    total_pnl: float
    trades: int
    winning_trades: int
    losing_trades: int
    test_start: str
    test_end: str
    sample_predictions: list[BacktestPrediction]


class SignalResponse(BaseModel):
    symbol: str
    action: Literal["buy", "hold", "sell"]
    confidence: float
    expected_return: float
    timestamp: datetime
    reason: str


class DecisionResponse(BaseModel):
    decision_id: str
    symbol: str
    invest: bool
    action: Literal["buy", "hold", "sell"]
    confidence: float
    expected_return: float
    news_sentiment: float
    horizon_minutes: int
    qty: int
    trade_status: str
    fill_price: float | None = None
    portfolio_total_pnl: float | None = None
    reason: str


class LlmAgentRunResponse(BaseModel):
    status: str
    decisions: list[DecisionResponse]
    portfolio_total_pnl: float
    realized_pnl: float
    unrealized_pnl: float


class TradeResponse(BaseModel):
    status: str
    symbol: str
    side: Literal["buy", "sell"]
    qty: int
    fill_price: float
    cash_after: float


class PositionResponse(BaseModel):
    qty: float
    avg_price: float
    last_price: float
    market_value: float
    unrealized_pnl: float


class PortfolioResponse(BaseModel):
    cash: float
    equity: float
    daily_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    positions: dict[str, PositionResponse]


class DecisionPerformanceResponse(BaseModel):
    updated: int
    total: int
    resolved: int
    accuracy: float
    avg_return_pct: float
    total_hypothetical_pnl: float
    recent: list[dict]
