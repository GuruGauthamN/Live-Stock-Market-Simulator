from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from stock_agent.agent.llm_decider import LlmDecisionClient
from stock_agent.agent.tracker import DecisionRecord, DecisionTracker
from stock_agent.config import get_settings
from stock_agent.data.storage import MarketDataStore
from stock_agent.data.stream import MockMarketDataStream
from stock_agent.engine import SignalEngine
from stock_agent.models.features import compute_features, merge_news_features, ticks_to_bars
from stock_agent.models.predict import SignalModel
from stock_agent.models.train import (
    build_training_frame,
    run_time_split_backtest,
    train_model,
)
from stock_agent.news.storage import NewsStore
from stock_agent.news.stream import build_news_stream
from stock_agent.schemas import (
    BacktestResponse,
    DecisionPerformanceResponse,
    DecisionResponse,
    LlmAgentRunResponse,
    PortfolioResponse,
    TradeResponse,
    TrainResponse,
)
from stock_agent.trading.paper import PaperBroker

settings = get_settings()
store = MarketDataStore(settings.market_data_file)
news_store = NewsStore(settings.news_data_file)
decision_tracker = DecisionTracker(settings.decision_file)
model = SignalModel(settings.model_file)
model.load()
broker = PaperBroker(
    starting_cash=settings.starting_cash,
    max_position_value=settings.max_position_value,
    max_daily_loss=settings.max_daily_loss,
)
engine = SignalEngine(settings=settings, store=store, model=model)
llm_client = LlmDecisionClient(
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
    model=settings.llm_model,
)
stream = MockMarketDataStream(
    symbols=settings.symbol_list, interval_seconds=settings.stream_interval_seconds
)
news_stream = build_news_stream(
    symbols=settings.symbol_list,
    provider=settings.news_provider,
    interval_seconds=settings.news_interval_seconds,
    api_key=settings.news_api_key,
    language=settings.news_language,
)

stream_task: asyncio.Task | None = None
news_task: asyncio.Task | None = None
llm_agent_task: asyncio.Task | None = None


def latest_prices() -> dict[str, float]:
    prices: dict[str, float] = {}
    for symbol in settings.symbol_list:
        ticks = store.load_ticks(symbol)
        if not ticks.empty:
            prices[symbol] = float(ticks.iloc[-1]["price"])
    return prices


def latest_news_sentiment(symbol: str, minutes: int = 30) -> float:
    news = news_store.load_news(symbol)
    if news.empty:
        return 0.0
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    recent = news[news["timestamp"] >= cutoff]
    if recent.empty:
        return 0.0
    return float(recent["sentiment"].mean())


def collect_symbol_features() -> tuple[dict[str, object], list[str]]:
    features_by_symbol = {}
    symbols_ready: list[str] = []
    for symbol in settings.symbol_list:
        ticks = store.load_ticks(symbol)
        bars_df = ticks_to_bars(ticks, bar_seconds=60)
        if len(bars_df) < settings.model_min_bars:
            continue
        feats = compute_features(bars_df)
        if feats.empty:
            continue
        symbol_news = news_store.load_news(symbol)
        feats = merge_news_features(feats, symbol_news)
        features_by_symbol[symbol] = feats
        symbols_ready.append(symbol)
    return features_by_symbol, symbols_ready


def build_llm_context(symbol: str) -> dict:
    symbol = symbol.upper()
    ticks_df = store.load_ticks(symbol)
    news_df = news_store.load_news(symbol)
    prices = latest_prices()
    portfolio_snapshot = broker.position_snapshot(prices)
    recent_ticks = []
    if not ticks_df.empty:
        recent_ticks = [
            {
                "timestamp": row.timestamp.isoformat(),
                "price": float(row.price),
                "volume": int(row.volume),
            }
            for row in ticks_df.tail(20).itertuples(index=False)
        ]
    recent_news = []
    if not news_df.empty:
        recent_news = [
            {
                "timestamp": row.timestamp.isoformat(),
                "headline": row.headline,
                "sentiment": float(row.sentiment),
                "source": row.source,
            }
            for row in news_df.tail(10).itertuples(index=False)
        ]
    sig = engine.signal_for_symbol(symbol)
    return {
        "symbol": symbol,
        "now_utc": datetime.utcnow().isoformat(),
        "latest_price": prices.get(symbol),
        "recent_ticks": recent_ticks,
        "recent_news": recent_news,
        "news_sentiment_30m": latest_news_sentiment(symbol),
        "model_signal": sig.model_dump(),
        "portfolio": {
            "cash": broker.cash,
            "equity": broker.equity(prices),
            "realized_pnl": broker.realized_pnl,
            "unrealized_pnl": broker.unrealized_pnl(prices),
            "total_pnl": broker.total_pnl(prices),
            "positions": portfolio_snapshot,
        },
        "risk_rules": {
            "max_position_value": settings.max_position_value,
            "max_daily_loss": settings.max_daily_loss,
            "min_llm_confidence_to_trade": settings.llm_min_confidence,
            "paper_trade_only": True,
        },
    }


def execute_recorded_decision(
    *,
    symbol: str,
    action: str,
    confidence: float,
    expected_return: float,
    news_sentiment: float,
    horizon_minutes: int,
    qty: int,
    reason: str,
) -> DecisionResponse:
    symbol = symbol.upper()
    ticks_df = store.load_ticks(symbol)
    if ticks_df.empty:
        raise HTTPException(status_code=404, detail=f"No price data for symbol {symbol}.")

    now = datetime.utcnow()
    price = float(ticks_df.iloc[-1]["price"])
    final_action = action if confidence >= settings.llm_min_confidence else "hold"
    invest = final_action == "buy"
    trade_status = "not_executed"
    fill_price = None
    portfolio_total_pnl = None
    if settings.auto_execute_decisions and final_action in {"buy", "sell"}:
        try:
            fill = broker.place_order(
                symbol=symbol,
                side=final_action,
                qty=qty,
                price=price,
                prices=latest_prices(),
            )
            trade_status = "filled"
            fill_price = fill.price
            portfolio_total_pnl = broker.total_pnl(latest_prices())
        except ValueError as exc:
            trade_status = f"rejected: {exc}"

    record = DecisionRecord(
        decision_id=str(uuid4()),
        created_at=now.isoformat(),
        symbol=symbol,
        invest=invest,
        action=final_action,
        confidence=confidence,
        entry_price=price,
        qty=qty,
        horizon_minutes=horizon_minutes,
        target_time=(now + timedelta(minutes=horizon_minutes)).isoformat(),
    )
    decision_tracker.add(record)
    return DecisionResponse(
        decision_id=record.decision_id,
        symbol=symbol,
        invest=invest,
        action=final_action,
        confidence=confidence,
        expected_return=expected_return,
        news_sentiment=news_sentiment,
        horizon_minutes=horizon_minutes,
        qty=qty,
        trade_status=trade_status,
        fill_price=fill_price,
        portfolio_total_pnl=portfolio_total_pnl,
        reason=reason,
    )


async def run_llm_agent_loop() -> None:
    while True:
        try:
            run_llm_agent_once()
            decision_tracker.evaluate_pending(store)
        except Exception:
            pass
        await asyncio.sleep(max(30, settings.llm_decision_interval_seconds))


def run_llm_agent_once(horizon_minutes: int = 30) -> LlmAgentRunResponse:
    if not settings.llm_enabled:
        raise HTTPException(status_code=400, detail="LLM agent is disabled in .env.")
    if not llm_client.enabled():
        raise HTTPException(status_code=400, detail="LLM_API_KEY is not configured.")

    decisions: list[DecisionResponse] = []
    for symbol in settings.symbol_list:
        context = build_llm_context(symbol)
        llm_decision = llm_client.decide(context)
        decisions.append(
            execute_recorded_decision(
                symbol=symbol,
                action=llm_decision.action,
                confidence=llm_decision.confidence,
                expected_return=0.0,
                news_sentiment=float(context["news_sentiment_30m"]),
                horizon_minutes=horizon_minutes,
                qty=settings.decision_trade_qty,
                reason=f"llm={settings.llm_model}: {llm_decision.reason}",
            )
        )
    prices = latest_prices()
    return LlmAgentRunResponse(
        status="completed",
        decisions=decisions,
        portfolio_total_pnl=broker.total_pnl(prices),
        realized_pnl=broker.realized_pnl,
        unrealized_pnl=broker.unrealized_pnl(prices),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global stream_task, news_task, llm_agent_task
    stream_task = asyncio.create_task(stream.run(store.append_tick))
    news_task = asyncio.create_task(news_stream.run(news_store.append_news))
    if settings.llm_enabled and settings.llm_auto_run and llm_client.enabled():
        llm_agent_task = asyncio.create_task(run_llm_agent_loop())
    yield
    for task in [stream_task, news_task, llm_agent_task]:
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


app = FastAPI(title=settings.app_name, lifespan=lifespan)
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(static_dir / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "symbols": settings.symbol_list,
        "model_loaded": model.is_loaded(),
        "news_provider": getattr(news_stream, "provider_name", "unknown"),
        "llm_enabled": settings.llm_enabled,
        "llm_auto_run": settings.llm_auto_run,
        "llm_model": settings.llm_model,
    }


@app.get("/symbols")
def symbols() -> dict:
    available = sorted(set(settings.symbol_list + store.list_symbols()))
    return {"symbols": available}


@app.get("/ticks/{symbol}")
def ticks(symbol: str, seconds: int = 300) -> dict:
    symbol = symbol.upper()
    df = store.load_ticks(symbol)
    if df.empty:
        return {"symbol": symbol, "ticks": []}
    cutoff = datetime.utcnow() - timedelta(seconds=max(5, seconds))
    df = df[df["timestamp"] >= cutoff].tail(max(20, seconds * 2))
    rows = [
        {
            "timestamp": row.timestamp.isoformat(),
            "price": float(row.price),
            "volume": int(row.volume),
        }
        for row in df.itertuples(index=False)
    ]
    return {"symbol": symbol, "ticks": rows}


@app.get("/news/{symbol}")
def news(symbol: str, minutes: int = 120, limit: int = 30) -> dict:
    symbol = symbol.upper()
    df = news_store.load_news(symbol)
    if df.empty:
        return {"symbol": symbol, "sentiment": 0.0, "items": []}
    cutoff = datetime.utcnow() - timedelta(minutes=max(1, minutes))
    recent = df[df["timestamp"] >= cutoff].tail(max(1, limit))
    rows = [
        {
            "timestamp": row.timestamp.isoformat(),
            "headline": row.headline,
            "sentiment": float(row.sentiment),
            "source": row.source,
        }
        for row in recent.itertuples(index=False)
    ]
    avg_sentiment = float(recent["sentiment"].mean()) if not recent.empty else 0.0
    return {"symbol": symbol, "sentiment": avg_sentiment, "items": rows}


@app.get("/bars/{symbol}")
def bars(symbol: str, seconds: int = 1800) -> dict:
    ticks_df = store.load_ticks(symbol.upper())
    if ticks_df.empty:
        return {"symbol": symbol.upper(), "bars": []}
    bars_df = ticks_to_bars(ticks_df, bar_seconds=60).tail(max(1, seconds // 60))
    rows = []
    for idx, row in bars_df.iterrows():
        rows.append(
            {
                "timestamp": idx.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            }
        )
    return {"symbol": symbol.upper(), "bars": rows}


@app.post("/train", response_model=TrainResponse)
def train() -> TrainResponse:
    features_by_symbol, symbols_trained = collect_symbol_features()
    frame = build_training_frame(
        feature_frames=features_by_symbol,
        horizon_bars=settings.prediction_horizon_bars,
    )
    if frame.empty:
        raise HTTPException(
            status_code=400,
            detail="No training frame available. Wait for more market/news data.",
        )
    result = train_model(frame, settings.model_file)
    model.load()
    return TrainResponse(
        status="trained",
        symbols_trained=symbols_trained,
        rows_used=result.rows_used,
    )


@app.post("/backtest", response_model=BacktestResponse)
def backtest(test_ratio: float = 0.2) -> BacktestResponse:
    features_by_symbol, _ = collect_symbol_features()
    frame = build_training_frame(
        feature_frames=features_by_symbol,
        horizon_bars=settings.prediction_horizon_bars,
    )
    if frame.empty:
        raise HTTPException(
            status_code=400,
            detail="No training frame available. Wait for more market/news data.",
        )

    try:
        result = run_time_split_backtest(
            labeled=frame,
            model_path=settings.model_file,
            test_ratio=test_ratio,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model.load()
    return BacktestResponse(
        status="backtest_complete",
        train_rows=result.train_rows,
        test_rows=result.test_rows,
        accuracy=result.accuracy,
        total_pnl=result.total_pnl,
        trades=result.trades,
        winning_trades=result.winning_trades,
        losing_trades=result.losing_trades,
        test_start=result.test_start,
        test_end=result.test_end,
        sample_predictions=result.sample_predictions,
    )


@app.get("/signal/{symbol}")
def signal(symbol: str):
    return engine.signal_for_symbol(symbol.upper())


@app.post("/decision/{symbol}", response_model=DecisionResponse)
def decision(
    symbol: str,
    horizon_minutes: int = 30,
    qty: int | None = None,
) -> DecisionResponse:
    symbol = symbol.upper()
    ticks_df = store.load_ticks(symbol)
    if ticks_df.empty:
        raise HTTPException(status_code=404, detail=f"No price data for symbol {symbol}.")

    sig = engine.signal_for_symbol(symbol)
    news_sent = latest_news_sentiment(symbol)
    buy_allowed = (
        sig.action == "buy"
        and sig.confidence >= settings.signal_threshold
        and news_sent >= settings.invest_news_floor
    )
    sell_allowed = (
        sig.action == "sell"
        and sig.confidence >= settings.signal_threshold
        and news_sent <= abs(settings.invest_news_floor)
    )
    final_action = "buy" if buy_allowed else "sell" if sell_allowed else "hold"
    reason = (
        f"model_action={sig.action}, decision_action={final_action}, "
        f"confidence={sig.confidence:.3f}, "
        f"news_sentiment={news_sent:.3f}"
    )
    return execute_recorded_decision(
        symbol=symbol,
        action=final_action,
        confidence=sig.confidence,
        expected_return=sig.expected_return,
        news_sentiment=news_sent,
        horizon_minutes=horizon_minutes,
        qty=max(1, qty or settings.decision_trade_qty),
        reason=reason,
    )


@app.get("/llm/agent/status")
def llm_agent_status() -> dict:
    prices = latest_prices()
    return {
        "enabled": settings.llm_enabled,
        "auto_run": settings.llm_auto_run,
        "configured": llm_client.enabled(),
        "model": settings.llm_model,
        "interval_seconds": settings.llm_decision_interval_seconds,
        "portfolio_total_pnl": broker.total_pnl(prices),
        "realized_pnl": broker.realized_pnl,
        "unrealized_pnl": broker.unrealized_pnl(prices),
    }


@app.post("/llm/agent/run", response_model=LlmAgentRunResponse)
def llm_agent_run(horizon_minutes: int = 30) -> LlmAgentRunResponse:
    try:
        return run_llm_agent_once(horizon_minutes=horizon_minutes)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/decision/performance", response_model=DecisionPerformanceResponse)
def decision_performance() -> DecisionPerformanceResponse:
    summary = decision_tracker.evaluate_pending(store)
    return DecisionPerformanceResponse(**summary)


@app.post("/trade/{symbol}", response_model=TradeResponse)
def trade(symbol: str, action: str, qty: int = 1) -> TradeResponse:
    symbol = symbol.upper()
    ticks_df = store.load_ticks(symbol)
    if ticks_df.empty:
        raise HTTPException(status_code=404, detail=f"No price data for symbol {symbol}.")
    price = float(ticks_df.iloc[-1]["price"])
    prices = latest_prices()
    try:
        fill = broker.place_order(
            symbol=symbol,
            side=action,
            qty=qty,
            price=price,
            prices=prices,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TradeResponse(
        status="filled",
        symbol=fill.symbol,
        side=fill.side,
        qty=fill.qty,
        fill_price=fill.price,
        cash_after=fill.cash_after,
    )


@app.get("/portfolio", response_model=PortfolioResponse)
def portfolio() -> PortfolioResponse:
    prices = latest_prices()
    broker.maybe_roll_day(prices)
    return PortfolioResponse(
        cash=broker.cash,
        equity=broker.equity(prices),
        daily_pnl=broker.daily_pnl(prices),
        realized_pnl=broker.realized_pnl,
        unrealized_pnl=broker.unrealized_pnl(prices),
        total_pnl=broker.total_pnl(prices),
        positions=broker.position_snapshot(prices),
    )
