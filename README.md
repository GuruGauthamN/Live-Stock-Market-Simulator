# Stock Prediction Agent (Market + News)

This app runs a prediction-agent loop with live market data and configurable news feed:
- streams live market ticks,
- streams news (`newsapi` or mock fallback),
- trains on market + news sentiment features,
- generates invest/not-invest decisions,
- automatically paper-executes buy/sell decisions when enabled,
- can use an LLM API to make autonomous paper-trading decisions,
- tracks each decision and later evaluates if it was correct.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn stock_agent.main:app --app-dir src --reload --host 0.0.0.0 --port 8000
```

Open:
- Dashboard: `http://localhost:8000/`
- API docs: `http://localhost:8000/docs`

## Configure real news stream

In `.env`:

```env
NEWS_PROVIDER=newsapi
NEWS_API_KEY=your_newsapi_key
NEWS_INTERVAL_SECONDS=20
NEWS_LANGUAGE=en
AUTO_EXECUTE_DECISIONS=True
DECISION_TRADE_QTY=1
LLM_ENABLED=True
LLM_AUTO_RUN=False
LLM_API_KEY=your_llm_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4.1-mini
LLM_DECISION_INTERVAL_SECONDS=300
LLM_MIN_CONFIDENCE=0.60
```

If `NEWS_PROVIDER=newsapi` but key is missing/empty, app falls back to mock news stream.
You can verify active provider from `GET /health` (`news_provider` field).

## Core endpoints

- `GET /ticks/{symbol}` live chart points
- `GET /news/{symbol}` latest news + sentiment
- `POST /train` train on market + news features
- `POST /backtest` dry-run on holdout set
- `POST /decision/{symbol}?horizon_minutes=30` invest/not-invest recommendation
- `POST /llm/agent/run?horizon_minutes=30` run the LLM agent across configured symbols once
- `GET /llm/agent/status` check LLM config and current paper P/L
- `GET /decision/performance` check if prior decisions were right
- `POST /trade/{symbol}?action=buy&qty=1` paper execution
- `GET /portfolio` live realized/unrealized P/L

## LLM Automation

Set `LLM_ENABLED=True` and provide `LLM_API_KEY` to enable manual LLM decisions from the dashboard.
Set `LLM_AUTO_RUN=True` to let the background agent run every `LLM_DECISION_INTERVAL_SECONDS`.
The agent is still paper trading only; it does not send real broker orders.

## How prediction accuracy is measured

When you call `/decision/{symbol}` or `/llm/agent/run`:
1. agent stores entry price + target time (`horizon_minutes`)
2. if `AUTO_EXECUTE_DECISIONS=True`, buy/sell decisions are paper-executed immediately
3. later `/decision/performance` checks price at/after target time
4. decision is marked correct if:
- buy and return > 0
- sell and return < 0
- hold and absolute return stays small
5. the endpoint reports cumulative `total_hypothetical_pnl` for resolved decisions

## Notes

- NewsAPI polling is interval-based (near-real-time), not websocket streaming.
- This is for paper trading research, not investment advice.
