from __future__ import annotations

import asyncio
import json
import random
from collections import deque
from datetime import datetime, timezone
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from pydantic import BaseModel


class NewsItem(BaseModel):
    timestamp: datetime
    symbol: str
    headline: str
    sentiment: float
    source: str = "mock"


class MockNewsStream:
    POSITIVE = [
        "beats earnings estimates",
        "announces strong guidance",
        "expands strategic partnership",
        "launches high-demand product",
    ]
    NEGATIVE = [
        "faces regulatory pressure",
        "misses revenue forecast",
        "cuts forward guidance",
        "reports supply chain issues",
    ]

    def __init__(self, symbols: list[str], interval_seconds: int = 20) -> None:
        self.symbols = [s.upper() for s in symbols]
        self.interval_seconds = interval_seconds
        self.provider_name = "mock"

    async def run(self, on_news) -> None:
        while True:
            now = datetime.utcnow()
            symbol = random.choice(self.symbols)
            sentiment = random.uniform(-1.0, 1.0)
            if sentiment >= 0:
                phrase = random.choice(self.POSITIVE)
            else:
                phrase = random.choice(self.NEGATIVE)
            item = NewsItem(
                timestamp=now,
                symbol=symbol,
                headline=f"{symbol} {phrase}",
                sentiment=round(sentiment, 4),
                source="mock",
            )
            on_news(item)
            await asyncio.sleep(self.interval_seconds)


class NewsApiStream:
    def __init__(
        self,
        symbols: list[str],
        api_key: str,
        interval_seconds: int = 20,
        language: str = "en",
    ) -> None:
        self.symbols = [s.upper() for s in symbols]
        self.api_key = api_key
        self.interval_seconds = max(8, interval_seconds)
        self.language = language
        self.provider_name = "newsapi"
        self._seen = deque(maxlen=2000)
        self._seen_set: set[str] = set()
        self._symbol_idx = 0
        self._company_alias = {
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOGL": "Google",
            "GOOG": "Google",
            "AMZN": "Amazon",
            "TSLA": "Tesla",
            "META": "Meta",
            "NVDA": "Nvidia",
        }

    async def run(self, on_news) -> None:
        while True:
            try:
                symbol = self.symbols[self._symbol_idx % len(self.symbols)]
                self._symbol_idx += 1
                for item in self._fetch_symbol_news(symbol):
                    on_news(item)
            except Exception:
                pass
            await asyncio.sleep(self.interval_seconds)

    def _fetch_symbol_news(self, symbol: str) -> list[NewsItem]:
        term = self._company_alias.get(symbol, symbol)
        query = f"({term} OR {symbol}) AND (stock OR shares OR market OR earnings)"
        url = (
            "https://newsapi.org/v2/everything?"
            f"q={quote_plus(query)}&"
            f"language={quote_plus(self.language)}&"
            "sortBy=publishedAt&pageSize=10"
        )
        request = Request(url, headers={"X-Api-Key": self.api_key})
        with urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))

        articles = payload.get("articles", []) if isinstance(payload, dict) else []
        results: list[NewsItem] = []
        for article in articles:
            title = str(article.get("title") or "").strip()
            if not title:
                continue
            ident = str(article.get("url") or f"{symbol}:{title}")
            if ident in self._seen_set:
                continue
            self._remember(ident)

            published = article.get("publishedAt")
            published_dt = _parse_dt(published)
            source = (article.get("source") or {}).get("name") or "newsapi"
            text = " ".join(
                [
                    title,
                    str(article.get("description") or ""),
                    str(article.get("content") or ""),
                ]
            )
            sentiment = _score_sentiment(text)
            results.append(
                NewsItem(
                    timestamp=published_dt,
                    symbol=symbol,
                    headline=title,
                    sentiment=sentiment,
                    source=str(source),
                )
            )
        return results

    def _remember(self, ident: str) -> None:
        if len(self._seen) == self._seen.maxlen:
            dropped = self._seen.popleft()
            self._seen_set.discard(dropped)
        self._seen.append(ident)
        self._seen_set.add(ident)


def build_news_stream(
    *,
    symbols: list[str],
    provider: str,
    interval_seconds: int,
    api_key: str,
    language: str,
):
    provider = (provider or "mock").strip().lower()
    if provider == "newsapi" and api_key.strip():
        return NewsApiStream(
            symbols=symbols,
            api_key=api_key.strip(),
            interval_seconds=interval_seconds,
            language=language,
        )
    return MockNewsStream(symbols=symbols, interval_seconds=interval_seconds)


def _parse_dt(value: str | None) -> datetime:
    if not value:
        return datetime.utcnow()
    text = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except ValueError:
        return datetime.utcnow()


def _score_sentiment(text: str) -> float:
    if not text:
        return 0.0
    words = text.lower()
    positive = {
        "beat",
        "beats",
        "bullish",
        "upgrade",
        "growth",
        "strong",
        "surge",
        "gain",
        "record",
        "profit",
        "outperform",
        "partnership",
        "expands",
    }
    negative = {
        "miss",
        "misses",
        "downgrade",
        "lawsuit",
        "probe",
        "fraud",
        "weak",
        "decline",
        "drop",
        "fall",
        "loss",
        "cut",
        "risk",
        "warning",
    }
    pos_hits = sum(1 for w in positive if w in words)
    neg_hits = sum(1 for w in negative if w in words)
    score = (pos_hits - neg_hits) / 4.0
    return float(max(-1.0, min(1.0, score)))
