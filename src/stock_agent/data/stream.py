from __future__ import annotations

import asyncio
import random
from datetime import datetime

from stock_agent.schemas import MarketTick


class MockMarketDataStream:
    def __init__(self, symbols: list[str], interval_seconds: int = 1) -> None:
        self.symbols = [s.upper() for s in symbols]
        self.interval_seconds = interval_seconds
        self._last_prices = {s: random.uniform(80.0, 300.0) for s in self.symbols}

    async def run(self, on_tick) -> None:
        while True:
            now = datetime.utcnow()
            for symbol in self.symbols:
                drift = random.uniform(-0.003, 0.003)
                next_price = max(1.0, self._last_prices[symbol] * (1.0 + drift))
                self._last_prices[symbol] = next_price
                tick = MarketTick(
                    timestamp=now,
                    symbol=symbol,
                    price=round(next_price, 4),
                    volume=random.randint(50, 5000),
                )
                on_tick(tick)
            await asyncio.sleep(self.interval_seconds)
