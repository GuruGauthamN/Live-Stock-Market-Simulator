from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class Fill:
    symbol: str
    side: str
    qty: int
    price: float
    cash_after: float


class PaperBroker:
    def __init__(
        self, starting_cash: float, max_position_value: float, max_daily_loss: float
    ) -> None:
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.max_position_value = float(max_position_value)
        self.max_daily_loss = float(max_daily_loss)
        self.positions: dict[str, dict[str, float]] = {}
        self.realized_pnl = 0.0
        self._day = date.today()
        self._day_start_equity = self.starting_cash

    def maybe_roll_day(self, prices: dict[str, float] | None = None) -> None:
        today = date.today()
        if today != self._day:
            self._day = today
            self._day_start_equity = self.equity(prices or {})

    def place_order(
        self, symbol: str, side: str, qty: int, price: float, prices: dict[str, float] | None = None
    ) -> Fill:
        mark_prices = dict(prices or {})
        mark_prices[symbol.upper()] = float(price)
        self.maybe_roll_day(mark_prices)

        symbol = symbol.upper()
        side = side.lower()
        if qty <= 0:
            raise ValueError("Quantity must be positive.")
        if side not in {"buy", "sell"}:
            raise ValueError("Side must be buy or sell.")
        if self.daily_pnl(mark_prices) <= -self.max_daily_loss:
            raise ValueError("Daily loss limit reached. Trading halted.")

        cost = qty * price
        pos = self.positions.get(symbol, {"qty": 0.0, "avg_price": 0.0})
        current_value = abs(pos["qty"] * price)

        if side == "buy":
            if cost > self.cash:
                raise ValueError("Insufficient cash.")
            if pos["qty"] < 0:
                cover_qty = min(qty, abs(pos["qty"]))
                self.realized_pnl += (pos["avg_price"] - price) * cover_qty
                new_qty = pos["qty"] + qty
                if new_qty < 0:
                    self.positions[symbol] = {
                        "qty": new_qty,
                        "avg_price": pos["avg_price"],
                    }
                elif new_qty == 0:
                    self.positions.pop(symbol, None)
                else:
                    if abs(new_qty * price) > self.max_position_value:
                        raise ValueError("Position value limit exceeded.")
                    self.positions[symbol] = {"qty": new_qty, "avg_price": price}
            else:
                if current_value + cost > self.max_position_value:
                    raise ValueError("Position value limit exceeded.")
                new_qty = pos["qty"] + qty
                new_avg = ((pos["qty"] * pos["avg_price"]) + cost) / new_qty
                self.positions[symbol] = {"qty": new_qty, "avg_price": new_avg}
            self.cash -= cost
        else:
            if pos["qty"] > 0:
                close_qty = min(qty, pos["qty"])
                self.realized_pnl += (price - pos["avg_price"]) * close_qty
                new_qty = pos["qty"] - qty
                if new_qty > 0:
                    self.positions[symbol] = {
                        "qty": new_qty,
                        "avg_price": pos["avg_price"],
                    }
                elif new_qty == 0:
                    self.positions.pop(symbol, None)
                else:
                    if abs(new_qty * price) > self.max_position_value:
                        raise ValueError("Position value limit exceeded.")
                    self.positions[symbol] = {"qty": new_qty, "avg_price": price}
            else:
                new_qty = pos["qty"] - qty
                if abs(new_qty * price) > self.max_position_value:
                    raise ValueError("Position value limit exceeded.")
                if pos["qty"] < 0:
                    new_avg = (
                        (abs(pos["qty"]) * pos["avg_price"]) + cost
                    ) / abs(new_qty)
                else:
                    new_avg = price
                self.positions[symbol] = {"qty": new_qty, "avg_price": new_avg}
            self.cash += cost

        return Fill(symbol=symbol, side=side, qty=qty, price=price, cash_after=self.cash)

    def equity(self, prices: dict[str, float]) -> float:
        value = self.cash
        for symbol, pos in self.positions.items():
            last = prices.get(symbol, pos["avg_price"])
            value += pos["qty"] * last
        return float(value)

    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        total = 0.0
        for symbol, pos in self.positions.items():
            last = prices.get(symbol, pos["avg_price"])
            total += (last - pos["avg_price"]) * pos["qty"]
        return float(total)

    def total_pnl(self, prices: dict[str, float]) -> float:
        return self.equity(prices) - self.starting_cash

    def daily_pnl(self, prices: dict[str, float]) -> float:
        return float(self.equity(prices) - self._day_start_equity)

    def position_snapshot(self, prices: dict[str, float]) -> dict[str, dict[str, float]]:
        snapshot: dict[str, dict[str, float]] = {}
        for symbol, pos in self.positions.items():
            last = float(prices.get(symbol, pos["avg_price"]))
            qty = float(pos["qty"])
            avg = float(pos["avg_price"])
            snapshot[symbol] = {
                "qty": qty,
                "avg_price": avg,
                "last_price": last,
                "market_value": qty * last,
                "unrealized_pnl": (last - avg) * qty,
            }
        return snapshot
