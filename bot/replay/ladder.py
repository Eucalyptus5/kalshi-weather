from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Context, Decimal, Inexact, InvalidOperation

from bot.lag.event_study import OrderbookSnapshotRow


_ZERO = Decimal("0")
_ONE = Decimal("1")
_PRICE_EXPONENT = Decimal("0.0001")
_SIZE_EXPONENT = Decimal("0.01")
_EXACT = Context(traps=[Inexact, InvalidOperation])


@dataclass(frozen=True, slots=True)
class BookEvent:
    received_at: datetime
    seq: int
    side: str
    price: str
    size: str
    is_snapshot: bool


class Ladder:
    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.levels: dict[str, dict[Decimal, Decimal]] = {"yes": {}, "no": {}}
        self.batch_key: tuple[datetime, int] | None = None
        self._live: dict[str, list[tuple[Decimal, Decimal]] | None] = {"yes": None, "no": None}

    def apply(self, event: BookEvent) -> None:
        self._live["yes"] = None
        self._live["no"] = None
        price = _quantized(self.ticker, "price", event.price, _PRICE_EXPONENT)
        size = _quantized(self.ticker, "size", event.size, _SIZE_EXPONENT)
        if event.is_snapshot:
            key = (event.received_at, event.seq)
            if key != self.batch_key:
                self.batch_key = key
                self.levels["yes"].clear()
                self.levels["no"].clear()
            self.levels[event.side][price] = size
            return
        levels = self.levels[event.side]
        total = levels.get(price, _ZERO) + size
        if total < _ZERO:
            raise ValueError(
                f"negative level for {self.ticker} seq={event.seq} "
                f"side={event.side} price={price} size={total}"
            )
        if total == _ZERO:
            levels.pop(price, None)
        else:
            levels[price] = total

    def live(self, side: str) -> list[tuple[Decimal, Decimal]]:
        cached = self._live[side]
        if cached is None:
            cached = sorted(
                ((price, size) for price, size in self.levels[side].items() if size > _ZERO),
                reverse=True,
            )
            self._live[side] = cached
        return cached

    def row(self, at: datetime) -> OrderbookSnapshotRow:
        yes_bid, yes_bid_depth = _best(self.live("yes"))
        no_bid, no_bid_depth = _best(self.live("no"))
        return OrderbookSnapshotRow(
            ticker=self.ticker,
            snapshot_at=at,
            yes_bid=yes_bid,
            yes_ask=_ONE - no_bid,
            no_bid=no_bid,
            no_ask=_ONE - yes_bid,
            yes_ask_depth=no_bid_depth,
            yes_bid_depth=yes_bid_depth,
            no_ask_depth=yes_bid_depth,
            no_bid_depth=no_bid_depth,
        )


def _quantized(ticker: str, field: str, value: str, exponent: Decimal) -> Decimal:
    try:
        return Decimal(value).quantize(exponent, context=_EXACT)
    except (Inexact, InvalidOperation) as exc:
        raise ValueError(
            f"{ticker} {field}={value} does not fit exponent {exponent.as_tuple().exponent}"
        ) from exc


# Deliberately not imported from bot.lag.ws_book: the parity check compares the two touch
# extractions against each other and shared code would make that comparison vacuous.
def _best(live: list[tuple[Decimal, Decimal]]) -> tuple[Decimal, int]:
    if not live:
        return _ZERO, 0
    price, size = live[0]
    return price, int(size)
