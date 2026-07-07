import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bot.lag.ws_book import WsGapError
from bot.replay.forward_pass import _DB_TS, JsonState, SourceRow, _decode_ts


_GAPS = "SELECT id, ticker, detected_at, last_seq, reason FROM ws_gaps ORDER BY detected_at, id"


@dataclass(frozen=True, slots=True)
class GapRow:
    id: int
    ticker: str
    detected_at: datetime
    last_seq: int
    reason: str


@dataclass(frozen=True, slots=True)
class ExclusionWindow:
    gap_id: int
    ticker: str
    start: datetime | None
    end: datetime | None
    detected_at: datetime
    last_seq: int
    reason: str

    def covers(self, ticker: str, t: datetime) -> bool:
        # An empty ticker on a ws_gaps row is connection-wide: it applies to every ticker.
        if self.ticker and self.ticker != ticker:
            return False
        if self.start is not None and t <= self.start:
            return False
        return self.end is None or t < self.end


def read_gap_rows(db_path: Path) -> list[GapRow]:
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        raw = conn.execute(_GAPS).fetchall()
    finally:
        conn.close()
    return [
        GapRow(
            id=row[0],
            ticker=row[1],
            detected_at=_decode_ts(row[2]),
            last_seq=row[3],
            reason=row[4],
        )
        for row in raw
    ]


def check_excluded(windows: Sequence[ExclusionWindow], ticker: str, t: datetime) -> None:
    for window in windows:
        if window.covers(ticker, t):
            raise WsGapError(
                f"ws gap {window.gap_id} excludes {ticker} at {t.isoformat()}: "
                f"reason={window.reason} detected_at={window.detected_at} "
                f"window={window.start}..{window.end}"
            )


class ExclusionInventory:
    name = "exclusions"

    def __init__(self, gaps: Sequence[GapRow]) -> None:
        self._gaps = sorted(gaps, key=lambda row: (row.detected_at, row.id))
        self._pending = list(self._gaps)
        self._edges: dict[int, tuple[datetime | None, datetime]] = {}
        self._last: datetime | None = None

    def observe(self, row: SourceRow) -> None:
        # detected_at is stamped at persist, after the reconnect slept its backoff, so the
        # disconnect is bounded by the last row before it and the resume by the first row after.
        while self._pending and self._pending[0].detected_at <= row.received_at:
            self._edges[self._pending.pop(0).id] = (self._last, row.received_at)
        self._last = row.received_at

    def windows(self) -> list[ExclusionWindow]:
        out: list[ExclusionWindow] = []
        for row in self._gaps:
            start, end = self._edges.get(row.id, (self._last, None))
            out.append(
                ExclusionWindow(
                    gap_id=row.id,
                    ticker=row.ticker,
                    start=start,
                    end=end,
                    detected_at=row.detected_at,
                    last_seq=row.last_seq,
                    reason=row.reason,
                )
            )
        return out

    def state(self) -> JsonState:
        return {
            "last": _format(self._last),
            "edges": {
                str(gap_id): [_format(start), _format(end)]
                for gap_id, (start, end) in self._edges.items()
            },
        }

    def restore(self, state: JsonState) -> None:
        self._last = _parse_optional(state["last"])
        self._edges = {
            int(gap_id): (_parse_optional(edges[0]), _decode_ts(edges[1]))
            for gap_id, edges in state["edges"].items()
        }
        self._pending = [row for row in self._gaps if row.id not in self._edges]


@dataclass(frozen=True, slots=True)
class SeqBoundary:
    id: int
    received_at: datetime
    prev_seq: int
    seq: int
    kind: str


class SeqBoundaryDetector:
    name = "seq"

    def __init__(self) -> None:
        self._last: tuple[datetime, int] | None = None
        self._boundaries: list[SeqBoundary] = []

    def observe(self, row: SourceRow) -> None:
        key = (row.received_at, row.seq)
        if key == self._last:
            return
        prev = self._last
        self._last = key
        if prev is None or row.seq == prev[1] + 1:
            return
        self._boundaries.append(
            SeqBoundary(
                id=row.id,
                received_at=row.received_at,
                prev_seq=prev[1],
                seq=row.seq,
                kind="skip" if row.seq > prev[1] else "resubscribe",
            )
        )

    def boundaries(self) -> list[SeqBoundary]:
        return list(self._boundaries)

    def skips(self) -> int:
        return sum(1 for boundary in self._boundaries if boundary.kind == "skip")

    def resubscribes(self) -> int:
        return sum(1 for boundary in self._boundaries if boundary.kind == "resubscribe")

    def state(self) -> JsonState:
        return {
            "last": None if self._last is None else [_format(self._last[0]), self._last[1]],
            "boundaries": [
                {
                    "id": boundary.id,
                    "received_at": _format(boundary.received_at),
                    "prev_seq": boundary.prev_seq,
                    "seq": boundary.seq,
                    "kind": boundary.kind,
                }
                for boundary in self._boundaries
            ],
        }

    def restore(self, state: JsonState) -> None:
        last = state["last"]
        self._last = None if last is None else (_decode_ts(last[0]), last[1])
        self._boundaries = [
            SeqBoundary(
                id=boundary["id"],
                received_at=_decode_ts(boundary["received_at"]),
                prev_seq=boundary["prev_seq"],
                seq=boundary["seq"],
                kind=boundary["kind"],
            )
            for boundary in state["boundaries"]
        ]


@dataclass(frozen=True, slots=True)
class TickerCoverage:
    ticker: str
    rows: int
    first_received_at: datetime
    last_received_at: datetime


class TickerInventory:
    name = "tickers"

    def __init__(self) -> None:
        self._rows: dict[str, int] = {}
        self._first: dict[str, datetime] = {}
        self._last: dict[str, datetime] = {}

    def observe(self, row: SourceRow) -> None:
        ticker = row.ticker
        seen = self._rows.get(ticker, 0)
        self._rows[ticker] = seen + 1
        if seen == 0:
            self._first[ticker] = row.received_at
        self._last[ticker] = row.received_at

    def coverage(self) -> list[TickerCoverage]:
        return [
            TickerCoverage(
                ticker=ticker,
                rows=self._rows[ticker],
                first_received_at=self._first[ticker],
                last_received_at=self._last[ticker],
            )
            for ticker in sorted(self._rows)
        ]

    def state(self) -> JsonState:
        return {
            "rows": self._rows,
            "first": {ticker: _format(at) for ticker, at in self._first.items()},
            "last": {ticker: _format(at) for ticker, at in self._last.items()},
        }

    def restore(self, state: JsonState) -> None:
        self._rows = dict(state["rows"])
        self._first = {ticker: _decode_ts(at) for ticker, at in state["first"].items()}
        self._last = {ticker: _decode_ts(at) for ticker, at in state["last"].items()}


@dataclass(frozen=True, slots=True)
class PassInventory:
    gap_rows: int
    seq_skips: int
    seq_resubscribes: int
    windows: tuple[ExclusionWindow, ...]
    boundaries: tuple[SeqBoundary, ...]
    coverage: tuple[TickerCoverage, ...]


def build_inventory(
    exclusions: ExclusionInventory,
    seq: SeqBoundaryDetector,
    tickers: TickerInventory,
) -> PassInventory:
    windows = exclusions.windows()
    return PassInventory(
        gap_rows=len(windows),
        seq_skips=seq.skips(),
        seq_resubscribes=seq.resubscribes(),
        windows=tuple(windows),
        boundaries=tuple(seq.boundaries()),
        coverage=tuple(tickers.coverage()),
    )


def _format(value: datetime | None) -> str | None:
    return None if value is None else value.strftime(_DB_TS)


def _parse_optional(value: str | None) -> datetime | None:
    return None if value is None else _decode_ts(value)
