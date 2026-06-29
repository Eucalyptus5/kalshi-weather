from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from bot.lag.event_study import OrderbookSnapshotRow
from bot.lag.ws_book import book_state_at


@dataclass(frozen=True, slots=True)
class Comparison:
    ticker: str
    snapshot_at: datetime
    rest: OrderbookSnapshotRow
    ws: OrderbookSnapshotRow | None
    verdict: str
    gap_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SeriesAgreement:
    label: str
    aligned_n: int
    agree_n: int
    disagree_n: int
    no_coverage_n: int
    gap_n: int
    fraction: Decimal | None


@dataclass(frozen=True, slots=True)
class AgreementReport:
    comparisons: list[Comparison]
    per_series: list[SeriesAgreement]
    pooled: SeriesAgreement
    disagreements: list[Comparison]


_REST_ROWS = (
    "SELECT ticker, snapshot_at, yes_bid, yes_ask, no_bid, no_ask, "
    "yes_ask_depth, yes_bid_depth, no_ask_depth, no_bid_depth "
    "FROM orderbook_snapshots WHERE ticker LIKE ? "
    "ORDER BY ticker, snapshot_at, id"
)


def check_rest_agreement(db_path: Path, series: list[str]) -> AgreementReport:
    conn = sqlite3.connect(f"file:{db_path.absolute()}?mode=ro", uri=True)
    try:
        rows_by_series = {s: conn.execute(_REST_ROWS, (f"{s}-%",)).fetchall() for s in series}
    finally:
        conn.close()

    comparisons: list[Comparison] = []
    per_series: list[SeriesAgreement] = []
    for s in sorted(series):
        series_comparisons = [_compare(db_path, row) for row in rows_by_series[s]]
        comparisons.extend(series_comparisons)
        per_series.append(_aggregate(s, series_comparisons))

    return AgreementReport(
        comparisons=comparisons,
        per_series=per_series,
        pooled=_aggregate("pooled", comparisons),
        disagreements=[c for c in comparisons if c.verdict == "disagree"],
    )


def _compare(db_path: Path, row: tuple) -> Comparison:
    rest = OrderbookSnapshotRow(
        ticker=row[0],
        snapshot_at=datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc),
        yes_bid=Decimal(str(row[2])),
        yes_ask=Decimal(str(row[3])),
        no_bid=Decimal(str(row[4])),
        no_ask=Decimal(str(row[5])),
        yes_ask_depth=row[6],
        yes_bid_depth=row[7],
        no_ask_depth=row[8],
        no_bid_depth=row[9],
    )
    try:
        ws = book_state_at(db_path, rest.ticker, rest.snapshot_at)
    except ValueError as exc:
        return Comparison(
            ticker=rest.ticker,
            snapshot_at=rest.snapshot_at,
            rest=rest,
            ws=None,
            verdict="gap",
            gap_reason=str(exc),
        )
    if ws is None:
        return Comparison(
            ticker=rest.ticker,
            snapshot_at=rest.snapshot_at,
            rest=rest,
            ws=None,
            verdict="no_coverage",
        )
    agree = rest.yes_bid == ws.yes_bid and rest.yes_ask == ws.yes_ask
    return Comparison(
        ticker=rest.ticker,
        snapshot_at=rest.snapshot_at,
        rest=rest,
        ws=ws,
        verdict="agree" if agree else "disagree",
    )


def _aggregate(label: str, comparisons: list[Comparison]) -> SeriesAgreement:
    agree_n = sum(1 for c in comparisons if c.verdict == "agree")
    disagree_n = sum(1 for c in comparisons if c.verdict == "disagree")
    aligned_n = agree_n + disagree_n
    return SeriesAgreement(
        label=label,
        aligned_n=aligned_n,
        agree_n=agree_n,
        disagree_n=disagree_n,
        no_coverage_n=sum(1 for c in comparisons if c.verdict == "no_coverage"),
        gap_n=sum(1 for c in comparisons if c.verdict == "gap"),
        fraction=Decimal(agree_n) / Decimal(aligned_n) if aligned_n else None,
    )
