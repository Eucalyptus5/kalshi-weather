import sqlite3
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from bot.backtest.depth_table import ALL_SERIES, lead_bucket_for, load_depth_table
from bot.backtest.engine import BacktestConfig, ReplaySnapshot, run_replay
from bot.backtest.forecast_replay import StationSpec
from bot.backtest.normalize import CanonicalSnapshot
from bot.forecast.cdf import EnsembleCDF

_LEAD = timedelta(hours=25)
_CLOSE = datetime(2026, 1, 16, 6, 0, tzinfo=timezone.utc)
_MEMBERS = np.array([59.5, 61.5, 63.5])


class FixedReplay:
    def __init__(self) -> None:
        self._cdf = EnsembleCDF.from_members(_MEMBERS, smoothing=0.65)

    async def replay(self, station: StationSpec, valid_date: date, as_of: datetime) -> EnsembleCDF:
        return self._cdf


def write_db(
    path: Path,
    markets: list[tuple[str, str, str]],
    snapshots: list[tuple[str, str, int, int]],
) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE markets (ticker TEXT PRIMARY KEY, series TEXT, close_time TEXT)")
    conn.execute(
        "CREATE TABLE orderbook_snapshots "
        "(ticker TEXT, snapshot_at TEXT, yes_bid_depth INTEGER, no_bid_depth INTEGER)"
    )
    conn.executemany("INSERT INTO markets VALUES (?, ?, ?)", markets)
    conn.executemany("INSERT INTO orderbook_snapshots VALUES (?, ?, ?, ?)", snapshots)
    conn.commit()
    conn.close()
    return path


def make_den_db(path: Path) -> Path:
    ticker = "KXHIGHDEN-26JAN15-B61.5"
    close = "2026-01-16 06:00:00+00:00"
    snapshot_at = "2026-01-15 00:00:00+00:00"
    return write_db(
        path,
        markets=[(ticker, "KXHIGHDEN", close)],
        snapshots=[
            (ticker, snapshot_at, 10, 99),
            (ticker, snapshot_at, 25, 99),
            (ticker, snapshot_at, 99, 80),
        ],
    )


def make_decision_row(ticker: str) -> ReplaySnapshot:
    snap = CanonicalSnapshot(
        ticker=ticker,
        event_ticker="-".join(ticker.split("-")[:2]),
        series_ticker=ticker.split("-", 1)[0],
        status="active",
        result="",
        yes_ask=Decimal("0.10"),
        yes_bid=Decimal("0.08"),
        no_ask=Decimal("0.92"),
        no_bid=Decimal("0.90"),
        last_price=Decimal("0.08"),
        volume=Decimal(0),
        volume_24h=Decimal(0),
        open_interest=Decimal(0),
        close_time=_CLOSE,
    )
    return ReplaySnapshot(snapshot_at=_CLOSE - _LEAD, snap=snap)


def test_median_of_min_depths_per_series_lead_bucket(tmp_path: Path) -> None:
    table = load_depth_table(make_den_db(tmp_path / "state.db"))

    assert table[("KXHIGHDEN", "24-72h")] == Decimal("25")
    assert table[(ALL_SERIES, "24-72h")] == Decimal("25")


def test_bucket_under_200_rows_falls_back_to_all_series_median(tmp_path: Path) -> None:
    den_ticker = "KXHIGHDEN-26JAN15-B61.5"
    aus_ticker = "KXHIGHAUS-26JAN15-B61.5"
    close = "2026-01-16 06:00:00+00:00"
    snapshot_at = "2026-01-15 00:00:00+00:00"
    db = write_db(
        tmp_path / "state.db",
        markets=[(den_ticker, "KXHIGHDEN", close), (aus_ticker, "KXHIGHAUS", close)],
        snapshots=[
            (den_ticker, snapshot_at, 10, 99),
            (den_ticker, snapshot_at, 25, 99),
            (den_ticker, snapshot_at, 99, 80),
        ]
        + [(aus_ticker, snapshot_at, 40, 99)] * 200,
    )

    table = load_depth_table(db)

    assert table[("KXHIGHAUS", "24-72h")] == Decimal("40")
    assert table[("KXHIGHDEN", "24-72h")] == Decimal("40")
    assert table[(ALL_SERIES, "24-72h")] == Decimal("40")


def test_missing_db_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_depth_table(tmp_path / "absent.db")


@pytest.mark.parametrize(
    ("lead", "bucket"),
    [
        (timedelta(hours=2), "<=2h"),
        (timedelta(hours=2, minutes=1), "2-8h"),
        (timedelta(hours=8), "2-8h"),
        (timedelta(hours=24), "8-24h"),
        (timedelta(hours=72), "24-72h"),
        (timedelta(hours=73), ">72h"),
    ],
)
def test_lead_bucket_boundaries(lead: timedelta, bucket: str) -> None:
    assert lead_bucket_for(lead) == bucket


async def test_snapshot_without_sizes_falls_back_to_series_lead_median(tmp_path: Path) -> None:
    config = BacktestConfig(
        lead=_LEAD,
        bankroll=Decimal("20000"),
        depth_table=load_depth_table(make_den_db(tmp_path / "state.db")),
    )

    orders = await run_replay(
        [make_decision_row("KXHIGHDEN-26JAN15-B61.5")], FixedReplay(), "edge", config
    )

    assert len(orders) == 1
    assert orders[0].depth_at_price == 25
    assert orders[0].depth_source == "series_lead_median"
    assert orders[0].lead_bucket == "24-72h"


async def test_series_absent_from_table_uses_all_series_lead_median(tmp_path: Path) -> None:
    config = BacktestConfig(
        lead=_LEAD,
        bankroll=Decimal("20000"),
        depth_table=load_depth_table(make_den_db(tmp_path / "state.db")),
    )

    orders = await run_replay(
        [make_decision_row("KXHIGHAUS-26JAN15-B61.5")], FixedReplay(), "edge", config
    )

    assert len(orders) == 1
    assert orders[0].depth_at_price == 25
    assert orders[0].depth_source == "all_series_lead_median"
    assert orders[0].lead_bucket == "24-72h"
