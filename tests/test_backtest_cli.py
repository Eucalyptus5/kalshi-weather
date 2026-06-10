from __future__ import annotations

import argparse
import asyncio
import csv
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.backtest.cli import build_parser, main, parse_lead
from bot.backtest.engine import BacktestConfig, iter_canonical_snapshots, run_replay
from bot.backtest.forecast_replay import StationSpec
from bot.backtest.ingest import CANONICAL_SCHEMA, ingest_weather_snapshots
from bot.backtest.normalize import CanonicalSnapshot
from bot.forecast.cdf import EnsembleCDF
from tests.fixtures.trevorjs_market_state_sample import make_market_state_table

REPO = Path(__file__).resolve().parents[1]
_LEAD = timedelta(hours=25)
_BASE_CLOSE = datetime(2026, 3, 31, 6, 0, tzinfo=timezone.utc)
_REPLAY_SCHEMA = pa.schema(
    [pa.field("snapshot_at", pa.timestamp("us", tz="UTC"))] + list(CANONICAL_SCHEMA)
)


class TailsStubReplay:
    def __init__(self) -> None:
        self._cdf = EnsembleCDF.from_members(np.array([40.0, 41.5, 43.0]), smoothing=0.65)

    async def replay(self, station: StationSpec, valid_date: date, as_of: datetime) -> EnsembleCDF:
        return self._cdf


class EdgeStubReplay:
    def __init__(self) -> None:
        self._cdf = EnsembleCDF.from_members(np.array([59.5, 61.5, 63.5]), smoothing=0.65)

    async def replay(self, station: StationSpec, valid_date: date, as_of: datetime) -> EnsembleCDF:
        return self._cdf


def _ingest_fixture_parquet(tmp_path: Path) -> Path:
    shard = tmp_path / "market_state.parquet"
    pq.write_table(make_market_state_table(), shard)
    out = tmp_path / "weather.parquet"
    ingest_weather_snapshots([shard], out, series=["KXHIGHDEN"])
    return out


def _replay_snapshot(ticker: str, close: datetime, result: str) -> CanonicalSnapshot:
    return CanonicalSnapshot(
        ticker=ticker,
        event_ticker="-".join(ticker.split("-")[:2]),
        series_ticker=ticker.split("-", 1)[0],
        status="finalized",
        result=result,
        yes_ask=Decimal("0.10"),
        yes_bid=Decimal("0.08"),
        no_ask=Decimal("0.92"),
        no_bid=Decimal("0.90"),
        last_price=Decimal("0.08"),
        volume=Decimal(0),
        volume_24h=Decimal(0),
        open_interest=Decimal(0),
        close_time=close,
        yes_bid_size=Decimal(3000),
        no_bid_size=Decimal(3000),
    )


def _write_replay_parquet(path: Path, result: str) -> None:
    tickers = (
        "KXHIGHDEN-26MAR30-B59.5",
        "KXHIGHDEN-26MAR31-B59.5",
        "KXHIGHDEN-26APR01-B59.5",
    )
    rows = []
    for i, ticker in enumerate(tickers):
        snap = _replay_snapshot(ticker, _BASE_CLOSE + timedelta(days=i), result)
        row = {name: getattr(snap, name) for name in CANONICAL_SCHEMA.names}
        row["snapshot_at"] = snap.close_time - _LEAD - timedelta(minutes=1)
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows, schema=_REPLAY_SCHEMA), path)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_dry_run_over_bt03_fixture_writes_report(tmp_path: Path) -> None:
    parquet = _ingest_fixture_parquet(tmp_path)
    out_dir = tmp_path / "report"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bot.backtest.cli",
            "--weather-parquet",
            str(parquet),
            "--strategy",
            "tails",
            "--bankroll",
            "20000",
            "--lead",
            "24h",
            "--out",
            str(out_dir),
            "--forecast-source",
            "tests.test_backtest_cli:TailsStubReplay",
            "--dry-run",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "orders.csv").exists()
    assert "orders=0" in proc.stdout
    assert "skips=4" in proc.stdout
    assert _csv_rows(out_dir / "orders.csv") == []
    summary = (out_dir / "summary.md").read_text()
    assert "KXHIGHDEN" in summary
    assert "verdict:" in summary


def test_orders_flow_through_to_report(tmp_path: Path) -> None:
    parquet = tmp_path / "weather.parquet"
    _write_replay_parquet(parquet, result="yes")
    out_dir = tmp_path / "report"

    config = BacktestConfig(lead=_LEAD, bankroll=Decimal("20000"))
    expected = asyncio.run(
        run_replay(list(iter_canonical_snapshots(parquet)), EdgeStubReplay(), "edge", config)
    )
    assert len(expected) > 0

    code = main(
        [
            "--weather-parquet",
            str(parquet),
            "--strategy",
            "edge",
            "--bankroll",
            "20000",
            "--lead",
            "25h",
            "--out",
            str(out_dir),
            "--forecast-source",
            "tests.test_backtest_cli:EdgeStubReplay",
        ]
    )

    assert code == 0
    rows = _csv_rows(out_dir / "orders.csv")
    assert len(rows) == len(expected)
    assert {r["market_ticker"] for r in rows} == {o.market_ticker for o in expected}
    summary = (out_dir / "summary.md").read_text()
    assert "verdict: survive" in summary


def test_losing_run_exits_nonzero_unless_dry_run(tmp_path: Path) -> None:
    parquet = tmp_path / "weather.parquet"
    _write_replay_parquet(parquet, result="no")
    argv = [
        "--weather-parquet",
        str(parquet),
        "--strategy",
        "edge",
        "--lead",
        "25h",
        "--out",
        str(tmp_path / "strict"),
        "--forecast-source",
        "tests.test_backtest_cli:EdgeStubReplay",
    ]

    assert main(argv) == 1

    argv[7] = str(tmp_path / "dry")
    assert main([*argv, "--dry-run"]) == 0


def test_strategy_outside_choices_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--weather-parquet",
                str(tmp_path / "w.parquet"),
                "--strategy",
                "momentum",
                "--lead",
                "24h",
                "--out",
                str(tmp_path / "report"),
            ]
        )
    assert excinfo.value.code == 2


def test_dry_run_requires_forecast_source(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--weather-parquet",
                str(tmp_path / "w.parquet"),
                "--strategy",
                "edge",
                "--lead",
                "24h",
                "--out",
                str(tmp_path / "report"),
                "--dry-run",
            ]
        )
    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("24h", timedelta(hours=24)),
        ("90m", timedelta(minutes=90)),
        ("2d", timedelta(days=2)),
    ],
)
def test_parse_lead_golden(raw: str, expected: timedelta) -> None:
    assert parse_lead(raw) == expected


@pytest.mark.parametrize("raw", ["24", "h24", "1w", "", "24hh"])
def test_parse_lead_rejects_malformed(raw: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_lead(raw)


def test_bankroll_lands_as_decimal() -> None:
    base = ["--weather-parquet", "w.parquet", "--strategy", "edge", "--lead", "24h", "--out", "r"]
    parser = build_parser()

    default = parser.parse_args(base).bankroll
    assert isinstance(default, Decimal)
    assert default == Decimal("20000")

    explicit = parser.parse_args([*base, "--bankroll", "12345.50"]).bankroll
    assert isinstance(explicit, Decimal)
    assert explicit == Decimal("12345.50")
