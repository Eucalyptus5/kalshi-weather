import argparse
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.lag.taker_side import Split
from scripts.taker_side_report import build_parser, run, tape_days
from tests.test_taker_side import (
    ACK_FRAME,
    DELTA_FRAME,
    TRADE_INDEX,
    TRADE_SCHEMA,
    trade_frame,
    trade_row,
    write_tape,
)


UTC = timezone.utc
SCOPE_START = datetime(2026, 7, 19, 5, tzinfo=UTC)
SCOPE_END = datetime(2026, 7, 21, 8, tzinfo=UTC)
TICKER = "KXHIGHNY-26JUL19-B79.5"
WINDOW_START = datetime(2026, 7, 19, 5, tzinfo=UTC)
WINDOW_END = datetime(2026, 7, 20, 5, tzinfo=UTC)


def write_scope(path: Path) -> Path:
    path.mkdir()
    (path / "split.json").write_text(
        json.dumps(
            {
                "cities": ["KXHIGHNY"],
                "discovery_days": ["2026-07-19"],
                "holdout_days": ["2026-07-20"],
                "boundary_event_day": "2026-07-20",
                "scope_start": SCOPE_START.isoformat(),
                "scope_end": SCOPE_END.isoformat(),
            }
        )
    )
    pq.write_table(
        pa.table(
            {
                "series": ["KXHIGHNY"],
                "event_date": [date(2026, 7, 19)],
                "window_start": [WINDOW_START],
                "window_end": [WINDOW_END],
                "in_scope": [True],
            }
        ),
        path / "event_days.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "start": [datetime(2026, 7, 19, 7, tzinfo=UTC)],
                "end": [datetime(2026, 7, 19, 9, tzinfo=UTC)],
            }
        ),
        path / "exclusions.parquet",
    )
    return path


def write_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute(TRADE_SCHEMA)
    conn.execute(TRADE_INDEX)
    conn.execute("CREATE TABLE markets (ticker VARCHAR NOT NULL, PRIMARY KEY (ticker))")
    conn.executemany(
        "INSERT INTO ws_trades VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            trade_row(1, TICKER, "kept", datetime(2026, 7, 19, 6, tzinfo=UTC), "yes"),
            trade_row(2, TICKER, "excluded", datetime(2026, 7, 19, 8, tzinfo=UTC), "no"),
            trade_row(3, TICKER, "kept", datetime(2026, 7, 19, 10, tzinfo=UTC), "no"),
            trade_row(4, "KXLOWTNY-26JUL19-B79.5", "other", WINDOW_START, "yes"),
        ],
    )
    conn.executemany(
        "INSERT INTO markets VALUES (?)",
        [(TICKER,), ("KXHIGHNY-26JUL25-B79.5",), ("KXLOWTNY-26JUL19-B79.5",)],
    )
    conn.commit()
    conn.close()
    return path


def args_for(tmp_path: Path, mode: str) -> argparse.Namespace:
    raw = tmp_path / "ws_raw"
    raw.mkdir(exist_ok=True)
    return build_parser().parse_args(
        [
            "--run-scope",
            str(write_scope(tmp_path / "scope")),
            "--db",
            str(write_db(tmp_path / "state.db")),
            "--raw-dir",
            str(raw),
            "--mode",
            mode,
        ]
    )


@pytest.fixture
def tape(tmp_path: Path) -> Path:
    raw = tmp_path / "ws_raw"
    raw.mkdir()
    write_tape(
        raw / "2026-07-19.jsonl.gz",
        [
            (SCOPE_START - timedelta(hours=1), trade_frame(trade_id="out-of-scope")),
            (SCOPE_START, ACK_FRAME),
            (SCOPE_START, DELTA_FRAME),
            (SCOPE_START, trade_frame(trade_id="first")),
            (SCOPE_START + timedelta(hours=1), trade_frame(taker_outcome_side=...)),
        ],
    )
    write_tape(raw / "2026-07-20.jsonl.gz", [(datetime(2026, 7, 20, 9, tzinfo=UTC), trade_frame())])
    write_tape(raw / "2026-07-21.jsonl.gz", [(datetime(2026, 7, 21, 7, tzinfo=UTC), trade_frame())])
    return raw


def test_tape_days_split_the_scope_at_utc_midnight() -> None:
    days = tape_days(
        Split(
            cities=("KXHIGHNY",),
            discovery_days=(date(2026, 7, 19),),
            holdout_days=(date(2026, 7, 20),),
            boundary_event_day=date(2026, 7, 20),
            scope_start=SCOPE_START,
            scope_end=SCOPE_END,
        )
    )

    assert days == [
        (date(2026, 7, 19), SCOPE_START, datetime(2026, 7, 20, tzinfo=UTC)),
        (date(2026, 7, 20), datetime(2026, 7, 20, tzinfo=UTC), datetime(2026, 7, 21, tzinfo=UTC)),
        (date(2026, 7, 21), datetime(2026, 7, 21, tzinfo=UTC), SCOPE_END),
    ]


def test_frame_mode_dumps_a_whole_trade_frame_and_its_keys(
    tmp_path: Path, tape: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(args_for(tmp_path, "frame")) == 0

    out = capsys.readouterr().out
    assert '"trade_id":"first"' in out
    assert "msg_keys ['count_fp', 'market_ticker'," in out
    assert "'taker_outcome_side', 'taker_side'," in out


def test_tape_mode_reports_every_day_and_the_pooled_total(
    tmp_path: Path, tape: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(args_for(tmp_path, "tape")) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("2026-07-19 frames=2")
    assert lines[1].startswith("2026-07-20 frames=1")
    assert lines[2].startswith("2026-07-21 frames=1")
    assert "pooled frames=4" in lines[3]
    assert "outcome={'present': 3, 'absent': 1}" in lines[3]
    assert "legacy={'present': 4}" in lines[3]
    assert len([line for line in lines if line.startswith("key_set ")]) == 2


def test_trades_mode_counts_in_scope_rows_and_proves_the_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(args_for(tmp_path, "trades")) == 0

    out = capsys.readouterr().out
    assert "tickers=1 exclusions=1" in out
    assert "USING INDEX ix_ws_trades_ticker_received_at" in out
    assert "pooled tickers=1 rows=2 dropped_excluded=1" in out
    assert "discovery tickers=1 rows=2" in out
    assert "holdout tickers=0 rows=0" in out
    assert "duplicated_values=1 surplus_rows=1" in out
