from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import ws_watchdog
from scripts.ws_watchdog import (
    build_parser,
    connect_ro,
    evaluate,
    in_restart_window,
    main,
    read_db,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXED_NOW = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
TOPIC = "wd-topic-c9f3e"

_SCHEMA = """
CREATE TABLE ws_heartbeats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    beat_at DATETIME NOT NULL,
    book_events INTEGER NOT NULL,
    trades INTEGER NOT NULL,
    gaps INTEGER NOT NULL,
    subscribed INTEGER NOT NULL,
    raw_bytes INTEGER NOT NULL,
    created_at DATETIME NOT NULL
);
CREATE TABLE ws_gaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR(64) NOT NULL,
    detected_at DATETIME NOT NULL,
    last_seq INTEGER NOT NULL,
    reason VARCHAR(64) NOT NULL,
    created_at DATETIME NOT NULL
);
"""


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "state.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()
    return path


def insert_heartbeat(db: Path, beat_at: str) -> None:
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO ws_heartbeats "
        "(beat_at, book_events, trades, gaps, subscribed, raw_bytes, created_at) "
        "VALUES (?, 120, 4, 0, 12, 48000, ?)",
        (beat_at, beat_at),
    )
    conn.commit()
    conn.close()


def insert_gaps(db: Path, n: int) -> None:
    conn = sqlite3.connect(str(db))
    for _ in range(n):
        conn.execute(
            "INSERT INTO ws_gaps (ticker, detected_at, last_seq, reason, created_at) "
            "VALUES ('KXHIGHDEN-26JUL19-B85', '2026-07-19 08:04:11.000000', 42, 'seq_gap', "
            "'2026-07-19 08:04:11.000000')"
        )
    conn.commit()
    conn.close()


def watchdog_argv(tmp_path: Path, db: Path) -> list[str]:
    return [
        "--db",
        str(db),
        "--state",
        str(tmp_path / "wd_state.json"),
        "--log",
        str(tmp_path / "wd.log"),
    ]


def mock_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    service_state: str = "active",
    free_fraction: float = 0.5,
) -> list[urllib.request.Request]:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert cmd == ["systemctl", "is-active", "kalshi-ws.service"]
        code = 0 if service_state == "active" else 3
        return subprocess.CompletedProcess(cmd, code, stdout=service_state + "\n", stderr="")

    monkeypatch.setattr(ws_watchdog.subprocess, "run", fake_run)

    blocks = 1000
    fake_stats = SimpleNamespace(f_bavail=int(free_fraction * blocks), f_blocks=blocks)
    monkeypatch.setattr(ws_watchdog.os, "statvfs", lambda path: fake_stats)
    monkeypatch.setattr(ws_watchdog, "utc_now", lambda: FIXED_NOW)

    requests: list[urllib.request.Request] = []

    class Resp:
        def __enter__(self) -> Resp:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> Resp:
        requests.append(request)
        return Resp()

    monkeypatch.setattr(ws_watchdog.urllib.request, "urlopen", fake_urlopen)
    return requests


GOLDEN = [
    ("active", 60.0, 0, 0.50, False, True, []),
    ("failed", 60.0, 0, 0.50, False, True, ["service_not_active state=failed"]),
    ("inactive", 60.0, 0, 0.50, False, True, ["service_not_active state=inactive"]),
    ("activating", 60.0, 0, 0.50, False, True, []),
    ("activating", 60.0, 0, 0.50, True, True, []),
    ("failed", 60.0, 0, 0.50, True, True, ["service_not_active state=failed"]),
    ("active", 300.0, 0, 0.50, False, True, []),
    ("active", 301.0, 0, 0.50, False, True, ["heartbeat_stale age_seconds=301"]),
    ("active", None, 0, 0.50, False, True, ["heartbeat_stale age_seconds=none"]),
    ("active", None, 0, 0.50, True, True, []),
    ("active", 512.0, 0, 0.50, True, True, []),
    ("active", 512.0, 0, 0.50, False, True, ["heartbeat_stale age_seconds=512"]),
    ("active", 60.0, 2, 0.50, False, True, []),
    ("active", 60.0, 3, 0.50, False, True, ["ws_gaps_delta delta=3"]),
    ("active", 60.0, None, 0.50, False, True, []),
    ("active", 60.0, -5, 0.50, False, True, []),
    ("active", 60.0, 0, 0.20, False, True, []),
    ("active", 60.0, 0, 0.19, False, True, ["disk_low free_fraction=0.19"]),
    ("active", 60.0, 0, 0.14, False, True, ["disk_low free_fraction=0.14"]),
    ("active", None, None, 0.50, False, False, ["db_unreadable"]),
    (
        "failed",
        512.0,
        5,
        0.14,
        False,
        True,
        [
            "service_not_active state=failed",
            "heartbeat_stale age_seconds=512",
            "ws_gaps_delta delta=5",
            "disk_low free_fraction=0.14",
        ],
    ),
    (
        "failed",
        None,
        None,
        0.14,
        False,
        False,
        [
            "service_not_active state=failed",
            "db_unreadable",
            "disk_low free_fraction=0.14",
        ],
    ),
]


@pytest.mark.parametrize(
    ("state", "age", "delta", "disk", "in_window", "db_ok", "expected"), GOLDEN
)
def test_evaluate_golden(
    state: str,
    age: float | None,
    delta: int | None,
    disk: float,
    in_window: bool,
    db_ok: bool,
    expected: list[str],
) -> None:
    assert evaluate(state, age, delta, disk, in_window, db_ok) == expected


@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [(7, 54, False), (7, 55, True), (8, 39, True), (8, 40, False)],
)
def test_restart_window_edges(hour: int, minute: int, expected: bool) -> None:
    now = datetime(2026, 7, 19, hour, minute, 30, tzinfo=timezone.utc)
    assert in_restart_window(now) is expected


def test_read_db_parses_production_beat_format(db_path: Path) -> None:
    insert_heartbeat(db_path, "2026-07-19 22:48:30.721057")
    insert_heartbeat(db_path, "2026-07-19 22:49:30.821057")
    latest, gap_count = read_db(db_path)
    assert latest == datetime(2026, 7, 19, 22, 49, 30, 821057, tzinfo=timezone.utc)
    assert gap_count == 0


def test_read_db_empty_tables(db_path: Path) -> None:
    latest, gap_count = read_db(db_path)
    assert latest is None
    assert gap_count == 0


def test_read_db_counts_gaps(db_path: Path) -> None:
    insert_gaps(db_path, 3)
    _, gap_count = read_db(db_path)
    assert gap_count == 3


def test_connect_ro_rejects_writes(db_path: Path) -> None:
    conn = connect_ro(db_path)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute(
            "INSERT INTO ws_gaps (ticker, detected_at, last_seq, reason, created_at) "
            "VALUES ('T', '2026-07-19 00:00:00.000000', 1, 'seq_gap', "
            "'2026-07-19 00:00:00.000000')"
        )
    conn.close()


def test_default_args() -> None:
    args = build_parser().parse_args([])
    assert args.db == Path("data/state.db")
    assert args.unit == "kalshi-ws.service"
    assert args.state.name == "ws_watchdog_state.json"
    assert args.log.name == "ws_watchdog.log"


def test_help_smoke() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "ws_watchdog.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--db" in result.stdout
    assert "--unit" in result.stdout


def test_run_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db_path: Path) -> None:
    requests = mock_boundaries(monkeypatch)
    monkeypatch.setenv("KW_WATCHDOG_NTFY_TOPIC", TOPIC)
    beat = (FIXED_NOW - timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S.%f")
    insert_heartbeat(db_path, beat)

    rc = main(watchdog_argv(tmp_path, db_path))

    assert rc == 0
    assert requests == []
    lines = (tmp_path / "wd.log").read_text().splitlines()
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("2026-07-19T12:00:00Z ")
    assert "verdict=ok" in line
    assert "service=active" in line
    assert "heartbeat_age_s=60 " in line
    assert "gap_count=0" in line
    assert "gap_delta=none" in line
    assert "disk_free_fraction=0.500" in line
    assert "ntfy=none" in line
    assert TOPIC not in line
    assert json.loads((tmp_path / "wd_state.json").read_text()) == {"gap_count": 0}


def test_run_alarm_posts_to_ntfy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db_path: Path
) -> None:
    requests = mock_boundaries(monkeypatch, service_state="failed")
    monkeypatch.setenv("KW_WATCHDOG_NTFY_TOPIC", TOPIC)
    beat = (FIXED_NOW - timedelta(seconds=600)).strftime("%Y-%m-%d %H:%M:%S.%f")
    insert_heartbeat(db_path, beat)

    rc = main(watchdog_argv(tmp_path, db_path))

    assert rc == 0
    assert len(requests) == 1
    request = requests[0]
    assert request.full_url == f"https://ntfy.sh/{TOPIC}"
    assert request.data == b"service_not_active state=failed\nheartbeat_stale age_seconds=600"
    lines = (tmp_path / "wd.log").read_text().splitlines()
    assert len(lines) == 1
    assert "verdict=alarm" in lines[0]
    assert "ntfy=sent" in lines[0]
    assert "service_not_active state=failed" in lines[0]
    assert "heartbeat_stale age_seconds=600" in lines[0]
    assert TOPIC not in lines[0]


def test_run_alarm_without_topic_skips_post(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db_path: Path
) -> None:
    requests = mock_boundaries(monkeypatch, service_state="failed")
    monkeypatch.delenv("KW_WATCHDOG_NTFY_TOPIC", raising=False)
    beat = (FIXED_NOW - timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S.%f")
    insert_heartbeat(db_path, beat)

    rc = main(watchdog_argv(tmp_path, db_path))

    assert rc == 0
    assert requests == []
    lines = (tmp_path / "wd.log").read_text().splitlines()
    assert len(lines) == 1
    assert "verdict=alarm" in lines[0]
    assert "ntfy=skipped" in lines[0]
    assert "service_not_active state=failed" in lines[0]


def test_run_post_failure_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db_path: Path
) -> None:
    mock_boundaries(monkeypatch, service_state="failed")

    def raising_urlopen(request: urllib.request.Request, timeout: float | None = None) -> None:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(ws_watchdog.urllib.request, "urlopen", raising_urlopen)
    monkeypatch.setenv("KW_WATCHDOG_NTFY_TOPIC", TOPIC)
    beat = (FIXED_NOW - timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S.%f")
    insert_heartbeat(db_path, beat)

    rc = main(watchdog_argv(tmp_path, db_path))

    assert rc == 1
    lines = (tmp_path / "wd.log").read_text().splitlines()
    assert len(lines) == 1
    assert "verdict=alarm" in lines[0]
    assert "ntfy=failed" in lines[0]
    assert TOPIC not in lines[0]


def test_run_missing_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    requests = mock_boundaries(monkeypatch)
    monkeypatch.delenv("KW_WATCHDOG_NTFY_TOPIC", raising=False)

    rc = main(watchdog_argv(tmp_path, tmp_path / "missing.db"))

    assert rc == 0
    assert requests == []
    lines = (tmp_path / "wd.log").read_text().splitlines()
    assert len(lines) == 1
    assert "verdict=alarm" in lines[0]
    assert "db_unreadable" in lines[0]
    assert "heartbeat_age_s=none" in lines[0]
    assert "gap_count=none" in lines[0]
    assert json.loads((tmp_path / "wd_state.json").read_text()) == {"gap_count": None}


def test_state_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db_path: Path) -> None:
    mock_boundaries(monkeypatch)
    monkeypatch.delenv("KW_WATCHDOG_NTFY_TOPIC", raising=False)
    beat = (FIXED_NOW - timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S.%f")
    insert_heartbeat(db_path, beat)
    insert_gaps(db_path, 1)
    argv = watchdog_argv(tmp_path, db_path)

    rc = main(argv)
    assert rc == 0
    assert json.loads((tmp_path / "wd_state.json").read_text()) == {"gap_count": 1}
    lines = (tmp_path / "wd.log").read_text().splitlines()
    assert len(lines) == 1
    assert "verdict=ok" in lines[0]
    assert "gap_delta=none" in lines[0]

    insert_gaps(db_path, 5)
    rc = main(argv)
    assert rc == 0
    assert json.loads((tmp_path / "wd_state.json").read_text()) == {"gap_count": 6}
    lines = (tmp_path / "wd.log").read_text().splitlines()
    assert len(lines) == 2
    assert "verdict=alarm" in lines[1]
    assert "gap_delta=5" in lines[1]
    assert "ws_gaps_delta delta=5" in lines[1]
    assert "ntfy=skipped" in lines[1]
