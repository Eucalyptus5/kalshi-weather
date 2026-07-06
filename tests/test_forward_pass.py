import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.replay.forward_pass import (
    JsonState,
    PassResult,
    SourceRow,
    _PartitionWriter,
    run_forward_pass,
)
from bot.replay.ladder import Ladder


UTC = timezone.utc
T0 = datetime(2026, 7, 19, 4, 59, 0, 155692, tzinfo=UTC)
DB_TS = "%Y-%m-%d %H:%M:%S.%f"
DEN = "KXHIGHDEN-26JUL19-B85"
CHI = "KXHIGHCHI-26JUL19-B75"
BOS = "KXHIGHBOS-26JUL19-B70"
KILL_CODE = 17
REPO_ROOT = Path(__file__).resolve().parents[1]
CHILD = "from tests.test_forward_pass import child_main; child_main()"
RESUME_KNOBS = {"read_batch_rows": 5, "row_group_rows": 3, "barrier_rows": 4}

SCHEMA_SQL = """
CREATE TABLE ws_book_events (
    id INTEGER NOT NULL, ticker VARCHAR(64) NOT NULL, received_at DATETIME NOT NULL,
    seq INTEGER NOT NULL, side VARCHAR(8) NOT NULL, price VARCHAR NOT NULL,
    size VARCHAR NOT NULL, is_snapshot BOOLEAN NOT NULL, created_at DATETIME NOT NULL,
    ts_ms INTEGER, PRIMARY KEY (id))
"""

TOUCH_SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("ticker", pa.string()),
        ("received_at", pa.string()),
        ("yes_bid", pa.string()),
        ("yes_bid_depth", pa.int64()),
        ("no_bid", pa.string()),
        ("no_bid_depth", pa.int64()),
    ]
)

WRITER_SCHEMA = pa.schema([("id", pa.int64()), ("ticker", pa.string()), ("ts_ms", pa.int64())])


def _ts(offset_s: int) -> str:
    return (T0 + timedelta(seconds=offset_s)).strftime(DB_TS)


def _row(
    row_id: int,
    ticker: str,
    offset_s: int,
    seq: int,
    side: str,
    price: str,
    size: str,
    is_snapshot: bool,
) -> tuple[int, str, str, int, str, str, str, int, str, int | None]:
    at = _ts(offset_s)
    ts_ms = None if is_snapshot else 1_753_000_000_000 + row_id
    return (row_id, ticker, at, seq, side, price, size, int(is_snapshot), at, ts_ms)


FIXTURE_ROWS = [
    _row(1, DEN, 0, 1, "yes", "0.4000", "10.00", True),
    _row(2, DEN, 0, 1, "yes", "0.3900", "5.00", True),
    _row(3, DEN, 0, 1, "no", "0.5500", "7.00", True),
    _row(4, CHI, 1, 2, "yes", "0.2000", "4.00", True),
    _row(5, CHI, 1, 2, "no", "0.7000", "6.00", True),
    _row(6, DEN, 2, 3, "yes", "0.4100", "2.00", False),
    _row(7, CHI, 3, 4, "no", "0.7100", "1.00", False),
    _row(8, DEN, 4, 5, "yes", "0.3900", "-5.00", False),
    _row(9, CHI, 5, 6, "yes", "0.2000", "-1.00", False),
    _row(10, DEN, 6, 7, "yes", "0.4200", "3.00", True),
    _row(11, DEN, 6, 7, "yes", "0.4100", "1.00", True),
    _row(12, DEN, 6, 7, "no", "0.5600", "8.00", True),
    _row(13, CHI, 7, 8, "no", "0.7000", "2.00", False),
    _row(14, DEN, 8, 9, "yes", "0.4200", "-1.00", False),
    _row(15, CHI, 9, 10, "yes", "0.3000", "5.00", True),
    _row(16, CHI, 9, 10, "yes", "0.2900", "2.00", True),
    _row(17, DEN, 10, 11, "no", "0.5600", "-8.00", False),
    _row(18, CHI, 11, 12, "yes", "0.3000", "1.00", False),
]

POISON_ROWS = [
    _row(1, DEN, 0, 1, "yes", "0.4000", "10.00", True),
    _row(2, CHI, 1, 2, "yes", "0.2000", "4.00", True),
    _row(3, BOS, 2, 3, "yes", "0.1000", "3.00", True),
    _row(4, DEN, 3, 4, "yes", "0.4000", "1.00", False),
    _row(5, DEN, 4, 5, "yes", "0.4000", "1.00", False),
    _row(6, DEN, 5, 6, "yes", "0.6000", "2.00", True),
    _row(7, DEN, 6, 7, "yes", "0.6000", "1.00", False),
]


def build_db(path: Path, rows: list[tuple[object, ...]]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(SCHEMA_SQL)
    conn.executemany("INSERT INTO ws_book_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


class TouchEmitter:
    name = "touch"
    schema = TOUCH_SCHEMA

    def __init__(self, die_at: int = 0) -> None:
        self.die_at = die_at

    def emit(self, row: SourceRow, ladder: Ladder) -> Iterator[tuple[str, dict[str, object]]]:
        if row.id == self.die_at:
            os._exit(KILL_CODE)
        book = ladder.row(row.received_at)
        yield (
            row.ticker.split("-")[0],
            {
                "id": row.id,
                "ticker": row.ticker,
                "received_at": row.received_at.strftime(DB_TS),
                "yes_bid": str(book.yes_bid),
                "yes_bid_depth": book.yes_bid_depth,
                "no_bid": str(book.no_bid),
                "no_bid_depth": book.no_bid_depth,
            },
        )


class ProbeEmitter:
    name = "probe"
    schema = TOUCH_SCHEMA

    def __init__(self, at_id: int, action: Callable[[], None]) -> None:
        self.at_id = at_id
        self.action = action

    def emit(self, row: SourceRow, ladder: Ladder) -> Iterable[tuple[str, dict[str, object]]]:
        if row.id == self.at_id:
            self.action()
        return ()


class CountAccumulator:
    name = "counts"

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.last_seq = 0

    def observe(self, row: SourceRow) -> None:
        self.counts[row.ticker] = self.counts.get(row.ticker, 0) + 1
        self.last_seq = row.seq

    def state(self) -> JsonState:
        return {"counts": self.counts, "last_seq": self.last_seq}

    def restore(self, state: JsonState) -> None:
        self.counts = dict(state["counts"])
        self.last_seq = state["last_seq"]


def child_main() -> None:
    db_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    die_at = int(sys.argv[3])
    run_forward_pass(
        db_path,
        out_dir,
        [TouchEmitter(die_at)],
        accumulators=[CountAccumulator()],
        **RESUME_KNOBS,
    )


def spawn(db_path: Path, out_dir: Path, die_at: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", CHILD, str(db_path), str(out_dir), str(die_at)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
    )


def artifact(out_dir: Path) -> dict[str, list[dict[str, object]]]:
    return {
        str(path.relative_to(out_dir)): pq.read_table(path).to_pylist()
        for path in sorted(out_dir.rglob("*.parquet"))
    }


def partitions(out_dir: Path) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = {}
    for name, rows in sorted(artifact(out_dir).items()):
        out.setdefault(name.rsplit("-b", 1)[0], []).extend(rows)
    return out


def row_group_rows(path: Path) -> list[int]:
    metadata = pq.ParquetFile(path).metadata
    return [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]


def checkpoint_of(out_dir: Path) -> dict[str, object]:
    return json.loads((out_dir / "checkpoint.json").read_text())


@dataclass(frozen=True)
class Reference:
    db_path: Path
    out_dir: Path
    artifact: dict[str, list[dict[str, object]]]
    checkpoint: dict[str, object]


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> Reference:
    root = tmp_path_factory.mktemp("reference")
    db_path = root / "state.db"
    build_db(db_path, FIXTURE_ROWS)
    out_dir = root / "out"
    run_forward_pass(
        db_path,
        out_dir,
        [TouchEmitter()],
        accumulators=[CountAccumulator()],
        **RESUME_KNOBS,
    )
    return Reference(
        db_path=db_path,
        out_dir=out_dir,
        artifact=artifact(out_dir),
        checkpoint=checkpoint_of(out_dir),
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "state.db"
    build_db(path, FIXTURE_ROWS)
    return path


def test_row_group_boundaries_do_not_follow_the_read_batch(tmp_path: Path, db_path: Path) -> None:
    small = tmp_path / "small"
    large = tmp_path / "large"
    run_forward_pass(
        db_path, small, [TouchEmitter()], read_batch_rows=1, row_group_rows=3, barrier_rows=10_000
    )
    run_forward_pass(
        db_path,
        large,
        [TouchEmitter()],
        read_batch_rows=10_000,
        row_group_rows=3,
        barrier_rows=10_000,
    )

    names = sorted(str(p.relative_to(small)) for p in small.rglob("*.parquet"))
    assert names == sorted(str(p.relative_to(large)) for p in large.rglob("*.parquet"))
    assert names == [
        "touch/KXHIGHCHI-b000001.parquet",
        "touch/KXHIGHDEN-b000001.parquet",
    ]
    for name in names:
        one, many = small / name, large / name
        assert pq.read_table(one).equals(pq.read_table(many))
        assert pq.read_table(one).to_pylist() == pq.read_table(many).to_pylist()
        assert row_group_rows(one) == row_group_rows(many)
    assert row_group_rows(small / "touch/KXHIGHDEN-b000001.parquet") == [3, 3, 3, 1]
    assert row_group_rows(small / "touch/KXHIGHCHI-b000001.parquet") == [3, 3, 2]
    assert [r["id"] for r in artifact(small)["touch/KXHIGHDEN-b000001.parquet"]] == [
        1,
        2,
        3,
        6,
        8,
        10,
        11,
        12,
        14,
        17,
    ]


@pytest.mark.parametrize("die_at", range(1, len(FIXTURE_ROWS) + 1))
def test_kill_at_every_id_resumes_to_the_uninterrupted_result(
    reference: Reference, tmp_path: Path, die_at: int
) -> None:
    out_dir = tmp_path / "out"
    dead = spawn(reference.db_path, out_dir, die_at)
    assert dead.returncode == KILL_CODE, dead.stderr

    resumed = spawn(reference.db_path, out_dir, 0)
    assert resumed.returncode == 0, resumed.stderr

    assert artifact(out_dir) == reference.artifact
    final = checkpoint_of(out_dir)
    assert final["ladders"] == reference.checkpoint["ladders"]
    assert final["accumulators"] == reference.checkpoint["accumulators"]
    assert final["id"] == reference.checkpoint["id"]
    assert final["barrier"] == reference.checkpoint["barrier"]


def test_reads_are_read_only_and_leave_the_source_untouched(
    tmp_path: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bot.replay import forward_pass

    opened: list[tuple[str, bool]] = []
    real_connect = sqlite3.connect

    def spy(target, *args, **kwargs):
        opened.append((target, kwargs.get("uri", False)))
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr(forward_pass.sqlite3, "connect", spy)
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    result = run_forward_pass(db_path, tmp_path / "out", [TouchEmitter()])

    assert result.rows == len(FIXTURE_ROWS)
    assert len(opened) == 1
    target, uri = opened[0]
    assert target.startswith("file:")
    assert target.endswith("?mode=ro")
    assert uri is True
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before
    assert not (tmp_path / "state.db-wal").exists()
    assert not (tmp_path / "state.db-journal").exists()


def test_each_read_is_its_own_statement_carrying_the_last_id(
    tmp_path: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bot.replay import forward_pass

    statements: list[str] = []
    real_connect = sqlite3.connect

    def spy(target, *args, **kwargs):
        conn = real_connect(target, *args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(forward_pass.sqlite3, "connect", spy)
    run_forward_pass(db_path, tmp_path / "out", [TouchEmitter()], read_batch_rows=1)

    selects = [s for s in statements if "ws_book_events" in s]
    assert len(selects) == len(FIXTURE_ROWS) + 1
    assert all("id > " in s for s in selects)


def test_source_stays_writable_while_the_pass_runs(tmp_path: Path, db_path: Path) -> None:
    blocked: list[str] = []

    def insert() -> None:
        writer = sqlite3.connect(db_path, timeout=1)
        try:
            writer.execute(
                "INSERT INTO ws_book_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _row(99, BOS, 20, 30, "yes", "0.5000", "1.00", True),
            )
            writer.commit()
        except sqlite3.OperationalError as exc:
            blocked.append(str(exc))
        writer.close()

    result = run_forward_pass(
        db_path,
        tmp_path / "out",
        [TouchEmitter(), ProbeEmitter(3, insert)],
        read_batch_rows=2,
    )
    assert blocked == []
    assert result.rows == len(FIXTURE_ROWS) + 1
    assert result.last_id == 99
    assert BOS in result.ladders


def test_every_ticker_stays_resident_by_default(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    build_db(path, POISON_ROWS)
    result = run_forward_pass(path, tmp_path / "out", [TouchEmitter()], barrier_rows=10_000)

    assert set(result.ladders) == {DEN, CHI, BOS}
    assert result.poisoned_rows == {}
    rows = artifact(tmp_path / "out")["touch/KXHIGHDEN-b000001.parquet"]
    assert [r["id"] for r in rows] == [1, 4, 5, 6, 7]


def test_an_evicted_ticker_is_poisoned_until_its_next_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    build_db(path, POISON_ROWS)
    result = run_forward_pass(
        path,
        tmp_path / "out",
        [TouchEmitter()],
        barrier_rows=10_000,
        max_resident_ladders=2,
    )

    assert result.poisoned_rows == {DEN: 2}
    rows = artifact(tmp_path / "out")["touch/KXHIGHDEN-b000001.parquet"]
    assert [r["id"] for r in rows] == [1, 6, 7]
    assert rows[-1]["yes_bid"] == "0.6000"
    assert rows[-1]["yes_bid_depth"] == 3
    assert set(result.ladders) == {BOS, DEN}
    assert result.ladders[DEN].levels["yes"] == {Decimal("0.6000"): Decimal("3.00")}
    final = checkpoint_of(tmp_path / "out")
    assert final["poisoned"] == [CHI]
    assert final["poisoned_rows"] == {DEN: 2}


def test_poisoned_state_survives_a_resume(tmp_path: Path) -> None:
    build_db(tmp_path / "early.db", POISON_ROWS[:4])
    build_db(tmp_path / "full.db", POISON_ROWS)
    whole = tmp_path / "whole"
    run_forward_pass(
        tmp_path / "full.db", whole, [TouchEmitter()], barrier_rows=4, max_resident_ladders=2
    )

    out_dir = tmp_path / "out"
    run_forward_pass(
        tmp_path / "early.db", out_dir, [TouchEmitter()], barrier_rows=4, max_resident_ladders=2
    )
    assert checkpoint_of(out_dir)["poisoned_rows"] == {DEN: 1}
    result = run_forward_pass(
        tmp_path / "full.db", out_dir, [TouchEmitter()], barrier_rows=4, max_resident_ladders=2
    )

    assert result.rows == 3
    assert result.poisoned_rows == {DEN: 2}
    assert artifact(out_dir) == artifact(whole)
    assert checkpoint_of(out_dir)["ladders"] == checkpoint_of(whole)["ladders"]


def test_checkpoint_carries_the_barrier_row_ladders_and_accumulators(
    tmp_path: Path, db_path: Path
) -> None:
    out_dir = tmp_path / "out"
    run_forward_pass(
        db_path,
        out_dir,
        [TouchEmitter()],
        accumulators=[CountAccumulator()],
        read_batch_rows=5,
        row_group_rows=3,
        barrier_rows=18,
    )
    at_end = checkpoint_of(out_dir)
    assert at_end["id"] == 18
    assert at_end["barrier"] == 1
    assert at_end["ladders"][CHI]["batch_key"] == [_ts(9), 10]
    assert at_end["ladders"][DEN]["levels"] == {
        "yes": {"0.4200": "2.00", "0.4100": "1.00"},
        "no": {},
    }
    assert at_end["accumulators"]["counts"] == {
        "counts": {DEN: 10, CHI: 8},
        "last_seq": 12,
    }
    assert at_end["poisoned"] == []
    assert at_end["poisoned_rows"] == {}


def test_checkpoint_at_a_barrier_inside_a_snapshot_batch(tmp_path: Path) -> None:
    out_dir = tmp_path / "early"
    build_db(tmp_path / "early.db", FIXTURE_ROWS[:4])
    run_forward_pass(
        tmp_path / "early.db",
        out_dir,
        [TouchEmitter()],
        accumulators=[CountAccumulator()],
        row_group_rows=3,
        barrier_rows=4,
    )
    early = checkpoint_of(out_dir)
    assert early["id"] == 4
    assert early["barrier"] == 1
    assert early["ladders"][CHI]["batch_key"] == [_ts(1), 2]
    assert early["ladders"][CHI]["levels"] == {"yes": {"0.2000": "4.00"}, "no": {}}
    assert early["accumulators"]["counts"]["counts"] == {DEN: 3, CHI: 1}


def test_checkpoint_replaces_a_temp_file_after_every_writer_is_closed(
    tmp_path: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bot.replay import forward_pass

    out_dir = tmp_path / "out"
    fsyncs: list[int] = []
    replaces: list[tuple[str, str, int, list[str]]] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def fsync_spy(fd: int) -> None:
        fsyncs.append(fd)
        real_fsync(fd)

    def replace_spy(src, dst):
        readable = []
        for path in sorted(out_dir.rglob("*.parquet")):
            try:
                pq.ParquetFile(path).metadata.num_rows
            except pa.ArrowInvalid:
                continue
            readable.append(str(path.relative_to(out_dir)))
        replaces.append((str(src), str(dst), len(fsyncs), readable))
        real_replace(src, dst)

    monkeypatch.setattr(forward_pass.os, "fsync", fsync_spy)
    monkeypatch.setattr(forward_pass.os, "replace", replace_spy)
    run_forward_pass(
        db_path,
        out_dir,
        [TouchEmitter()],
        read_batch_rows=5,
        row_group_rows=3,
        barrier_rows=4,
    )

    assert len(replaces) == 5
    src, dst, fsync_count, readable = replaces[0]
    assert src == str(out_dir / "checkpoint.json.tmp")
    assert dst == str(out_dir / "checkpoint.json")
    assert readable == [
        "touch/KXHIGHCHI-b000001.parquet",
        "touch/KXHIGHDEN-b000001.parquet",
    ]
    assert fsync_count == 3
    assert [count for _, _, count, _ in replaces] == [3, 6, 9, 12, 15]


def test_each_output_path_carries_its_barrier_and_is_written_once(
    tmp_path: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bot.replay import forward_pass

    opened: list[str] = []
    real_open = Path.open

    def open_spy(self, mode="r", *args, **kwargs):
        if self.suffix == ".parquet":
            opened.append(str(self))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(forward_pass.Path, "open", open_spy)
    out_dir = tmp_path / "out"
    run_forward_pass(
        db_path,
        out_dir,
        [TouchEmitter()],
        read_batch_rows=5,
        row_group_rows=3,
        barrier_rows=4,
    )

    assert len(opened) == len(set(opened))
    names = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*.parquet"))
    assert names == [
        "touch/KXHIGHCHI-b000001.parquet",
        "touch/KXHIGHCHI-b000002.parquet",
        "touch/KXHIGHCHI-b000003.parquet",
        "touch/KXHIGHCHI-b000004.parquet",
        "touch/KXHIGHCHI-b000005.parquet",
        "touch/KXHIGHDEN-b000001.parquet",
        "touch/KXHIGHDEN-b000002.parquet",
        "touch/KXHIGHDEN-b000003.parquet",
        "touch/KXHIGHDEN-b000004.parquet",
        "touch/KXHIGHDEN-b000005.parquet",
    ]
    assert sorted(opened) == [str(out_dir / name) for name in names]


def test_resume_deletes_files_stamped_above_the_checkpoint_barrier(
    tmp_path: Path, db_path: Path
) -> None:
    out_dir = tmp_path / "out"
    build_db(tmp_path / "early.db", FIXTURE_ROWS[:4])
    run_forward_pass(
        tmp_path / "early.db",
        out_dir,
        [TouchEmitter()],
        accumulators=[CountAccumulator()],
        row_group_rows=3,
        barrier_rows=4,
    )
    survivor = out_dir / "touch" / "KXHIGHDEN-b000001.parquet"
    stale = out_dir / "touch" / "KXHIGHCHI-b000009.parquet"
    stale.write_bytes(b"PAR1 truncated")
    assert survivor.exists()

    result = run_forward_pass(
        db_path,
        out_dir,
        [TouchEmitter()],
        accumulators=[CountAccumulator()],
        row_group_rows=3,
        barrier_rows=4,
    )

    assert not stale.exists()
    assert survivor.exists()
    assert result.rows == len(FIXTURE_ROWS) - 4
    assert result.last_id == 18


def test_progress_is_logged_and_nothing_reaches_stdout(
    tmp_path: Path,
    db_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="bot.replay.forward_pass")
    run_forward_pass(db_path, tmp_path / "out", [TouchEmitter()], barrier_rows=4)

    messages = [r.getMessage() for r in caplog.records if r.name == "bot.replay.forward_pass"]
    assert any("rows=4" in m and "elapsed_s=" in m for m in messages)
    assert any("rows=18" in m and "elapsed_s=" in m for m in messages)
    assert capsys.readouterr().out == ""


def test_the_id_ceiling_consumes_exactly_the_rows_at_or_below_it(
    reference: Reference, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    result = run_forward_pass(
        reference.db_path,
        out_dir,
        [TouchEmitter()],
        accumulators=[CountAccumulator()],
        max_id=10,
        **RESUME_KNOBS,
    )
    files = artifact(out_dir)

    assert result.rows == 10
    assert result.last_id == 10
    assert sorted(files) == [
        name
        for name, rows in sorted(reference.artifact.items())
        if any(r["id"] <= 10 for r in rows)
    ]
    for name, rows in files.items():
        kept = [r for r in reference.artifact[name] if r["id"] <= 10]
        assert rows == kept
        assert pq.read_table(out_dir / name).equals(pa.Table.from_pylist(kept, schema=TOUCH_SCHEMA))


def test_a_ceiling_above_the_last_id_is_the_same_as_no_ceiling(
    reference: Reference, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    result = run_forward_pass(
        reference.db_path,
        out_dir,
        [TouchEmitter()],
        accumulators=[CountAccumulator()],
        max_id=10_000,
        **RESUME_KNOBS,
    )

    assert result.rows == len(FIXTURE_ROWS)
    assert result.last_id == 18
    assert result.barriers == 5
    assert artifact(out_dir) == reference.artifact
    assert checkpoint_of(out_dir) == reference.checkpoint


def test_a_pass_bounded_mid_barrier_resumes_to_the_uninterrupted_result(
    reference: Reference, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    bounded = run_forward_pass(
        reference.db_path,
        out_dir,
        [TouchEmitter()],
        accumulators=[CountAccumulator()],
        max_id=10,
        **RESUME_KNOBS,
    )
    assert bounded.barriers == 3
    resumed = run_forward_pass(
        reference.db_path,
        out_dir,
        [TouchEmitter()],
        accumulators=[CountAccumulator()],
        **RESUME_KNOBS,
    )

    assert resumed.rows == len(FIXTURE_ROWS) - 10
    assert partitions(out_dir) == partitions(reference.out_dir)
    final = checkpoint_of(out_dir)
    assert final["ladders"] == reference.checkpoint["ladders"]
    assert final["accumulators"] == reference.checkpoint["accumulators"]
    assert final["id"] == reference.checkpoint["id"]


def test_the_ceiling_bounds_the_end_and_leaves_the_checkpoint_to_bound_the_start(
    reference: Reference, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    first = run_forward_pass(
        reference.db_path,
        out_dir,
        [TouchEmitter()],
        accumulators=[CountAccumulator()],
        max_id=8,
        **RESUME_KNOBS,
    )
    second = run_forward_pass(
        reference.db_path,
        out_dir,
        [TouchEmitter()],
        accumulators=[CountAccumulator()],
        max_id=14,
        **RESUME_KNOBS,
    )

    assert (first.rows, first.last_id) == (8, 8)
    assert (second.rows, second.last_id) == (6, 14)
    assert [r["id"] for r in partitions(out_dir)["touch/KXHIGHDEN"]] == [
        1,
        2,
        3,
        6,
        8,
        10,
        11,
        12,
        14,
    ]


def test_a_null_field_shares_its_row_group_with_populated_rows(tmp_path: Path) -> None:
    path = tmp_path / "nulls.parquet"
    rows: list[dict[str, object]] = [
        {"id": 1, "ticker": DEN, "ts_ms": 1_753_000_000_001},
        {"id": 2, "ticker": DEN, "ts_ms": None},
        {"id": 3, "ticker": CHI, "ts_ms": None},
        {"id": 4, "ticker": CHI, "ts_ms": 1_753_000_000_004},
    ]
    writer = _PartitionWriter(path, WRITER_SCHEMA, 4)
    for row in rows:
        writer.add(row)
    writer.close()

    table = pq.read_table(path)
    assert table.schema.equals(WRITER_SCHEMA)
    assert table.to_pylist() == rows
    assert table.column("ts_ms").null_count == 2
    assert row_group_rows(path) == [4]


def test_row_groups_close_at_exactly_the_row_group_size(tmp_path: Path) -> None:
    path = tmp_path / "groups.parquet"
    size = 5
    writer = _PartitionWriter(path, WRITER_SCHEMA, size)
    for row_id in range(2 * size + 1):
        writer.add({"id": row_id, "ticker": DEN, "ts_ms": None})
    writer.close()

    assert row_group_rows(path) == [size, size, 1]
    assert [r["id"] for r in pq.read_table(path).to_pylist()] == list(range(2 * size + 1))


def test_close_writes_the_rows_left_in_a_partial_buffer(tmp_path: Path) -> None:
    path = tmp_path / "partial.parquet"
    writer = _PartitionWriter(path, WRITER_SCHEMA, 100)
    for row_id in range(3):
        writer.add({"id": row_id, "ticker": CHI, "ts_ms": row_id})
    writer.close()

    assert row_group_rows(path) == [3]
    assert pq.read_table(path).to_pylist() == [
        {"id": 0, "ticker": CHI, "ts_ms": 0},
        {"id": 1, "ticker": CHI, "ts_ms": 1},
        {"id": 2, "ticker": CHI, "ts_ms": 2},
    ]


def test_a_flush_leaves_nothing_behind_for_the_next_one(tmp_path: Path) -> None:
    path = tmp_path / "cleared.parquet"
    size = 4
    writer = _PartitionWriter(path, WRITER_SCHEMA, size)
    for row_id in range(2 * size):
        writer.add({"id": row_id, "ticker": DEN, "ts_ms": None})
    writer.close()

    assert pq.ParquetFile(path).metadata.num_rows == 2 * size
    assert row_group_rows(path) == [size, size]
    assert [r["id"] for r in pq.read_table(path).to_pylist()] == list(range(2 * size))


def test_pass_result_reports_what_it_consumed(tmp_path: Path, db_path: Path) -> None:
    result = run_forward_pass(
        db_path, tmp_path / "out", [TouchEmitter()], read_batch_rows=5, barrier_rows=4
    )
    assert isinstance(result, PassResult)
    assert result.rows == 18
    assert result.last_id == 18
    assert result.barriers == 5
    assert result.elapsed_s >= 0
    assert set(result.ladders) == {DEN, CHI}
    assert result.ladders[DEN].levels["no"] == {}
