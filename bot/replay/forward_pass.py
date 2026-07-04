import json
import logging
import os
import sqlite3
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from bot.replay.ladder import BookEvent, Ladder


logger = logging.getLogger(__name__)

READ_BATCH_ROWS = 50_000
ROW_GROUP_ROWS = 100_000
BARRIER_ROWS = 5_000_000

CHECKPOINT_NAME = "checkpoint.json"

_DB_TS = "%Y-%m-%d %H:%M:%S.%f"
_COLUMNS = "id, ticker, received_at, seq, side, price, size, is_snapshot, ts_ms"
_SELECT = f"SELECT {_COLUMNS} FROM ws_book_events WHERE id > ? ORDER BY id LIMIT ?"
_SELECT_BOUNDED = (
    f"SELECT {_COLUMNS} FROM ws_book_events WHERE id > ? AND id <= ? ORDER BY id LIMIT ?"
)

type JsonState = dict[str, JsonState] | list[JsonState] | str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class SourceRow:
    id: int
    ticker: str
    received_at: datetime
    seq: int
    side: str
    price: str
    size: str
    is_snapshot: bool
    ts_ms: int | None

    def book_event(self) -> BookEvent:
        return BookEvent(
            received_at=self.received_at,
            seq=self.seq,
            side=self.side,
            price=self.price,
            size=self.size,
            is_snapshot=self.is_snapshot,
        )


class Emitter(Protocol):
    name: str
    schema: pa.Schema

    def emit(self, row: SourceRow, ladder: Ladder) -> Iterable[tuple[str, dict[str, object]]]: ...


class Accumulator(Protocol):
    name: str

    def observe(self, row: SourceRow) -> None: ...

    def state(self) -> JsonState: ...

    def restore(self, state: JsonState) -> None: ...


@dataclass(frozen=True, slots=True)
class PassResult:
    rows: int
    last_id: int
    barriers: int
    elapsed_s: float
    ladders: dict[str, Ladder]
    poisoned_rows: dict[str, int]


class _PartitionWriter:
    def __init__(self, path: Path, schema: pa.Schema, row_group_rows: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("wb")
        self._writer = pq.ParquetWriter(self._handle, schema)
        self._schema = schema
        self._row_group_rows = row_group_rows
        self._buffer: list[dict[str, object]] = []

    def add(self, row: dict[str, object]) -> None:
        self._buffer.append(row)
        if len(self._buffer) == self._row_group_rows:
            self._flush()

    def close(self) -> None:
        if self._buffer:
            self._flush()
        self._writer.close()
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()

    def _flush(self) -> None:
        table = pa.Table.from_pylist(self._buffer, schema=self._schema)
        self._writer.write_table(table, row_group_size=self._row_group_rows)
        self._buffer.clear()


class _Pass:
    def __init__(
        self,
        out_dir: Path,
        emitters: Sequence[Emitter],
        accumulators: Sequence[Accumulator],
        row_group_rows: int,
        barrier_rows: int,
        max_resident_ladders: int | None,
    ) -> None:
        self.out_dir = out_dir
        self.emitters = emitters
        self.accumulators = accumulators
        self.row_group_rows = row_group_rows
        self.barrier_rows = barrier_rows
        self.ceiling = max_resident_ladders
        self.ladders: dict[str, Ladder] = {}
        self.poisoned: set[str] = set()
        self.poisoned_rows: dict[str, int] = {}
        self.last_id = 0
        self.barrier = 0
        self.rows = 0
        self.since_barrier = 0
        self.started = time.monotonic()
        self._writers: dict[tuple[str, str], _PartitionWriter] = {}

    def restore(self) -> None:
        path = self.out_dir / CHECKPOINT_NAME
        if path.exists():
            payload = json.loads(path.read_text())
            self.last_id = payload["id"]
            self.barrier = payload["barrier"]
            self.ladders = {
                ticker: _ladder_from_state(ticker, state)
                for ticker, state in payload["ladders"].items()
            }
            self.poisoned = set(payload["poisoned"])
            self.poisoned_rows = dict(payload["poisoned_rows"])
            for accumulator in self.accumulators:
                accumulator.restore(payload["accumulators"][accumulator.name])
            logger.info(
                "forward_pass resume barrier=%d id=%d ladders=%d poisoned=%d",
                self.barrier,
                self.last_id,
                len(self.ladders),
                len(self.poisoned),
            )
        for stale in sorted(self.out_dir.rglob("*.parquet")):
            if int(stale.stem.rsplit("-b", 1)[1]) > self.barrier:
                stale.unlink()
                logger.info("forward_pass dropped path=%s", stale)

    def consume(self, row: SourceRow) -> None:
        self.last_id = row.id
        self.rows += 1
        self.since_barrier += 1
        for accumulator in self.accumulators:
            accumulator.observe(row)
        ladder = self._ladder(row)
        if ladder is not None:
            for emitter in self.emitters:
                for partition, out in emitter.emit(row, ladder):
                    self._writer(emitter, partition).add(out)
        if self.since_barrier == self.barrier_rows:
            self.commit()

    def commit(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()
        self.barrier += 1
        self.since_barrier = 0
        self._checkpoint()
        logger.info(
            "forward_pass barrier=%d id=%d rows=%d ladders=%d levels=%d elapsed_s=%.3f",
            self.barrier,
            self.last_id,
            self.rows,
            len(self.ladders),
            sum(len(side) for ladder in self.ladders.values() for side in ladder.levels.values()),
            time.monotonic() - self.started,
        )

    def _ladder(self, row: SourceRow) -> Ladder | None:
        ticker = row.ticker
        if ticker in self.poisoned:
            if not row.is_snapshot:
                self.poisoned_rows[ticker] = self.poisoned_rows.get(ticker, 0) + 1
                return None
            self.poisoned.remove(ticker)
        ladder = self.ladders.get(ticker)
        if ladder is None:
            if self.ceiling is not None and len(self.ladders) >= self.ceiling:
                self._evict()
            ladder = Ladder(ticker)
            self.ladders[ticker] = ladder
        elif self.ceiling is not None:
            self.ladders[ticker] = self.ladders.pop(ticker)
        ladder.apply(row.book_event())
        return ladder

    def _evict(self) -> None:
        victim = next(iter(self.ladders))
        del self.ladders[victim]
        self.poisoned.add(victim)
        logger.warning(
            "forward_pass evicted ticker=%s id=%d resident=%d",
            victim,
            self.last_id,
            len(self.ladders),
        )

    def _writer(self, emitter: Emitter, partition: str) -> _PartitionWriter:
        key = (emitter.name, partition)
        writer = self._writers.get(key)
        if writer is None:
            path = self.out_dir / emitter.name / f"{partition}-b{self.barrier + 1:06d}.parquet"
            writer = _PartitionWriter(path, emitter.schema, self.row_group_rows)
            self._writers[key] = writer
        return writer

    def _checkpoint(self) -> None:
        payload = {
            "id": self.last_id,
            "barrier": self.barrier,
            "ladders": {ticker: _ladder_state(ladder) for ticker, ladder in self.ladders.items()},
            "poisoned": sorted(self.poisoned),
            "poisoned_rows": self.poisoned_rows,
            "accumulators": {a.name: a.state() for a in self.accumulators},
        }
        temp = self.out_dir / f"{CHECKPOINT_NAME}.tmp"
        with temp.open("w") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, self.out_dir / CHECKPOINT_NAME)


def run_forward_pass(
    db_path: Path,
    out_dir: Path,
    emitters: Sequence[Emitter],
    *,
    accumulators: Sequence[Accumulator] = (),
    read_batch_rows: int = READ_BATCH_ROWS,
    row_group_rows: int = ROW_GROUP_ROWS,
    barrier_rows: int = BARRIER_ROWS,
    max_resident_ladders: int | None = None,
    max_id: int | None = None,
) -> PassResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = _Pass(
        out_dir, emitters, accumulators, row_group_rows, barrier_rows, max_resident_ladders
    )
    state.restore()
    select = _SELECT if max_id is None else _SELECT_BOUNDED
    bound = () if max_id is None else (max_id,)
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        # One held ORDER BY id cursor pins the recorder's WAL, which it cannot checkpoint
        # before shutdown, so every batch is its own statement carrying the last id forward.
        while True:
            batch = conn.execute(select, (state.last_id, *bound, read_batch_rows)).fetchall()
            if not batch:
                break
            for raw in batch:
                state.consume(_source_row(raw))
    finally:
        conn.close()
    if state.since_barrier:
        state.commit()
    elapsed = time.monotonic() - state.started
    logger.info(
        "forward_pass complete rows=%d id=%d barrier=%d elapsed_s=%.3f",
        state.rows,
        state.last_id,
        state.barrier,
        elapsed,
    )
    return PassResult(
        rows=state.rows,
        last_id=state.last_id,
        barriers=state.barrier,
        elapsed_s=elapsed,
        ladders=state.ladders,
        poisoned_rows=state.poisoned_rows,
    )


def _source_row(raw: tuple[int, str, str, int, str, str, str, int, int | None]) -> SourceRow:
    return SourceRow(
        id=raw[0],
        ticker=raw[1],
        received_at=datetime.strptime(raw[2], _DB_TS).replace(tzinfo=timezone.utc),
        seq=raw[3],
        side=raw[4],
        price=raw[5],
        size=raw[6],
        is_snapshot=bool(raw[7]),
        ts_ms=raw[8],
    )


def _ladder_state(ladder: Ladder) -> dict[str, JsonState]:
    batch_key = ladder.batch_key
    return {
        "levels": {
            side: {str(price): str(size) for price, size in levels.items()}
            for side, levels in ladder.levels.items()
        },
        "batch_key": None if batch_key is None else [batch_key[0].strftime(_DB_TS), batch_key[1]],
    }


def _ladder_from_state(ticker: str, state: dict[str, JsonState]) -> Ladder:
    ladder = Ladder(ticker)
    ladder.levels = {
        side: {Decimal(price): Decimal(size) for price, size in levels.items()}
        for side, levels in state["levels"].items()
    }
    batch_key = state["batch_key"]
    if batch_key is not None:
        at = datetime.strptime(batch_key[0], _DB_TS).replace(tzinfo=timezone.utc)
        ladder.batch_key = (at, batch_key[1])
    return ladder
