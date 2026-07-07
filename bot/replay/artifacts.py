import logging
import os
import sqlite3
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bot.replay.forward_pass import (
    ROW_GROUP_ROWS,
    JsonState,
    SourceRow,
    _PartitionWriter,
    _decode_ts,
)
from bot.replay.inventory import PassInventory
from bot.replay.ladder import _PRICE_EXPONENT, _SIZE_EXPONENT, _quantized, Ladder


logger = logging.getLogger(__name__)

MIN_DISK_FREE_FRACTION = Decimal("0.20")
RECORDER_DAILY_BYTES = 5_100_000_000
TRADES_BATCH_ROWS = 50_000
BUDGET_CHECK_ROWS = 100_000
LADDER_SCOPE = ("KXHIGH", "KXLOW")
# Six, not five: five populated levels span only four ticks of price, one short of the
# cumulative-through-five-ticks depth the consumer reads.
LADDER_DEPTH = 6

_ZERO_PRICE = Decimal("0").quantize(_PRICE_EXPONENT)
_ZERO_SIZE = Decimal("0").quantize(_SIZE_EXPONENT)
_ONE = Decimal("1")
_HOURS_PER_DAY = Decimal(24)
_TRADES_SELECT = (
    "SELECT id, ticker, received_at, yes_price, count, taker_side, trade_id, ts_ms "
    "FROM ws_trades WHERE id > ? ORDER BY id LIMIT ?"
)

_KEYS = [
    ("id", pa.int64()),
    ("ticker", pa.string()),
    ("received_at", pa.timestamp("us", tz="UTC")),
    ("ts_ms", pa.int64()),
]
_TOUCH = [
    ("yes_bid", pa.string()),
    ("yes_bid_depth", pa.string()),
    ("yes_ask", pa.string()),
    ("yes_ask_depth", pa.string()),
    ("no_bid", pa.string()),
    ("no_bid_depth", pa.string()),
    ("no_ask", pa.string()),
    ("no_ask_depth", pa.string()),
]

TOUCH_SCHEMA = pa.schema(_KEYS + _TOUCH)
LADDER_SCHEMA = pa.schema(
    _KEYS
    + _TOUCH
    + [
        ("yes_prices", pa.list_(pa.string())),
        ("yes_sizes", pa.list_(pa.string())),
        ("yes_levels", pa.int64()),
        ("no_prices", pa.list_(pa.string())),
        ("no_sizes", pa.list_(pa.string())),
        ("no_levels", pa.int64()),
    ]
)
TRADES_SCHEMA = pa.schema(
    _KEYS
    + [
        ("yes_price", pa.string()),
        ("no_price", pa.string()),
        ("count", pa.string()),
        ("taker_side", pa.string()),
        ("trade_id", pa.string()),
    ]
)
WINDOWS_SCHEMA = pa.schema(
    [
        ("gap_id", pa.int64()),
        ("ticker", pa.string()),
        ("start", pa.timestamp("us", tz="UTC")),
        ("end", pa.timestamp("us", tz="UTC")),
        ("detected_at", pa.timestamp("us", tz="UTC")),
        ("last_seq", pa.int64()),
        ("reason", pa.string()),
    ]
)
BOUNDARIES_SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("received_at", pa.timestamp("us", tz="UTC")),
        ("prev_seq", pa.int64()),
        ("seq", pa.int64()),
        ("kind", pa.string()),
    ]
)
COVERAGE_SCHEMA = pa.schema(
    [
        ("ticker", pa.string()),
        ("rows", pa.int64()),
        ("first_received_at", pa.timestamp("us", tz="UTC")),
        ("last_received_at", pa.timestamp("us", tz="UTC")),
    ]
)
SCALARS_SCHEMA = pa.schema([("name", pa.string()), ("value", pa.string())])


# An opaque split: parse_ticker raises on KXRAINCHIM-26JUL-1, the recorded form of the KXRAIN
# roots, and the date a ticker names is not the date its rows arrived on.
def partition_key(ticker: str, received_at: datetime) -> str:
    return f"{ticker.split('-')[0]}-{received_at.date().isoformat()}"


class TouchEmitter:
    name = "touch"
    schema = TOUCH_SCHEMA

    def emit(self, row: SourceRow, ladder: Ladder) -> Iterator[tuple[str, dict[str, object]]]:
        yield (
            partition_key(row.ticker, row.received_at),
            _touch_row(row, ladder.live("yes"), ladder.live("no")),
        )


# A prefix on the series root rather than the roots this tape happens to hold: a KXHIGH root
# first recorded after the roots were counted would otherwise drop out of the artifact silently.
class LadderEmitter:
    name = "ladder"
    schema = LADDER_SCHEMA
    scope = LADDER_SCOPE
    depth = LADDER_DEPTH

    def __init__(self) -> None:
        self.roots: set[str] = set()

    def emit(self, row: SourceRow, ladder: Ladder) -> Iterator[tuple[str, dict[str, object]]]:
        root = row.ticker.split("-")[0]
        if not root.startswith(self.scope):
            return
        self.roots.add(root)
        yes = ladder.live("yes")
        no = ladder.live("no")
        out = _touch_row(row, yes, no)
        out["yes_prices"] = [str(price) for price, _ in yes[: self.depth]]
        out["yes_sizes"] = [str(size) for _, size in yes[: self.depth]]
        out["yes_levels"] = len(yes)
        out["no_prices"] = [str(price) for price, _ in no[: self.depth]]
        out["no_sizes"] = [str(size) for _, size in no[: self.depth]]
        out["no_levels"] = len(no)
        yield partition_key(row.ticker, row.received_at), out


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ByteBudget:
    free: int
    floor: int
    recorder_day: int
    ws_raw_day: int
    wal: int
    budget_bytes: int


def byte_budget(
    *,
    f_bavail: int,
    f_frsize: int,
    f_blocks: int,
    ws_raw_daily_bytes: int,
    pass_hours: Decimal,
    min_free_fraction: Decimal = MIN_DISK_FREE_FRACTION,
    recorder_daily_bytes: int = RECORDER_DAILY_BYTES,
) -> ByteBudget:
    free = f_bavail * f_frsize
    # read_disk_free_fraction divides by f_blocks, so the watchdog's floor is a fraction of
    # total rather than of free.
    floor = int(min_free_fraction * f_blocks * f_frsize)
    wal = int(pass_hours / _HOURS_PER_DAY * recorder_daily_bytes)
    return ByteBudget(
        free=free,
        floor=floor,
        recorder_day=recorder_daily_bytes,
        ws_raw_day=ws_raw_daily_bytes,
        wal=wal,
        budget_bytes=free - floor - recorder_daily_bytes - ws_raw_daily_bytes - wal,
    )


def ws_raw_daily_bytes(raw_dir: Path) -> int:
    # The recorder gzips unconditionally and closes one file per UTC day, so the widest complete
    # day on disk is the accrual; the pre-gzip counter is several times larger.
    sizes = [path.stat().st_size for path in raw_dir.glob("*.jsonl.gz")]
    if not sizes:
        raise ValueError(f"no gzipped days under {raw_dir}")
    return max(sizes)


def measure_byte_budget(path: Path, raw_dir: Path, pass_hours: Decimal) -> ByteBudget:
    stat = os.statvfs(path)
    return byte_budget(
        f_bavail=stat.f_bavail,
        f_frsize=stat.f_frsize,
        f_blocks=stat.f_blocks,
        ws_raw_daily_bytes=ws_raw_daily_bytes(raw_dir),
        pass_hours=pass_hours,
    )


def preflight(budget: ByteBudget) -> None:
    if budget.budget_bytes < 0:
        raise BudgetExceeded(
            f"negative byte budget: free={budget.free} floor={budget.floor} "
            f"recorder_day={budget.recorder_day} ws_raw_day={budget.ws_raw_day} "
            f"wal={budget.wal} budget={budget.budget_bytes}"
        )
    logger.info(
        "byte_budget free=%d floor=%d recorder_day=%d ws_raw_day=%d wal=%d budget=%d",
        budget.free,
        budget.floor,
        budget.recorder_day,
        budget.ws_raw_day,
        budget.wal,
        budget.budget_bytes,
    )


def bytes_written(out_dir: Path) -> int:
    return sum(path.stat().st_size for path in out_dir.rglob("*") if path.is_file())


class BudgetGuard:
    name = "budget"

    def __init__(
        self, out_dir: Path, budget: ByteBudget, *, check_rows: int = BUDGET_CHECK_ROWS
    ) -> None:
        preflight(budget)
        self.out_dir = out_dir
        self.budget = budget
        self.check_rows = check_rows
        self.written = 0
        self._since_check = 0

    def observe(self, row: SourceRow) -> None:
        self._since_check += 1
        if self._since_check < self.check_rows:
            return
        self._since_check = 0
        self.written = bytes_written(self.out_dir)
        if self.written >= self.budget.budget_bytes:
            raise BudgetExceeded(
                f"artifact hit the byte budget at id={row.id}: "
                f"written={self.written} budget={self.budget.budget_bytes}"
            )

    def state(self) -> JsonState:
        return {"written": self.written}

    def restore(self, state: JsonState) -> None:
        self.written = state["written"]


@dataclass(frozen=True, slots=True)
class Projection:
    measured_bytes: int
    measured_rows: int
    measured_source_bytes: int
    total_rows: int
    total_bytes: int
    by_rows: int
    by_bytes: int
    basis: str
    projected_bytes: int
    bytes_per_row: Decimal
    budget_bytes: int
    fits: bool


def project_artifact_bytes(
    *,
    measured_bytes: int,
    measured_rows: int,
    measured_source_bytes: int,
    total_rows: int,
    total_bytes: int,
    budget: ByteBudget,
) -> Projection:
    by_rows = measured_bytes * total_rows // measured_rows
    by_bytes = measured_bytes * total_bytes // measured_source_bytes
    projected = max(by_rows, by_bytes)
    projection = Projection(
        measured_bytes=measured_bytes,
        measured_rows=measured_rows,
        measured_source_bytes=measured_source_bytes,
        total_rows=total_rows,
        total_bytes=total_bytes,
        by_rows=by_rows,
        by_bytes=by_bytes,
        basis="rows" if by_rows >= by_bytes else "bytes",
        projected_bytes=projected,
        bytes_per_row=Decimal(measured_bytes) / Decimal(measured_rows),
        budget_bytes=budget.budget_bytes,
        fits=projected <= budget.budget_bytes,
    )
    logger.info(
        "projection by_rows=%d by_bytes=%d basis=%s projected=%d budget=%d fits=%s",
        projection.by_rows,
        projection.by_bytes,
        projection.basis,
        projection.projected_bytes,
        projection.budget_bytes,
        projection.fits,
    )
    return projection


def write_trades(
    db_path: Path,
    out_dir: Path,
    budget: ByteBudget,
    *,
    batch_rows: int = TRADES_BATCH_ROWS,
    row_group_rows: int = ROW_GROUP_ROWS,
) -> int:
    preflight(budget)
    writers: dict[str, _PartitionWriter] = {}
    rows = 0
    last_id = 0
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        while True:
            batch = conn.execute(_TRADES_SELECT, (last_id, batch_rows)).fetchall()
            if not batch:
                break
            for raw in batch:
                last_id = raw[0]
                rows += 1
                partition, out = _trade_row(raw)
                writer = writers.get(partition)
                if writer is None:
                    writer = _PartitionWriter(
                        _stamped(out_dir / "trades", partition), TRADES_SCHEMA, row_group_rows
                    )
                    writers[partition] = writer
                writer.add(out)
            written = bytes_written(out_dir)
            if written >= budget.budget_bytes:
                raise BudgetExceeded(
                    f"trades hit the byte budget at id={last_id}: "
                    f"written={written} budget={budget.budget_bytes}"
                )
    finally:
        conn.close()
    for writer in writers.values():
        writer.close()
    logger.info("trades rows=%d id=%d partitions=%d", rows, last_id, len(writers))
    return rows


def write_inventory(
    out_dir: Path,
    inventory: PassInventory,
    budget: ByteBudget,
    *,
    ladder: LadderEmitter | None = None,
    extra: Mapping[str, str] | None = None,
) -> list[Path]:
    preflight(budget)
    scalars = {
        "gap_rows": str(inventory.gap_rows),
        "seq_skips": str(inventory.seq_skips),
        "seq_resubscribes": str(inventory.seq_resubscribes),
        "tickers": str(len(inventory.coverage)),
        "budget_free_bytes": str(budget.free),
        "budget_floor_bytes": str(budget.floor),
        "budget_recorder_day_bytes": str(budget.recorder_day),
        "budget_ws_raw_day_bytes": str(budget.ws_raw_day),
        "budget_wal_bytes": str(budget.wal),
        "budget_bytes": str(budget.budget_bytes),
        "ladder_scope": "none" if ladder is None else ",".join(ladder.scope),
        "ladder_roots": "none" if ladder is None else _emitted_roots(out_dir / ladder.name),
        "ladder_depth": "none" if ladder is None else str(ladder.depth),
        "bytes_written": str(bytes_written(out_dir)),
        **(extra or {}),
    }
    tables = [
        (
            "windows",
            WINDOWS_SCHEMA,
            [
                {
                    "gap_id": window.gap_id,
                    "ticker": window.ticker,
                    "start": window.start,
                    "end": window.end,
                    "detected_at": window.detected_at,
                    "last_seq": window.last_seq,
                    "reason": window.reason,
                }
                for window in inventory.windows
            ],
        ),
        (
            "boundaries",
            BOUNDARIES_SCHEMA,
            [
                {
                    "id": boundary.id,
                    "received_at": boundary.received_at,
                    "prev_seq": boundary.prev_seq,
                    "seq": boundary.seq,
                    "kind": boundary.kind,
                }
                for boundary in inventory.boundaries
            ],
        ),
        (
            "coverage",
            COVERAGE_SCHEMA,
            [
                {
                    "ticker": row.ticker,
                    "rows": row.rows,
                    "first_received_at": row.first_received_at,
                    "last_received_at": row.last_received_at,
                }
                for row in inventory.coverage
            ],
        ),
        ("scalars", SCALARS_SCHEMA, [{"name": k, "value": v} for k, v in scalars.items()]),
    ]
    directory = out_dir / "inventory"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, schema, rows in tables:
        path = _stamped(directory, name)
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
        paths.append(path)
    logger.info("inventory %s", " ".join(f"{k}={v}" for k, v in scalars.items()))
    return paths


# Every parquet name under a pass output directory carries the barrier that closed it, and the
# resume sweep parses that stamp off every file it finds; barrier zero is never above a
# checkpoint's, so a hand-written artifact stamped this way survives a resume.
def _stamped(directory: Path, name: str) -> Path:
    return directory / f"{name}-b000000.parquet"


# A resumed pass re-reads only the rows above its checkpoint, so the emitter's in-process root
# set names the last segment alone; the directory carries every root the pass ever wrote.
def _emitted_roots(directory: Path) -> str:
    return ",".join(sorted({path.name.split("-")[0] for path in directory.glob("*.parquet")}))


def _best(live: list[tuple[Decimal, Decimal]]) -> tuple[Decimal, Decimal]:
    if not live:
        return _ZERO_PRICE, _ZERO_SIZE
    return live[0]


def _touch_row(
    row: SourceRow,
    yes: list[tuple[Decimal, Decimal]],
    no: list[tuple[Decimal, Decimal]],
) -> dict[str, object]:
    yes_bid, yes_bid_depth = _best(yes)
    no_bid, no_bid_depth = _best(no)
    return {
        "id": row.id,
        "ticker": row.ticker,
        "received_at": row.received_at,
        "ts_ms": row.ts_ms,
        "yes_bid": str(yes_bid),
        "yes_bid_depth": str(yes_bid_depth),
        "yes_ask": str((_ONE - no_bid).quantize(_PRICE_EXPONENT)),
        "yes_ask_depth": str(no_bid_depth),
        "no_bid": str(no_bid),
        "no_bid_depth": str(no_bid_depth),
        "no_ask": str((_ONE - yes_bid).quantize(_PRICE_EXPONENT)),
        "no_ask_depth": str(yes_bid_depth),
    }


def _trade_row(
    raw: tuple[int, str, str, str, str, str, str, int],
) -> tuple[str, dict[str, object]]:
    ticker = raw[1]
    received_at = _decode_ts(raw[2])
    yes_price = _quantized(ticker, "yes_price", raw[3], _PRICE_EXPONENT)
    return partition_key(ticker, received_at), {
        "id": raw[0],
        "ticker": ticker,
        "received_at": received_at,
        "ts_ms": raw[7],
        "yes_price": str(yes_price),
        # no_price is dropped at persist and is the complement of the yes side.
        "no_price": str((_ONE - yes_price).quantize(_PRICE_EXPONENT)),
        "count": str(_quantized(ticker, "count", raw[4], _SIZE_EXPONENT)),
        "taker_side": raw[5],
        "trade_id": raw[6],
    }
