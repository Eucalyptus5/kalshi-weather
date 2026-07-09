import logging
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from bot.lag.event_study import OrderbookSnapshotRow
from bot.lag.ws_book import WsGapError, book_state_at
from bot.replay.forward_pass import _COLUMNS, _DB_TS, SourceRow, _decode_ts, _source_row
from bot.replay.inventory import (
    ExclusionInventory,
    ExclusionWindow,
    check_excluded,
    read_gap_rows,
)
from bot.replay.ladder import _PRICE_EXPONENT, _SIZE_EXPONENT, _quantized, BookEvent, Ladder


logger = logging.getLogger(__name__)

AGREED = "agreed"
AGREED_ON_RAISE = "agreed_on_raise"
DISAGREED = "disagreed"
EXCLUDED = "excluded_tied_delta"

GAP_EDGE_ROWS = 4
INTERIOR_ROWS = 5
SNAPSHOT_BATCHES = 3
POINTS_PER_TICKER = INTERIOR_ROWS + 3 * SNAPSHOT_BATCHES + 1
CHICAGO = "CHI"

TICK = timedelta(microseconds=1)

_ZERO = Decimal("0")

_STREAM = (
    "SELECT id, received_at, seq, side, price, size, is_snapshot FROM ws_book_events "
    "WHERE ticker = ? AND received_at <= ? ORDER BY received_at, id"
)
# Copied from bot.lag.ws_book rather than imported, for the reason ladder._best gives, and must
# stay identical to it: _oracle_levels has to govern off the same batch and deltas book_state_at
# used, or the re-derived ladder describes a different book than the touch row it is compared to.
_LATEST_SNAPSHOT = (
    "SELECT received_at, seq FROM ws_book_events "
    "WHERE ticker = ? AND is_snapshot = 1 AND received_at <= ? "
    "ORDER BY received_at DESC, id DESC LIMIT 1"
)
_SNAPSHOT_BATCH = (
    "SELECT side, price, size FROM ws_book_events "
    "WHERE ticker = ? AND is_snapshot = 1 AND received_at = ? AND seq = ? "
    "ORDER BY id"
)
_DELTAS = (
    "SELECT side, price, size FROM ws_book_events "
    "WHERE ticker = ? AND is_snapshot = 0 AND received_at > ? AND received_at <= ? "
    "ORDER BY received_at, id"
)
_TIED_DELTA = (
    "SELECT 1 FROM ws_book_events WHERE ticker = ? AND is_snapshot = 0 AND received_at = ? LIMIT 1"
)
_ROW_AT_OR_AFTER = "SELECT 1 FROM ws_book_events WHERE ticker = ? AND received_at >= ? LIMIT 1"
_MIN_ID = "SELECT MIN(id) FROM ws_book_events"
_MAX_ID = "SELECT MAX(id) FROM ws_book_events"
_ID_AT_OR_AFTER = "SELECT id, received_at FROM ws_book_events WHERE id >= ? ORDER BY id LIMIT 1"
_RUN_UP_TO_ID = f"SELECT {_COLUMNS} FROM ws_book_events WHERE id <= ? ORDER BY id DESC LIMIT ?"
_TICKER_COUNTS = (
    "SELECT ticker, COUNT(*) FROM ws_book_events "
    "WHERE received_at >= ? AND received_at < ? GROUP BY ticker"
)
_TICKER_STREAM = (
    "SELECT received_at, seq, is_snapshot FROM ws_book_events "
    "WHERE ticker = ? AND received_at >= ? AND received_at < ? ORDER BY received_at, id"
)
_SNAPSHOT_BEFORE = (
    "SELECT received_at FROM ws_book_events "
    "WHERE ticker = ? AND is_snapshot = 1 AND received_at <= ? "
    "ORDER BY received_at DESC LIMIT 1"
)
_LAST_BEFORE = (
    "SELECT received_at FROM ws_book_events "
    "WHERE ticker = ? AND received_at < ? ORDER BY received_at DESC LIMIT 1"
)
_FIRST_AT_OR_AFTER = (
    "SELECT received_at FROM ws_book_events "
    "WHERE ticker = ? AND received_at >= ? ORDER BY received_at LIMIT 1"
)


@dataclass(frozen=True, slots=True)
class SamplePoint:
    ticker: str
    t: datetime
    kind: str
    cohort: str


@dataclass(frozen=True, slots=True)
class BookView:
    touch: tuple[tuple[str, str], ...]
    depth: tuple[tuple[str, int], ...]
    ladder: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]


@dataclass(frozen=True, slots=True)
class PointResult:
    point: SamplePoint
    status: str
    blind: bool
    detail: str
    pass_view: BookView | None
    oracle_view: BookView | None


@dataclass(frozen=True, slots=True)
class ParityResult:
    results: tuple[PointResult, ...]
    rows_read: int

    def counts(self) -> dict[str, int]:
        out = {AGREED: 0, AGREED_ON_RAISE: 0, DISAGREED: 0, EXCLUDED: 0}
        for result in self.results:
            out[result.status] = out.get(result.status, 0) + 1
        return out

    def disagreements(self) -> tuple[PointResult, ...]:
        return tuple(r for r in self.results if r.status == DISAGREED)

    def scalars(self) -> dict[str, str]:
        counts = self.counts()
        out = {
            "parity_points": str(len(self.results)),
            "parity_compared": str(counts[AGREED] + counts[DISAGREED]),
            "parity_blind": str(sum(1 for r in self.results if r.blind)),
            "parity_rows_read": str(self.rows_read),
        }
        for status, n in sorted(counts.items()):
            out[f"parity_{status}"] = str(n)
        for field, values in (("kind", self.kinds()), ("cohort", self.cohorts())):
            for name, n in sorted(values.items()):
                out[f"parity_{field}_{name}"] = str(n)
        return out

    def kinds(self) -> dict[str, int]:
        return _tally(result.point.kind for result in self.results)

    def cohorts(self) -> dict[str, int]:
        return _tally(result.point.cohort for result in self.results)


def _tally(labels: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for label in labels:
        out[label] = out.get(label, 0) + 1
    return out


def compare_points(
    db_path: Path,
    points: Sequence[SamplePoint],
    windows: Sequence[ExclusionWindow],
    *,
    exclude_tied_deltas: bool = True,
) -> ParityResult:
    ordered = list(points)
    wanted: dict[str, set[datetime]] = {}
    for point in ordered:
        wanted.setdefault(point.ticker, set()).add(point.t)

    conn = _connect(db_path)
    rows_read = 0
    try:
        views: dict[str, dict[datetime, BookView]] = {}
        for ticker in sorted(wanted):
            views[ticker], read = _pass_views(conn, ticker, sorted(wanted[ticker]))
            rows_read += read
            logger.info(
                "parity replayed ticker=%s rows=%d points=%d", ticker, read, len(views[ticker])
            )
        results = [
            _evaluate(
                conn, db_path, point, views[point.ticker][point.t], windows, exclude_tied_deltas
            )
            for point in ordered
        ]
    finally:
        conn.close()
    return ParityResult(results=tuple(results), rows_read=rows_read)


def sample_points(
    db_path: Path,
    budget: int,
    windows: Sequence[ExclusionWindow],
    *,
    since: datetime,
    until: datetime,
) -> list[SamplePoint]:
    since_db = _db(since)
    until_db = _db(until)
    conn = _connect(db_path)
    try:
        ranked = sorted(
            conn.execute(_TICKER_COUNTS, (since_db, until_db)).fetchall(),
            key=lambda row: (-row[1], row[0]),
        )
        points: list[SamplePoint] = []
        for cohort, ticker, rows in _cohorts(ranked, budget):
            points.extend(_ticker_points(conn, ticker, rows, cohort, since_db, until_db))
        for window in windows:
            points.extend(_window_points(conn, ranked, window))
    finally:
        conn.close()
    unique = sorted(set(points), key=lambda point: (point.ticker, point.t, point.kind))
    logger.info("parity sampled points=%d tickers=%d", len(unique), len({p.ticker for p in unique}))
    return unique


def _cohorts(ranked: list[tuple[str, int]], budget: int) -> list[tuple[str, str, int]]:
    tickers = max(3, -(-budget // POINTS_PER_TICKER))
    quiet_n = max(1, tickers // 3)
    spread_n = max(1, tickers - quiet_n - 1)
    step = max(1, len(ranked) // spread_n)
    chicago = [row for row in ranked if CHICAGO in row[0].split("-")[0]][:1]
    if not chicago:
        raise ValueError(f"no recorded ticker whose series root contains {CHICAGO}")
    picked: dict[str, tuple[str, str, int]] = {}
    for cohort, chosen in (
        ("chicago", chicago),
        ("quiet", list(reversed(ranked[-quiet_n:]))),
        ("spread", [ranked[i * step] for i in range(spread_n) if i * step < len(ranked)]),
    ):
        for ticker, rows in chosen:
            picked.setdefault(ticker, (cohort, ticker, rows))
    return list(picked.values())


def _ticker_points(
    conn: sqlite3.Connection,
    ticker: str,
    rows: int,
    cohort: str,
    since_db: str,
    until_db: str,
) -> list[SamplePoint]:
    ranks = iter(sorted({rows * (i + 1) // (INTERIOR_ROWS + 1) for i in range(INTERIOR_ROWS)}))
    target = next(ranks, None)
    interior: list[datetime] = []
    batches: list[tuple[datetime, datetime]] = []
    batch_key: tuple[str, int] | None = None
    previous: datetime | None = None
    last = None
    for index, (received_at_db, seq, is_snapshot) in enumerate(
        conn.execute(_TICKER_STREAM, (ticker, since_db, until_db))
    ):
        received_at = _decode_ts(received_at_db)
        if is_snapshot and (received_at_db, seq) != batch_key:
            batch_key = (received_at_db, seq)
            if previous is not None:
                batches.append((previous, received_at))
        if index == target:
            interior.append(received_at)
            target = next(ranks, None)
        previous = received_at
        last = received_at

    points = [SamplePoint(ticker, t, "interior", cohort) for t in interior]
    for rank in range(SNAPSHOT_BATCHES):
        if not batches:
            break
        before, at = batches[len(batches) * rank // SNAPSHOT_BATCHES]
        points.append(SamplePoint(ticker, before, "snapshot", cohort))
        points.append(SamplePoint(ticker, at, "snapshot", cohort))
        points.append(SamplePoint(ticker, at + TICK, "snapshot", cohort))
    points.append(SamplePoint(ticker, last + TICK, "blind", cohort))
    return points


# The oracle answers None rather than raising for a ticker with no snapshot before the gap, so
# the point that has to agree on raising needs a ticker that was already subscribed.
def _window_points(
    conn: sqlite3.Connection,
    ranked: list[tuple[str, int]],
    window: ExclusionWindow,
) -> list[SamplePoint]:
    detected_at_db = _db(window.detected_at)
    candidates = [window.ticker] if window.ticker else [row[0] for row in ranked]
    ticker = next(
        (
            name
            for name in candidates
            if conn.execute(_SNAPSHOT_BEFORE, (name, detected_at_db)).fetchone() is not None
        ),
        None,
    )
    if ticker is None:
        raise ValueError(
            f"ws gap {window.gap_id} has no ticker with a snapshot at or before it: "
            f"ticker={window.ticker} detected_at={detected_at_db} reason={window.reason}"
        )
    points = [SamplePoint(ticker, window.detected_at, "post_gap", "gap")]
    before = conn.execute(_LAST_BEFORE, (ticker, detected_at_db)).fetchone()
    points.append(SamplePoint(ticker, _decode_ts(before[0]), "gap_edge", "gap"))
    after = conn.execute(_FIRST_AT_OR_AFTER, (ticker, detected_at_db)).fetchone()
    if after is not None:
        points.append(SamplePoint(ticker, _decode_ts(after[0]), "gap_edge", "gap"))
        points.append(SamplePoint(ticker, _decode_ts(after[0]) + TICK, "gap_edge", "gap"))
    return points


def gap_windows(db_path: Path, *, since: datetime, until: datetime) -> list[ExclusionWindow]:
    gaps = [row for row in read_gap_rows(db_path) if since <= row.detected_at < until]
    inventory = ExclusionInventory(gaps)
    conn = _connect(db_path)
    try:
        for gap in sorted(gaps, key=lambda row: (row.detected_at, row.id)):
            for row in _gap_edge_rows(conn, _db(gap.detected_at)):
                inventory.observe(row)
    finally:
        conn.close()
    return inventory.windows()


def _gap_edge_rows(conn: sqlite3.Connection, detected_at_db: str) -> list[SourceRow]:
    crossing = _crossing_id(conn, detected_at_db)
    anchor = conn.execute(_MAX_ID).fetchone()[0] if crossing is None else crossing
    if anchor is None:
        return []
    raw = conn.execute(_RUN_UP_TO_ID, (anchor, GAP_EDGE_ROWS)).fetchall()
    rows = [_source_row(one) for one in reversed(raw)]
    for earlier, later in zip(rows, rows[1:]):
        if later.received_at < earlier.received_at:
            raise ValueError(
                f"received_at falls from id={earlier.id} to id={later.id}: "
                f"{earlier.received_at} then {later.received_at}"
            )
    if crossing is not None and len(rows) > 1 and _db(rows[-2].received_at) >= detected_at_db:
        raise ValueError(
            f"id={rows[-2].id} at {rows[-2].received_at} precedes the crossing at "
            f"id={rows[-1].id} yet is not below detected_at={detected_at_db}"
        )
    return rows


# received_at is non-decreasing in id because _persist_ws_events is the table's only writer and
# appends in arrival order, so the crossing is reachable in a logarithmic number of rowid seeks
# rather than by streaming the tape.
def _crossing_id(conn: sqlite3.Connection, detected_at_db: str) -> int | None:
    lo = conn.execute(_MIN_ID).fetchone()[0]
    hi = conn.execute(_MAX_ID).fetchone()[0]
    if lo is None:
        return None
    found = None
    while lo <= hi:
        mid = (lo + hi) // 2
        row = conn.execute(_ID_AT_OR_AFTER, (mid,)).fetchone()
        if row is None:
            hi = mid - 1
        elif row[1] >= detected_at_db:
            found = row[0]
            hi = mid - 1
        else:
            lo = row[0] + 1
    return found


def _evaluate(
    conn: sqlite3.Connection,
    db_path: Path,
    point: SamplePoint,
    pass_view: BookView,
    windows: Sequence[ExclusionWindow],
    exclude_tied_deltas: bool,
) -> PointResult:
    t_db = _db(point.t)
    blind = conn.execute(_ROW_AT_OR_AFTER, (point.ticker, t_db)).fetchone() is None
    oracle_row = None
    oracle_raised = ""
    try:
        oracle_row = book_state_at(db_path, point.ticker, point.t)
    except WsGapError as exc:
        oracle_raised = str(exc)
    pass_raised = ""
    try:
        check_excluded(windows, point.ticker, point.t)
    except WsGapError as exc:
        pass_raised = str(exc)
    if oracle_raised and pass_raised:
        return PointResult(point, AGREED_ON_RAISE, blind, "", None, None)
    if oracle_raised or pass_raised:
        side = "oracle" if oracle_raised else "pass"
        detail = (
            f"{point.ticker} {point.t.isoformat()} only the {side} path raised: "
            f"{oracle_raised or pass_raised}"
        )
        return PointResult(point, DISAGREED, blind, detail, None, None)
    governing = _oracle_levels(conn, point.ticker, t_db)
    if governing is None:
        detail = f"{point.ticker} {point.t.isoformat()} has no oracle snapshot at or before t"
        return PointResult(point, DISAGREED, blind, detail, pass_view, None)
    if exclude_tied_deltas and _tied(conn, point.ticker, governing[0]):
        return PointResult(point, EXCLUDED, blind, "", None, None)
    oracle_view = _view(point.ticker, oracle_row, governing[1])
    if pass_view == oracle_view:
        return PointResult(point, AGREED, blind, "", pass_view, oracle_view)
    detail = _diff(point, pass_view, oracle_view)
    return PointResult(point, DISAGREED, blind, detail, pass_view, oracle_view)


def _tied(conn: sqlite3.Connection, ticker: str, snapshot_at_db: str) -> bool:
    return conn.execute(_TIED_DELTA, (ticker, snapshot_at_db)).fetchone() is not None


def _diff(point: SamplePoint, pass_view: BookView, oracle_view: BookView) -> str:
    parts = [f"{point.ticker} {point.t.isoformat()}"]
    for (name, mine), (_, theirs) in zip(
        pass_view.touch + pass_view.depth, oracle_view.touch + oracle_view.depth
    ):
        if mine != theirs:
            parts.append(f"{name} pass={mine} oracle={theirs}")
    for (side, mine), (_, theirs) in zip(pass_view.ladder, oracle_view.ladder):
        if mine != theirs:
            parts.append(f"{side}_ladder pass={_levels(mine)} oracle={_levels(theirs)}")
    return "; ".join(parts)


def _levels(levels: tuple[tuple[str, str], ...]) -> str:
    return "[" + ",".join(f"{price}@{size}" for price, size in levels) + "]"


def _pass_views(
    conn: sqlite3.Connection,
    ticker: str,
    times: list[datetime],
) -> tuple[dict[datetime, BookView], int]:
    ladder = Ladder(ticker)
    pending = list(times)
    out: dict[datetime, BookView] = {}
    last_id = 0
    rows = 0
    for row_id, received_at_db, seq, side, price, size, is_snapshot in conn.execute(
        _STREAM, (ticker, _db(times[-1]))
    ):
        if row_id <= last_id:
            raise ValueError(f"{ticker} id {row_id} follows id {last_id} in received_at order")
        last_id = row_id
        rows += 1
        received_at = _decode_ts(received_at_db)
        while pending and pending[0] < received_at:
            t = pending.pop(0)
            out[t] = _view(ticker, ladder.row(t), ladder.levels)
        ladder.apply(BookEvent(received_at, seq, side, price, size, bool(is_snapshot)))
    for t in pending:
        out[t] = _view(ticker, ladder.row(t), ladder.levels)
    return out, rows


def _oracle_levels(
    conn: sqlite3.Connection,
    ticker: str,
    t_db: str,
) -> tuple[str, dict[str, dict[Decimal, Decimal]]] | None:
    latest = conn.execute(_LATEST_SNAPSHOT, (ticker, t_db)).fetchone()
    if latest is None:
        return None
    snapshot_at_db, seq = latest
    levels: dict[str, dict[Decimal, Decimal]] = {"yes": {}, "no": {}}
    for side, price, size in conn.execute(_SNAPSHOT_BATCH, (ticker, snapshot_at_db, seq)):
        levels[side][_quantized(ticker, "price", price, _PRICE_EXPONENT)] = _quantized(
            ticker, "size", size, _SIZE_EXPONENT
        )
    for side, price, size in conn.execute(_DELTAS, (ticker, snapshot_at_db, t_db)):
        key = _quantized(ticker, "price", price, _PRICE_EXPONENT)
        total = levels[side].get(key, _ZERO) + _quantized(ticker, "size", size, _SIZE_EXPONENT)
        if total < _ZERO:
            raise ValueError(f"negative level for {ticker} side={side} price={key} size={total}")
        if total == _ZERO:
            levels[side].pop(key, None)
        else:
            levels[side][key] = total
    return snapshot_at_db, levels


def _view(
    ticker: str,
    row: OrderbookSnapshotRow,
    levels: dict[str, dict[Decimal, Decimal]],
) -> BookView:
    return BookView(
        touch=(
            ("yes_bid", _price(ticker, "yes_bid", row.yes_bid)),
            ("yes_ask", _price(ticker, "yes_ask", row.yes_ask)),
            ("no_bid", _price(ticker, "no_bid", row.no_bid)),
            ("no_ask", _price(ticker, "no_ask", row.no_ask)),
        ),
        depth=(
            ("yes_bid_depth", int(row.yes_bid_depth)),
            ("yes_ask_depth", int(row.yes_ask_depth)),
            ("no_bid_depth", int(row.no_bid_depth)),
            ("no_ask_depth", int(row.no_ask_depth)),
        ),
        ladder=tuple(
            (
                side,
                tuple(
                    (str(price), str(size))
                    for price, size in sorted(
                        ((p, s) for p, s in levels[side].items() if s > _ZERO), reverse=True
                    )
                ),
            )
            for side in ("yes", "no")
        ),
    )


def _price(ticker: str, field: str, value: Decimal) -> str:
    return str(_quantized(ticker, field, str(value), _PRICE_EXPONENT))


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)


def _db(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime(_DB_TS)
