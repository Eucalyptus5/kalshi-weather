import gzip
import json
import logging
from bisect import bisect_left
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bot.replay.blind_windows import BlindWindow


logger = logging.getLogger(__name__)

# The tape sink writes json.dumps({"received_at": ..., "raw": ...}), so the stamp starts at a
# fixed offset and runs to the next quote; comparing it as bytes only orders in time if every
# stamp carries the six microsecond digits isoformat drops when the microsecond is zero.
_STAMP_AT = len('{"received_at": "')
_SECONDS_CHARS = len("2026-07-30T03:01:02")
_ZERO_MICROSECOND_CHARS = len("2026-07-30T03:01:02+00:00")
_TAPE_TS = "%Y-%m-%dT%H:%M:%S.%f+00:00"

# The frame is escaped into the record's raw string, so the exchange's compact type field survives
# as these bytes. Screening on the channel name alone also matches the subscribe ack that names the
# channel, and an ack lands inside a blind window by construction.
_SNAPSHOT_TYPE = rb"\"type\":\"orderbook_snapshot\""
_DELTA_TYPE = rb"\"type\":\"orderbook_delta\""


@dataclass(frozen=True, slots=True)
class ClearFrame:
    received_at: datetime
    ticker: str


# An empty side is the absence of the key, so a snapshot clearing the book carries neither
# yes_dollars_fp nor no_dollars_fp and decodes to no rows at all. Twenty million lines a day are
# too many to parse, hence the byte screen ahead of json.loads.
def read_clear_frames(path: Path) -> Iterator[ClearFrame]:
    with gzip.open(path, "rb") as handle:
        for line in handle:
            if _SNAPSHOT_TYPE not in line or b"dollars_fp" in line:
                continue
            record = json.loads(line)
            frame = json.loads(record["raw"])
            yield ClearFrame(
                received_at=datetime.fromisoformat(record["received_at"]),
                ticker=frame["msg"]["market_ticker"],
            )


@dataclass(frozen=True, slots=True)
class TapeCheck:
    sampled: int
    agreed: int
    wider: int
    extra_frames: int
    falsified: int
    falsifying: tuple[BlindWindow, ...]


def count_frames_in_windows(
    paths: Sequence[Path], windows: Sequence[BlindWindow]
) -> dict[int, int]:
    ordered = sorted(windows, key=lambda window: window.start)
    starts = [window.start.strftime(_TAPE_TS).encode() for window in ordered]
    ends = [window.end.strftime(_TAPE_TS).encode() for window in ordered]
    ids = [window.boundary_id for window in ordered]
    counts = dict.fromkeys(ids, 0)
    for path in paths:
        lines = 0
        inside = 0
        with gzip.open(path, "rb") as handle:
            for line in handle:
                lines += 1
                if _SNAPSHOT_TYPE not in line and _DELTA_TYPE not in line:
                    continue
                stamp = line[_STAMP_AT : line.index(b'"', _STAMP_AT)]
                if len(stamp) == _ZERO_MICROSECOND_CHARS:
                    stamp = stamp[:_SECONDS_CHARS] + b".000000" + stamp[_SECONDS_CHARS:]
                index = bisect_left(starts, stamp) - 1
                if index >= 0 and stamp < ends[index]:
                    counts[ids[index]] += 1
                    inside += 1
        logger.info("raw_tape path=%s lines=%d inside=%d", path, lines, inside)
    return counts


def check_tape_counts(windows: Sequence[BlindWindow], counts: Mapping[int, int]) -> TapeCheck:
    agreed = 0
    wider = 0
    extra_frames = 0
    falsifying: list[BlindWindow] = []
    for window in windows:
        # Every frame is stamped just before the row it becomes, so the one behind the message at
        # start falls outside and the one behind the closing message falls in: a window that agrees
        # holds exactly the burst it covers, and a wider one holds what the database never kept.
        count = counts[window.boundary_id]
        if count < window.burst_messages:
            falsifying.append(window)
        elif count == window.burst_messages:
            agreed += 1
        else:
            wider += 1
            extra_frames += count - window.burst_messages
    return TapeCheck(
        sampled=len(windows),
        agreed=agreed,
        wider=wider,
        extra_frames=extra_frames,
        falsified=len(falsifying),
        falsifying=tuple(falsifying),
    )
