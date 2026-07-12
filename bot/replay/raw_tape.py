import gzip
import logging
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
                if b"orderbook_snapshot" not in line and b"orderbook_delta" not in line:
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
        # The frame behind the row at end is stamped just before it and falls inside, so a window
        # that agrees with the tape holds exactly one frame and a wider one holds the frames the
        # database never persisted.
        count = counts[window.boundary_id]
        if count == 0:
            falsifying.append(window)
        elif count == 1:
            agreed += 1
        else:
            wider += 1
            extra_frames += count - 1
    return TapeCheck(
        sampled=len(windows),
        agreed=agreed,
        wider=wider,
        extra_frames=extra_frames,
        falsified=len(falsifying),
        falsifying=tuple(falsifying),
    )
