import gzip
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from bot.main import WsRawTape
from bot.replay.blind_windows import BlindWindow
from bot.replay.raw_tape import (
    ClearFrame,
    check_tape_counts,
    count_frames_in_windows,
    read_clear_frames,
)


UTC = timezone.utc
AUS = "KXHIGHAUS-26JUL29-B100.5"
DEN = "KXHIGHDEN-26JUL29-B85.5"


def at(hour: int, minute: int, second: int, microsecond: int = 0) -> datetime:
    return datetime(2026, 7, 30, hour, minute, second, microsecond, tzinfo=UTC)


# The exchange sends compact json and terminates every frame with a newline, and the byte screen
# under test matches that spelling of the type field, so fixtures have to carry it too.
def wire(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, separators=(",", ":")) + "\n"


def frame(kind: str, seq: int = 1, ticker: str = AUS) -> str:
    msg = {"market_ticker": ticker, "market_id": "68b7d0a1"}
    return wire({"type": kind, "sid": 1, "seq": seq, "msg": msg})


def levelled(kind: str, seq: int, *sides: str, ticker: str = AUS) -> str:
    msg: dict[str, object] = {"market_ticker": ticker, "market_id": "68b7d0a1"}
    for side in sides:
        msg[side] = [["0.0800", "300.00"], ["0.2200", "333.00"]]
    return wire({"type": kind, "sid": 1, "seq": seq, "msg": msg})


def ack(channel: str, sid: int = 1) -> str:
    return wire({"type": "subscribed", "id": 10, "msg": {"channel": channel, "sid": sid}})


def write_tape(directory: Path, frames: Sequence[tuple[datetime, str]]) -> list[Path]:
    tape = WsRawTape(directory)
    for received_at, raw in frames:
        tape.write(raw, received_at)
    tape.close()
    return sorted(directory.glob("*.jsonl.gz"))


def window(boundary_id: int, start: datetime, end: datetime) -> BlindWindow:
    return BlindWindow(
        boundary_id=boundary_id,
        prev_id=boundary_id - 1,
        ticker="",
        start=start,
        end=end,
        prev_seq=2,
        seq=1,
        gap_id=None,
        gap_reason=None,
        gap_detected_at=None,
    )


def test_a_window_that_agrees_with_the_tape_counts_the_boundary_frame_alone(
    tmp_path: Path,
) -> None:
    boundary = window(10, at(3, 1, 1), at(3, 1, 2, 398_909))
    paths = write_tape(
        tmp_path,
        [
            (at(3, 1, 0, 999_990), frame("orderbook_delta")),
            (at(3, 1, 0, 999_995), frame("orderbook_delta", 2)),
            (at(3, 1, 2, 398_900), frame("orderbook_snapshot", 1)),
            (at(3, 1, 3), frame("orderbook_delta", 2)),
        ],
    )

    assert [path.name for path in paths] == ["2026-07-30.jsonl.gz"]
    assert count_frames_in_windows(paths, [boundary]) == {10: 1}


def test_frames_the_database_never_kept_make_the_window_wider_than_the_tape(
    tmp_path: Path,
) -> None:
    boundary = window(10, at(3, 1, 1), at(3, 1, 5))
    paths = write_tape(
        tmp_path,
        [
            (at(3, 1, 0, 999_990), frame("orderbook_delta")),
            (at(3, 1, 2), frame("orderbook_delta", 2)),
            (at(3, 1, 3), frame("orderbook_delta", 3)),
            (at(3, 1, 4, 999_990), frame("orderbook_snapshot", 1)),
        ],
    )

    counts = count_frames_in_windows(paths, [boundary])
    check = check_tape_counts([boundary], counts)

    assert counts == {10: 3}
    assert (check.sampled, check.agreed, check.wider, check.extra_frames) == (1, 0, 1, 2)
    assert (check.falsified, check.falsifying) == (0, ())


def test_a_tape_with_nothing_inside_the_window_falsifies_it(tmp_path: Path) -> None:
    boundary = window(10, at(3, 1, 1), at(3, 1, 2))
    paths = write_tape(
        tmp_path,
        [
            (at(3, 1, 0, 900_000), frame("orderbook_delta")),
            (at(3, 1, 2, 500_000), frame("orderbook_delta", 2)),
        ],
    )

    counts = count_frames_in_windows(paths, [boundary])
    check = check_tape_counts([boundary], counts)

    assert counts == {10: 0}
    assert (check.sampled, check.agreed, check.wider, check.extra_frames) == (1, 0, 0, 0)
    assert check.falsified == 1
    assert check.falsifying == (boundary,)


def test_only_orderbook_frames_are_counted(tmp_path: Path) -> None:
    boundary = window(10, at(3, 1, 1), at(3, 1, 5))
    paths = write_tape(
        tmp_path,
        [
            (at(3, 1, 2), frame("trade")),
            (at(3, 1, 2, 500_000), frame("ticker_v2")),
            (at(3, 1, 3), frame("market_lifecycle_v2")),
            (at(3, 1, 4), frame("orderbook_delta", 2)),
        ],
    )

    assert count_frames_in_windows(paths, [boundary]) == {10: 1}


def test_a_subscribe_ack_naming_an_orderbook_channel_is_not_counted(tmp_path: Path) -> None:
    boundary = window(10, at(5, 0, 12), at(5, 0, 14))
    paths = write_tape(
        tmp_path,
        [
            (at(5, 0, 13, 3_749), ack("orderbook_delta")),
            (at(5, 0, 13, 100_000), ack("orderbook_snapshot", 2)),
            (at(5, 0, 13, 200_000), frame("orderbook_delta", 2)),
        ],
    )

    first = gzip.decompress(paths[0].read_bytes()).splitlines()[0].decode()
    assert first == (
        '{"received_at": "2026-07-30T05:00:13.003749+00:00", "raw": '
        '"{\\"type\\":\\"subscribed\\",\\"id\\":10,\\"msg\\":'
        '{\\"channel\\":\\"orderbook_delta\\",\\"sid\\":1}}\\n"}'
    )
    assert count_frames_in_windows(paths, [boundary]) == {10: 1}


def test_a_stamp_without_microseconds_lands_on_the_right_side_of_a_bound(tmp_path: Path) -> None:
    early = window(10, at(4, 59, 59), at(5, 0, 0))
    late = window(20, at(6, 0, 0), at(6, 0, 1))
    paths = write_tape(
        tmp_path,
        [
            (at(4, 59, 59, 500_000), frame("orderbook_delta")),
            (at(5, 0, 0), frame("orderbook_delta", 2)),
            (at(6, 0, 0), frame("orderbook_snapshot", 1)),
            (at(6, 0, 0, 500_000), frame("orderbook_delta", 2)),
        ],
    )

    lines = gzip.decompress(paths[0].read_bytes()).splitlines()
    assert json.loads(lines[1])["received_at"] == "2026-07-30T05:00:00+00:00"
    assert count_frames_in_windows(paths, [early, late]) == {10: 1, 20: 1}


def test_several_days_are_scanned_into_one_tally(tmp_path: Path) -> None:
    def next_day(second: int, microsecond: int = 0) -> datetime:
        return datetime(2026, 7, 31, 4, 0, second, microsecond, tzinfo=UTC)

    first = window(10, at(3, 1, 1), at(3, 1, 3))
    second = window(20, next_day(0), next_day(2))
    paths = write_tape(
        tmp_path,
        [
            (at(3, 1, 2), frame("orderbook_delta")),
            (next_day(1), frame("orderbook_delta", 2)),
            (next_day(1, 500_000), frame("orderbook_delta", 3)),
        ],
    )

    counts = count_frames_in_windows(paths, [first, second])
    check = check_tape_counts([first, second], counts)

    assert [path.name for path in paths] == ["2026-07-30.jsonl.gz", "2026-07-31.jsonl.gz"]
    assert counts == {10: 1, 20: 2}
    assert (check.sampled, check.agreed, check.wider, check.extra_frames) == (2, 1, 1, 1)


def test_only_a_snapshot_carrying_neither_side_reads_as_a_clear(tmp_path: Path) -> None:
    paths = write_tape(
        tmp_path,
        [
            (at(3, 1, 0), frame("orderbook_snapshot", 1)),
            (at(3, 1, 1), levelled("orderbook_snapshot", 2, "no_dollars_fp")),
            (at(3, 1, 2), levelled("orderbook_snapshot", 3, "yes_dollars_fp")),
            (at(3, 1, 3), levelled("orderbook_snapshot", 4, "yes_dollars_fp", "no_dollars_fp")),
            (at(3, 1, 4), frame("trade", 5)),
            (at(3, 1, 5), frame("orderbook_delta", 6)),
            (at(3, 1, 6, 398_909), frame("orderbook_snapshot", 7, ticker=DEN)),
            (at(3, 1, 7), ack("orderbook_snapshot")),
        ],
    )

    assert list(read_clear_frames(paths[0])) == [
        ClearFrame(received_at=at(3, 1, 0), ticker=AUS),
        ClearFrame(received_at=at(3, 1, 6, 398_909), ticker=DEN),
    ]


def test_a_tape_with_no_clear_yields_nothing(tmp_path: Path) -> None:
    paths = write_tape(
        tmp_path,
        [
            (at(3, 1, 0), levelled("orderbook_snapshot", 1, "yes_dollars_fp", "no_dollars_fp")),
            (at(3, 1, 1), frame("orderbook_delta", 2)),
            (at(3, 1, 2), frame("trade", 3)),
        ],
    )

    assert list(read_clear_frames(paths[0])) == []


def test_the_verdict_names_every_falsifying_window() -> None:
    windows = [
        window(10, at(3, 1, 1), at(3, 1, 2)),
        window(20, at(3, 2, 1), at(3, 2, 2)),
        window(30, at(3, 3, 1), at(3, 3, 2)),
        window(40, at(3, 4, 1), at(3, 4, 2)),
    ]

    check = check_tape_counts(windows, {10: 1, 20: 3, 30: 0, 40: 5})

    assert (check.sampled, check.agreed, check.wider) == (4, 1, 2)
    assert (check.extra_frames, check.falsified) == (6, 1)
    assert check.falsifying == (windows[2],)
