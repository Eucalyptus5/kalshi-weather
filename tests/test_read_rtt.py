from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bot.lag.read_rtt import (
    ReadSample,
    append_sample,
    decode_sample,
    due_at,
    encode_sample,
    hourly_counts,
    load_samples,
    quantile_seconds,
    reanchor_start,
    resume_point,
    sample_interval_seconds,
    summarize,
)


UTC = timezone.utc
API_HOST = "api.elections.kalshi.com"
TICKER = "KXHIGHDEN-26AUG11-T95"
START = datetime(2026, 8, 11, 18, 0, tzinfo=UTC)


def _sample(
    sequence: int = 0,
    requested_at: datetime = START,
    elapsed_s: float = 0.25,
    outcome: str = "ok",
    status_code: int | None = 200,
) -> ReadSample:
    return ReadSample(
        sequence=sequence,
        requested_at=requested_at,
        elapsed_s=elapsed_s,
        ticker=TICKER,
        outcome=outcome,
        status_code=status_code,
        api_host=API_HOST,
        endpoint=f"GET /trade-api/v2/markets/{TICKER}/orderbook",
        source_host="kalshi-ws",
    )


@pytest.mark.parametrize(
    "sample",
    [
        _sample(requested_at=datetime(2026, 8, 11, 18, 0, 0, 123456, tzinfo=UTC)),
        _sample(outcome="transport", status_code=None, elapsed_s=30.0),
        _sample(outcome="http_status", status_code=503, elapsed_s=1.5),
    ],
)
def test_encode_decode_round_trip(sample: ReadSample) -> None:
    assert decode_sample(encode_sample(sample)) == sample


def test_encode_is_one_line_of_json() -> None:
    line = encode_sample(_sample())
    assert "\n" not in line
    assert line.startswith("{") and line.endswith("}")


def test_append_and_load_preserve_order(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    first = _sample(sequence=0)
    second = _sample(sequence=1, requested_at=START + timedelta(seconds=432))
    append_sample(path, first)
    append_sample(path, second)
    assert load_samples(path) == [first, second]


def test_load_ignores_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    sample = _sample()
    path.write_text(f"\n{encode_sample(sample)}\n\n")
    assert load_samples(path) == [sample]


def test_sample_interval_seconds_golden() -> None:
    assert sample_interval_seconds(200, 86400.0) == 432.0


@pytest.mark.parametrize(("target", "span"), [(0, 86400.0), (-1, 86400.0), (200, 0.0)])
def test_sample_interval_seconds_rejects_degenerate_inputs(target: int, span: float) -> None:
    with pytest.raises(ValueError):
        sample_interval_seconds(target, span)


def test_due_at_is_anchored_on_start() -> None:
    assert due_at(START, 0, 432.0) == START
    assert due_at(START, 200, 432.0) == START + timedelta(seconds=86400)


def test_reanchor_start_keeps_a_schedule_that_is_on_time() -> None:
    now = START + timedelta(seconds=432 * 5 + 100)
    assert reanchor_start(START, 5, 432.0, now) == START


def test_reanchor_start_skips_missed_slots() -> None:
    now = START + timedelta(hours=3)
    moved = reanchor_start(START, 5, 432.0, now)
    assert due_at(moved, 5, 432.0) == now
    assert due_at(moved, 6, 432.0) == now + timedelta(seconds=432)


def test_resume_point_on_empty_samples() -> None:
    assert resume_point([], START) == (START, 0)


def test_resume_point_continues_the_original_schedule() -> None:
    samples = [
        _sample(sequence=0, requested_at=START),
        _sample(sequence=1, requested_at=START + timedelta(seconds=432)),
    ]
    assert resume_point(samples, START + timedelta(hours=3)) == (START, 2)


def _elapsed_ladder() -> list[ReadSample]:
    return [
        _sample(sequence=i, elapsed_s=round(0.1 * (i + 1), 1), requested_at=START)
        for i in range(10)
    ]


@pytest.mark.parametrize(("q", "expected"), [(0.5, 0.5), (0.9, 0.9), (1.0, 1.0), (0.01, 0.1)])
def test_quantile_seconds_nearest_rank(q: float, expected: float) -> None:
    assert quantile_seconds(_elapsed_ladder(), q) == pytest.approx(expected)


def test_quantile_seconds_ignores_failed_reads() -> None:
    samples = _elapsed_ladder() + [
        _sample(sequence=10, elapsed_s=99.0, outcome="transport", status_code=None),
        _sample(sequence=11, elapsed_s=98.0, outcome="http_status", status_code=503),
    ]
    assert quantile_seconds(samples, 0.9) == pytest.approx(0.9)


def test_quantile_seconds_without_successful_reads() -> None:
    samples = [_sample(outcome="transport", status_code=None)]
    assert quantile_seconds(samples, 0.9) is None


@pytest.mark.parametrize("q", [0.0, -0.1, 1.5])
def test_quantile_seconds_rejects_out_of_range(q: float) -> None:
    with pytest.raises(ValueError):
        quantile_seconds(_elapsed_ladder(), q)


def test_hourly_counts_spread() -> None:
    samples = [
        _sample(sequence=0, requested_at=datetime(2026, 8, 11, 18, 5, tzinfo=UTC)),
        _sample(sequence=1, requested_at=datetime(2026, 8, 11, 18, 55, tzinfo=UTC)),
        _sample(sequence=2, requested_at=datetime(2026, 8, 11, 21, 0, tzinfo=UTC)),
        _sample(sequence=3, requested_at=datetime(2026, 8, 12, 3, 0, tzinfo=UTC)),
    ]
    assert hourly_counts(samples) == {3: 1, 18: 2, 21: 1}


def test_summarize_mixed_outcomes() -> None:
    samples = _elapsed_ladder() + [
        _sample(sequence=10, elapsed_s=30.0, outcome="transport", status_code=None),
    ]
    summary = summarize(samples)
    assert summary.n_total == 11
    assert summary.n_ok == 10
    assert summary.n_error == 1
    assert summary.p50_s == pytest.approx(0.5)
    assert summary.p90_s == pytest.approx(0.9)
    assert summary.min_s == pytest.approx(0.1)
    assert summary.max_s == pytest.approx(1.0)
    assert summary.first_at == START
    assert summary.last_at == START
    assert summary.hourly == {18: 11}


def test_summarize_empty() -> None:
    summary = summarize([])
    assert summary.n_total == 0
    assert summary.n_ok == 0
    assert summary.n_error == 0
    assert summary.p50_s is None
    assert summary.p90_s is None
    assert summary.min_s is None
    assert summary.max_s is None
    assert summary.first_at is None
    assert summary.last_at is None
    assert summary.hourly == {}
