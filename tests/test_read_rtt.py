from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bot.lag.read_rtt import (
    FloorSource,
    InadequateSamples,
    ReadSample,
    append_sample,
    check_adequacy,
    decode_sample,
    derive_latency_floor,
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


def _round_robin(elapsed: list[float], outcome: str = "ok") -> list[ReadSample]:
    return [
        _sample(
            sequence=i,
            requested_at=START.replace(hour=i % 24) + timedelta(days=i // 24),
            elapsed_s=e,
            outcome=outcome,
            status_code=200 if outcome == "ok" else None,
        )
        for i, e in enumerate(elapsed)
    ]


def _by_hour(
    counts: dict[int, int], elapsed_s: float = 0.2, outcome: str = "ok"
) -> list[ReadSample]:
    samples: list[ReadSample] = []
    for hour, n in counts.items():
        for k in range(n):
            samples.append(
                _sample(
                    sequence=len(samples),
                    requested_at=START.replace(hour=hour) + timedelta(seconds=k),
                    elapsed_s=elapsed_s,
                    outcome=outcome,
                    status_code=200 if outcome == "ok" else None,
                )
            )
    return samples


def test_check_adequacy_accepts_a_full_day_of_uniform_samples() -> None:
    adequacy = check_adequacy(_round_robin([0.2] * 216))
    assert adequacy.adequate
    assert adequacy.n_usable == 216
    assert adequacy.failures == ()
    assert adequacy.missing_hours == ()
    assert adequacy.overweight_hours == ()
    assert adequacy.hourly == {hour: 9 for hour in range(24)}


def test_derive_floor_is_the_p90_of_usable_elapsed() -> None:
    floor = derive_latency_floor(
        _round_robin([round(0.01 * (i + 1), 2) for i in range(240)]), FloorSource.SIGNED_READ
    )
    assert floor.floor_s == pytest.approx(2.16)
    assert floor.n_usable == 240
    assert floor.source is FloorSource.SIGNED_READ


def test_persist_threshold_holds_at_ten_seconds_for_a_fast_floor() -> None:
    floor = derive_latency_floor(_round_robin([0.1] * 240), FloorSource.SIGNED_READ)
    assert floor.floor_s == pytest.approx(0.1)
    assert floor.t_persist_s == pytest.approx(10.0)


def test_persist_threshold_triples_a_slow_floor() -> None:
    floor = derive_latency_floor(_round_robin([5.0] * 240), FloorSource.SIGNED_READ)
    assert floor.floor_s == pytest.approx(5.0)
    assert floor.t_persist_s == pytest.approx(15.0)


def test_both_floor_sources_derive_the_same_threshold() -> None:
    samples = _round_robin([4.0] * 240)
    demo = derive_latency_floor(samples, FloorSource.DEMO_ORDER)
    read = derive_latency_floor(samples, FloorSource.SIGNED_READ)
    assert demo.source is FloorSource.DEMO_ORDER
    assert read.source is FloorSource.SIGNED_READ
    assert demo.floor_s == pytest.approx(read.floor_s)
    assert demo.t_persist_s == pytest.approx(12.0)
    assert read.t_persist_s == pytest.approx(12.0)


def test_one_sample_short_of_the_minimum_refuses() -> None:
    samples = _round_robin([0.2] * 199)
    adequacy = check_adequacy(samples)
    assert not adequacy.adequate
    assert adequacy.n_usable == 199
    assert adequacy.missing_hours == ()
    assert adequacy.overweight_hours == ()
    assert len(adequacy.failures) == 1
    with pytest.raises(InadequateSamples) as excinfo:
        derive_latency_floor(samples, FloorSource.SIGNED_READ)
    assert excinfo.value.adequacy.n_usable == 199


def test_bunched_samples_refuse_despite_clearing_the_count() -> None:
    samples = _by_hour({9: 80, 10: 80, 11: 80})
    adequacy = check_adequacy(samples)
    assert not adequacy.adequate
    assert adequacy.n_usable == 240
    assert adequacy.overweight_hours == (9, 10, 11)
    assert len(adequacy.missing_hours) == 21
    with pytest.raises(InadequateSamples):
        derive_latency_floor(samples, FloorSource.SIGNED_READ)


def test_a_missing_hour_refuses_on_its_own() -> None:
    samples = _by_hour({hour: 10 for hour in range(24) if hour != 7})
    adequacy = check_adequacy(samples)
    assert not adequacy.adequate
    assert adequacy.n_usable == 230
    assert adequacy.missing_hours == (7,)
    assert adequacy.overweight_hours == ()
    assert len(adequacy.failures) == 1


@pytest.mark.parametrize(("heavy_hour_n", "adequate"), [(16, True), (17, False)])
def test_hour_concentration_binds_at_twice_the_uniform_share(
    heavy_hour_n: int, adequate: bool
) -> None:
    counts = {0: heavy_hour_n} | {hour: 8 for hour in range(1, 24)}
    counts[23] += 200 - sum(counts.values())
    adequacy = check_adequacy(_by_hour(counts))
    assert adequacy.n_usable == 200
    assert adequacy.adequate is adequate
    assert adequacy.overweight_hours == (() if adequate else (0,))


def test_failed_reads_count_toward_neither_the_minimum_nor_the_spread() -> None:
    samples = _round_robin([0.2] * 199) + _by_hour({7: 40}, elapsed_s=99.0, outcome="transport")
    adequacy = check_adequacy(samples)
    assert adequacy.n_usable == 199
    assert adequacy.hourly[7] == 8
    assert not adequacy.adequate


def test_failed_reads_do_not_move_the_floor() -> None:
    usable = _round_robin([round(0.01 * (i + 1), 2) for i in range(240)])
    with_errors = usable + _by_hour({3: 5}, elapsed_s=120.0, outcome="http_status")
    assert derive_latency_floor(with_errors, FloorSource.SIGNED_READ).floor_s == pytest.approx(2.16)
