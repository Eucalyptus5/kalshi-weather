from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, localcontext
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bot.lag.forecast_sample import (
    DEPTH_WINDOW,
    F4_LEADS,
    F4_SERIES,
    PRE_WEEKLY_ERA,
    SPLIT_BOUNDARY,
    SampleLeg,
    SamplePlan,
    TickTape,
    build_sample,
    era_caps,
    era_index,
    era_of,
    read_sample_freeze,
    read_sample_plan,
    read_tick_tape,
    sidecar_path,
    write_sample_freeze,
)
from bot.main import STATIONS
from bot.markets.parser import parse_ticker, resolve_event_kinds
from scripts.freeze_f4_sample import (
    DEFAULT_OUT,
    ERA_REPORT,
    FREEZE_NAME,
    INGEST_REPORT,
    MARKETS,
    SAMPLE_PLAN,
    TICKS,
)


pytestmark = pytest.mark.skipif(
    not SAMPLE_PLAN.exists(),
    reason="frozen study artifacts under data/ are not committed",
)

UTC = timezone.utc
AMBIENT_PRECISIONS = (20, 28, 50)

FROZEN_SAMPLE = DEFAULT_OUT / FREEZE_NAME
FROZEN_LEGS = 20782
FROZEN_SHA256 = "c5bfac9a4f9267ce1887c9f25119fadff2f12815521532163030ce0bcebb0e2b"

CAP_MINUTES = {
    "0010": Decimal("575.0"),
    "0016": Decimal("171.0"),
    "0017": Decimal("109.1"),
    "0018": Decimal("62.7"),
    "0019": Decimal("61.6"),
    "0004": Decimal("65.9"),
    "0005": Decimal("56.4"),
    "0006": Decimal("50.8"),
    "0007": Decimal("64.5"),
    "0008": Decimal("47.6"),
    "0009": Decimal("42.9"),
}

LADDER_DAY = date(2024, 10, 24)
LADDER_CLOSE = datetime(2024, 10, 25, 3, 59, tzinfo=UTC)
LADDER_AS_OF = LADDER_CLOSE - timedelta(hours=24)
BOTTOM = "KXHIGHMIA-24OCT24-T80"
TOP = "KXHIGHMIA-24OCT24-T83"
RUNGS = (BOTTOM, "KXHIGHMIA-24OCT24-B81.5", "KXHIGHMIA-24OCT24-B82.5", TOP)

POISON_STRIKE = 999
OFF_SERIES = "KXHIGHTATL-24OCT24-T80"
STALENESS_MICROS = 40601
PINNED_STALENESS = "0.0006766833333333333333333333333"

_MARKET_SCHEMA = pa.schema(
    [
        pa.field("ticker", pa.string()),
        pa.field("series_ticker", pa.string()),
        pa.field("result", pa.string()),
        pa.field("close_time", pa.timestamp("us", tz="UTC")),
        pa.field("floor_strike", pa.int32()),
    ]
)

_TICK_SCHEMA = pa.schema(
    [
        pa.field("ticker", pa.string()),
        pa.field("created_time", pa.timestamp("us", tz="UTC")),
        pa.field("yes_price", pa.decimal128(10, 2)),
        pa.field("count", pa.int64()),
    ]
)


@pytest.fixture(scope="module")
def plan() -> SamplePlan:
    return read_sample_plan(SAMPLE_PLAN)


@pytest.fixture(scope="module")
def caps() -> Mapping[str, timedelta | None]:
    return era_caps(ERA_REPORT)


@pytest.fixture(scope="module")
def eras() -> Sequence[tuple[str, datetime]]:
    return era_index(INGEST_REPORT)


@pytest.fixture(scope="module")
def tape() -> TickTape:
    return read_tick_tape(TICKS)


@pytest.fixture(scope="module")
def legs(
    plan: SamplePlan,
    caps: Mapping[str, timedelta | None],
    eras: Sequence[tuple[str, datetime]],
    tape: TickTape,
) -> Mapping[int, list[SampleLeg]]:
    return {lead: build_sample(MARKETS, tape, plan, caps, eras, lead) for lead in F4_LEADS}


@pytest.fixture(scope="module")
def uncapped(
    plan: SamplePlan,
    caps: Mapping[str, timedelta | None],
    eras: Sequence[tuple[str, datetime]],
    tape: TickTape,
) -> Mapping[int, list[SampleLeg]]:
    unbounded = dict.fromkeys(caps, None)
    return {lead: build_sample(MARKETS, tape, plan, unbounded, eras, lead) for lead in F4_LEADS}


def day_split(rows: Sequence[SampleLeg]) -> tuple[int, int, int]:
    days = {leg.event_date for leg in rows}
    discovery = {day for day in days if day < SPLIT_BOUNDARY}
    return len(days), len(discovery), len(days - discovery)


def write_markets(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    pq.write_table(pa.Table.from_pylist(list(rows), schema=_MARKET_SCHEMA), path)
    return path


def write_ticks(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    pq.write_table(pa.Table.from_pylist(list(rows), schema=_TICK_SCHEMA), path)
    return path


def market_row(ticker: str, close: datetime, result: str = "no") -> dict[str, object]:
    return {
        "ticker": ticker,
        "series_ticker": ticker.split("-", 1)[0],
        "result": result,
        "close_time": close,
        "floor_strike": POISON_STRIKE,
    }


def tick_row(ticker: str, created: datetime, price: str, count: int) -> dict[str, object]:
    return {
        "ticker": ticker,
        "created_time": created,
        "yes_price": Decimal(price),
        "count": count,
    }


def one_day_plan(day: date) -> SamplePlan:
    return SamplePlan(
        candidate_days=(day,),
        sampled_days=(),
        unread_days=(day,),
        discovery_days=(day,),
        holdout_days=(),
    )


def ladder_sample(
    tmp_path: Path,
    caps: Mapping[str, timedelta | None],
    eras: Sequence[tuple[str, datetime]],
    ticks: Sequence[Mapping[str, object]],
) -> list[SampleLeg]:
    markets = write_markets(
        tmp_path / "markets.parquet", [market_row(rung, LADDER_CLOSE) for rung in RUNGS]
    )
    tape = read_tick_tape(write_ticks(tmp_path / "ticks.parquet", ticks))
    return build_sample(markets, tape, one_day_plan(LADDER_DAY), caps, eras, 24)


def test_sample_plan_counts(plan: SamplePlan) -> None:
    assert len(plan.candidate_days) == 458
    assert len(plan.sampled_days) == 115
    assert len(plan.unread_days) == 343
    assert plan.unread_days[0] == date(2024, 10, 26)
    assert plan.unread_days[-1] == date(2026, 1, 27)
    assert len(plan.discovery_days) == 220
    assert plan.discovery_days[0] == date(2024, 10, 26)
    assert plan.discovery_days[-1] == date(2025, 8, 14)
    assert len(plan.holdout_days) == 123
    assert plan.holdout_days[0] == date(2025, 8, 15)
    assert plan.holdout_days[-1] == date(2026, 1, 27)
    assert SPLIT_BOUNDARY == date(2025, 8, 15)


def test_sample_plan_stride_holds(plan: SamplePlan) -> None:
    assert plan.candidate_days[::4] == plan.sampled_days


def test_sample_plan_rejects_a_slid_stride(tmp_path: Path, plan: SamplePlan) -> None:
    days = [day.isoformat() for day in plan.candidate_days]
    path = tmp_path / "sample_plan.json"
    path.write_text(json.dumps({"candidate_days": days, "sampled_days": days[::5]}))
    with pytest.raises(ValueError, match="candidate_days"):
        read_sample_plan(path)


def test_sample_plan_holes(plan: SamplePlan) -> None:
    unread = set(plan.unread_days)
    assert date(2025, 11, 23) not in unread
    assert date(2025, 11, 24) not in unread
    assert date(2025, 11, 22) in unread
    assert date(2025, 11, 25) in unread


def test_unread_days_hold_the_file_order(plan: SamplePlan) -> None:
    positions = [plan.candidate_days.index(day) for day in plan.unread_days]

    assert positions == sorted(positions)
    assert len(set(positions)) == len(plan.unread_days)
    assert len(plan.unread_days) + len(plan.sampled_days) == len(plan.candidate_days)


def test_unread_days_follow_the_file_and_not_the_calendar(tmp_path: Path) -> None:
    candidate = [
        "2025-03-02",
        "2025-03-01",
        "2025-03-05",
        "2025-03-04",
        "2025-03-03",
        "2025-03-02",
        "2025-03-06",
        "2025-03-08",
    ]
    path = tmp_path / "sample_plan.json"
    path.write_text(json.dumps({"candidate_days": candidate, "sampled_days": candidate[::4]}))
    plan = read_sample_plan(path)

    assert plan.unread_days == (
        date(2025, 3, 1),
        date(2025, 3, 5),
        date(2025, 3, 4),
        date(2025, 3, 6),
        date(2025, 3, 8),
    )
    assert plan.discovery_days == plan.unread_days
    assert plan.holdout_days == ()


def test_era_caps_are_read_never_derived(caps: Mapping[str, timedelta | None]) -> None:
    minutes = {
        era: None
        if cap is None
        else Decimal(cap // timedelta(microseconds=1)) / Decimal(60_000_000)
        for era, cap in caps.items()
    }
    assert minutes == CAP_MINUTES


def test_a_null_cap_is_unbounded(tmp_path: Path) -> None:
    path = tmp_path / "era_report.json"
    path.write_text(json.dumps({"eras": {"0009": {"cap_minutes": None}, "0010": {"n_markets": 0}}}))
    assert era_caps(path) == {"0009": None}


def test_a_zero_market_era_carries_no_cap(tmp_path: Path) -> None:
    path = tmp_path / "era_report.json"
    path.write_text(json.dumps({"eras": {"0009": {"n_markets": 0}, "0010": {"cap_minutes": 1.5}}}))
    assert era_caps(path) == {"0010": timedelta(minutes=1, seconds=30)}


def test_era_of_reproduces_the_weekly_order(eras: Sequence[tuple[str, datetime]]) -> None:
    assert [shard for shard, _ in eras] == [
        "0016",
        "0017",
        "0018",
        "0019",
        "0004",
        "0005",
        "0006",
        "0007",
        "0008",
        "0009",
    ]
    first = eras[0][1]
    assert era_of(first - timedelta(microseconds=1), eras) == PRE_WEEKLY_ERA
    assert era_of(first, eras) == "0016"
    assert era_of(eras[4][1] - timedelta(microseconds=1), eras) == "0019"
    assert era_of(eras[4][1], eras) == "0004"
    assert era_of(eras[-1][1] + timedelta(days=30), eras) == "0009"


def test_legs_at_the_24h_lead(legs: Mapping[int, list[SampleLeg]]) -> None:
    assert len(legs[24]) == 12431
    assert day_split(legs[24]) == (342, 220, 122)


def test_legs_at_the_36h_lead(legs: Mapping[int, list[SampleLeg]]) -> None:
    assert len(legs[36]) == 8351
    assert day_split(legs[36]) == (336, 217, 119)


def test_the_one_unread_day_carrying_no_leg(
    plan: SamplePlan, legs: Mapping[int, list[SampleLeg]]
) -> None:
    scored = {leg.event_date for leg in legs[24]}
    assert sorted(set(plan.unread_days) - scored) == [date(2025, 11, 25)]


def test_the_cap_costs_legs_and_no_days(
    legs: Mapping[int, list[SampleLeg]], uncapped: Mapping[int, list[SampleLeg]]
) -> None:
    assert len(uncapped[24]) == 12821
    assert len(uncapped[36]) == 8657
    assert day_split(uncapped[24]) == day_split(legs[24])
    assert day_split(uncapped[36]) == day_split(legs[36])


def test_leg_fields(
    legs: Mapping[int, list[SampleLeg]], caps: Mapping[str, timedelta | None]
) -> None:
    for lead, rows in legs.items():
        for leg in rows:
            station = STATIONS[leg.series]
            assert leg.series in F4_SERIES
            assert leg.station == station.station
            assert leg.timezone == station.timezone
            assert leg.lead_hours == lead
            assert leg.close_time.tzinfo is not None
            assert leg.as_of == leg.close_time - timedelta(hours=lead)
            assert leg.as_of.utcoffset() == timedelta(0)
            assert Decimal(0) <= leg.entry_price <= Decimal(1)
            assert leg.result in ("yes", "no")
            assert leg.kind in ("above", "below", "bracket")
            expected = "discovery" if leg.event_date < SPLIT_BOUNDARY else "holdout"
            assert leg.split == expected
            cap = caps[leg.era]
            assert Decimal(0) <= leg.staleness_minutes
            assert leg.staleness_minutes * 60 <= Decimal(cap // timedelta(seconds=1))
            assert leg.trailing_contracts >= Decimal(0)
            if leg.staleness_minutes < Decimal(DEPTH_WINDOW // timedelta(minutes=1)):
                assert leg.trailing_prints >= 1


def test_strikes_come_from_the_ticker(legs: Mapping[int, list[SampleLeg]]) -> None:
    bracket = parse_ticker("KXHIGHMIA-24OCT24-B86.5")
    assert bracket.strikes == (Decimal(86), Decimal(87))
    assert bracket.kind == "bracket"
    above = parse_ticker("KXHIGHMIA-24OCT24-T87")
    assert above.strikes == (Decimal(87),)
    assert above.kind == "above"

    table = pq.read_table(MARKETS)
    assert table.num_rows == 18710
    assert "cap_strike" not in table.schema.names
    assert set(table.column("floor_strike").to_pylist()) == {None}

    for leg in legs[24]:
        strikes = parse_ticker(leg.ticker).strikes
        assert leg.strike_lo == strikes[0]
        assert leg.strike_hi == (strikes[1] if len(strikes) == 2 else None)
        assert leg.strike_lo != Decimal(POISON_STRIKE)


def test_a_poisoned_parquet_strike_is_never_read(
    tmp_path: Path, caps: Mapping[str, timedelta | None], eras: Sequence[tuple[str, datetime]]
) -> None:
    ticks = [tick_row(rung, LADDER_AS_OF - timedelta(minutes=5), "0.40", 3) for rung in RUNGS]
    built = {leg.ticker: leg for leg in ladder_sample(tmp_path, caps, eras, ticks)}
    assert built[BOTTOM].strike_lo == Decimal(80)
    assert built["KXHIGHMIA-24OCT24-B81.5"].strike_lo == Decimal(81)
    assert built["KXHIGHMIA-24OCT24-B81.5"].strike_hi == Decimal(82)
    assert all(leg.strike_lo != Decimal(POISON_STRIKE) for leg in built.values())


def test_tagging_runs_over_the_full_ladder(
    tmp_path: Path, caps: Mapping[str, timedelta | None], eras: Sequence[tuple[str, datetime]]
) -> None:
    ticks = [
        tick_row(rung, LADDER_AS_OF - timedelta(minutes=5), "0.40", 3)
        for rung in RUNGS
        if rung != TOP
    ]
    ticks.append(tick_row(TOP, LADDER_AS_OF - timedelta(minutes=600), "0.05", 1))
    built = {leg.ticker: leg for leg in ladder_sample(tmp_path, caps, eras, ticks)}

    assert TOP not in built
    assert built[BOTTOM].kind == "below"
    assert built[BOTTOM].era == PRE_WEEKLY_ERA
    survivors = resolve_event_kinds([parse_ticker(ticker) for ticker in sorted(built)])
    assert [row.kind for row in survivors if row.raw == BOTTOM] == ["above"]


def test_trailing_depth_window(
    tmp_path: Path, caps: Mapping[str, timedelta | None], eras: Sequence[tuple[str, datetime]]
) -> None:
    edge = LADDER_AS_OF - timedelta(hours=6)
    ticks = [
        tick_row(BOTTOM, edge, "0.10", 100),
        tick_row(BOTTOM, edge + timedelta(microseconds=1), "0.20", 7),
        tick_row(BOTTOM, LADDER_AS_OF - timedelta(minutes=30), "0.30", 11),
        tick_row(BOTTOM, LADDER_AS_OF, "0.44", 5),
        tick_row(BOTTOM, LADDER_AS_OF + timedelta(microseconds=1), "0.99", 900),
    ]
    built = {leg.ticker: leg for leg in ladder_sample(tmp_path, caps, eras, ticks)}
    leg = built[BOTTOM]
    assert DEPTH_WINDOW == timedelta(hours=6)
    assert leg.entry_price == Decimal("0.44")
    assert leg.staleness_minutes == Decimal(0)
    assert leg.trailing_prints == 3
    assert leg.trailing_contracts == Decimal(23)
    assert isinstance(leg.trailing_contracts, Decimal)
    assert isinstance(leg.entry_price, Decimal)
    assert isinstance(leg.staleness_minutes, Decimal)


def test_tagging_the_survivors_disagrees_on_233_ladders(
    legs: Mapping[int, list[SampleLeg]],
) -> None:
    ladders: dict[tuple[str, date], list[SampleLeg]] = {}
    for leg in legs[24]:
        ladders.setdefault((leg.series, leg.event_date), []).append(leg)

    disagreed: set[tuple[str, date]] = set()
    for key, rows in ladders.items():
        tagged = resolve_event_kinds([parse_ticker(leg.ticker) for leg in rows])
        if any(leg.kind != row.kind for leg, row in zip(rows, tagged)):
            disagreed.add(key)

    days = {event_date for _, event_date in disagreed}
    assert len(disagreed) == 233
    assert len(days) == 152
    assert len([day for day in days if day < SPLIT_BOUNDARY]) == 82


def test_freeze_round_trips(tmp_path: Path, legs: Mapping[int, list[SampleLeg]]) -> None:
    picked = [next(leg for leg in legs[24] if leg.kind == kind) for kind in ("above", "below")]
    picked += [leg for leg in legs[24][:20] if leg.kind == "bracket"]
    picked += legs[36][:5]
    path = tmp_path / "sample.jsonl"
    digest = write_sample_freeze(picked, path)

    assert sidecar_path(path).exists()
    assert json.loads(sidecar_path(path).read_text())["sha256"] == digest
    assert read_sample_freeze(path) == picked


def test_freeze_refuses_to_overwrite(tmp_path: Path, legs: Mapping[int, list[SampleLeg]]) -> None:
    path = tmp_path / "sample.jsonl"
    write_sample_freeze(legs[24][:3], path)
    with pytest.raises(FileExistsError):
        write_sample_freeze(legs[24][:3], path)


def test_freeze_refuses_a_stale_sidecar(
    tmp_path: Path, legs: Mapping[int, list[SampleLeg]]
) -> None:
    path = tmp_path / "sample.jsonl"
    write_sample_freeze(legs[24][:3], path)
    first = json.loads(path.read_text().splitlines()[0])
    prints = first["trailing_prints"]
    path.write_text(
        path.read_text().replace(
            f'"trailing_prints": {prints}', f'"trailing_prints": {prints + 1}', 1
        )
    )

    assert json.loads(path.read_text().splitlines()[0])["trailing_prints"] == prints + 1
    with pytest.raises(ValueError, match="does not match the sha256 it carries"):
        read_sample_freeze(path)


def test_the_cap_boundary_keeps_a_tick_at_the_cap(
    tmp_path: Path, caps: Mapping[str, timedelta | None], eras: Sequence[tuple[str, datetime]]
) -> None:
    cap = caps[PRE_WEEKLY_ERA]
    at_cap = ladder_sample(tmp_path, caps, eras, [tick_row(BOTTOM, LADDER_AS_OF - cap, "0.40", 3)])
    past_cap = ladder_sample(
        tmp_path,
        caps,
        eras,
        [tick_row(BOTTOM, LADDER_AS_OF - cap - timedelta(microseconds=1), "0.40", 3)],
    )

    assert [leg.ticker for leg in at_cap] == [BOTTOM]
    assert at_cap[0].staleness_minutes == Decimal(575)
    assert past_cap == []


def test_the_eligibility_screen_drops_the_unscored_rungs(
    tmp_path: Path, caps: Mapping[str, timedelta | None], eras: Sequence[tuple[str, datetime]]
) -> None:
    rows = [
        market_row(BOTTOM, LADDER_CLOSE),
        {**market_row("KXHIGHMIA-24OCT24-B81.5", LADDER_CLOSE), "result": ""},
        {**market_row("KXHIGHMIA-24OCT24-B82.5", LADDER_CLOSE), "close_time": None},
        market_row(OFF_SERIES, LADDER_CLOSE),
    ]
    markets = write_markets(tmp_path / "markets.parquet", rows)
    ticks = [
        tick_row(str(row["ticker"]), LADDER_AS_OF - timedelta(minutes=5), "0.40", 3) for row in rows
    ]
    tape = read_tick_tape(write_ticks(tmp_path / "ticks.parquet", ticks))
    built = build_sample(markets, tape, one_day_plan(LADDER_DAY), caps, eras, 24)

    assert [leg.ticker for leg in built] == [BOTTOM]


@pytest.mark.parametrize("prec", AMBIENT_PRECISIONS)
def test_the_staleness_reads_the_same_figure_at_every_ambient_precision(
    prec: int,
    tmp_path: Path,
    caps: Mapping[str, timedelta | None],
    eras: Sequence[tuple[str, datetime]],
) -> None:
    ticks = [tick_row(BOTTOM, LADDER_AS_OF - timedelta(microseconds=STALENESS_MICROS), "0.40", 3)]
    with localcontext(prec=prec):
        built = ladder_sample(tmp_path, caps, eras, ticks)

    assert str(built[0].staleness_minutes) == PINNED_STALENESS


def test_the_published_freeze_matches_its_recorded_digest(
    legs: Mapping[int, list[SampleLeg]],
) -> None:
    frozen = read_sample_freeze(FROZEN_SAMPLE)

    assert hashlib.sha256(FROZEN_SAMPLE.read_bytes()).hexdigest() == FROZEN_SHA256
    assert len(frozen) == FROZEN_LEGS
    assert frozen == legs[24] + legs[36]
