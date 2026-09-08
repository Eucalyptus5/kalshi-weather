import math
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, localcontext
from pathlib import Path

import pytest
import pytz

from bot.backtest.depth_table import lead_bucket_for
from bot.backtest.historical_open_meteo import CALIBRATION_PATH, SpreadCalibration, day_end_utc
from bot.lag.forecast_classes import (
    CLASS_A,
    CLASS_B,
    LEAD_ANCHORED_COMPOSITE,
    LST_FULL,
    LST_FULL_LESS_LAST_HOUR,
    MODEL_RUN,
    UTC_12Z_00Z,
    ClassRecord,
    read_class_freeze,
)
from bot.lag.forecast_probability import (
    EXTERNAL_CALIBRATION,
    NATIVE_XND,
    SIGMA_MULTIPLIERS,
    baseline_brier,
    brier_skill,
    class_brier,
    class_cdf,
    decision_lead,
    effective_sigma,
    event_ladder,
    event_probability,
    ladder_sum,
    probabilities,
    probability_for,
    sensitivity_band,
    settles_yes,
    sigma_for,
    sigma_source_tally,
)
from bot.lag.forecast_sample import F4_SERIES, SampleLeg, read_sample_freeze
from bot.main import STATIONS
from bot.markets.parser import parse_ticker
from bot.validation.reconcile import settle_bracket
from bot.validation.scoring import brier_score


F4_INPUTS = Path("data/tape_studies/f4_inputs")
SAMPLE = F4_INPUTS / "sample.jsonl"
CLASS_A_FREEZE = F4_INPUTS / "class_a.jsonl"
CLASS_B_FREEZE = F4_INPUTS / "class_b.jsonl"

needs_corpus = pytest.mark.skipif(
    not (SAMPLE.exists() and CLASS_A_FREEZE.exists() and CLASS_B_FREEZE.exists()),
    reason="the frozen forecast corpus is not on this host",
)

CALIBRATION = SpreadCalibration.load(CALIBRATION_PATH)
BUCKET_SIGMA = Decimal(str(CALIBRATION.sigma_for(timedelta(hours=24, seconds=60))))

MIAMI_GEOMETRY = (
    "KXHIGHMIA-24OCT24-T80",
    "KXHIGHMIA-24OCT24-B80.5",
    "KXHIGHMIA-24OCT24-B82.5",
    "KXHIGHMIA-24OCT24-B84.5",
    "KXHIGHMIA-24OCT24-B86.5",
    "KXHIGHMIA-24OCT24-T87",
)
MIAMI_EVENT = "KXHIGHMIA-24OCT24"
GEOMETRY_HIGH = Decimal("86")
AMBIENT_PRECISIONS = (12, 20, 28, 50)
SUMMER = date(2025, 7, 15)
WINTER = date(2025, 1, 15)


def leg(
    *,
    ticker: str = "KXHIGHMIA-24OCT24-B86.5",
    series: str = "KXHIGHMIA",
    station: str = "KMIA",
    tz: str = "America/New_York",
    event_date: date = date(2024, 10, 24),
    split: str = "discovery",
    lead_hours: int = 24,
    kind: str = "bracket",
    strike_lo: Decimal = Decimal("86"),
    strike_hi: Decimal | None = Decimal("87"),
    entry_price: Decimal = Decimal("0.20"),
    result: str = "no",
) -> SampleLeg:
    close_time = pytz.timezone(tz).localize(datetime.combine(event_date, time(23, 59)))
    return SampleLeg(
        ticker=ticker,
        series=series,
        station=station,
        timezone=tz,
        event_date=event_date,
        split=split,
        lead_hours=lead_hours,
        close_time=close_time,
        as_of=close_time - timedelta(hours=lead_hours),
        entry_price=entry_price,
        staleness_minutes=Decimal("1"),
        era="0010",
        trailing_prints=4,
        trailing_contracts=Decimal("40"),
        strike_lo=strike_lo,
        strike_hi=strike_hi,
        kind=kind,
        result=result,
    )


def record(
    *,
    forecast_class: str = CLASS_B,
    member: str = "nbm_nbs",
    daily_high_f: Decimal = GEOMETRY_HIGH,
    native_sigma_f: Decimal | None = Decimal("2.0"),
    window_basis: str = UTC_12Z_00Z,
) -> ClassRecord:
    return ClassRecord(
        station="KMIA",
        event_date=date(2024, 10, 24),
        lead_hours=24,
        forecast_class=forecast_class,
        member=member,
        daily_high_f=daily_high_f,
        issue_time=datetime(2024, 10, 23, 12, tzinfo=timezone.utc),
        issue_rule=MODEL_RUN if forecast_class == CLASS_B else LEAD_ANCHORED_COMPOSITE,
        window_basis=window_basis,
        native_sigma_f=native_sigma_f,
        grid_latitude=None,
        grid_longitude=None,
        source_url="https://example.invalid/forecast",
    )


def test_the_continuity_corrected_interval_is_wider_than_the_half_open_one() -> None:
    cdf = class_cdf(GEOMETRY_HIGH, Decimal("0"))

    assert set(cdf.members.tolist()) == {86.0}
    assert cdf.prob_range(85.5, 87.5) > cdf.prob_range(86.0, 87.0)
    assert event_probability("bracket", Decimal("86"), Decimal("87"), cdf) == Decimal(
        "0.6246552600051549"
    )
    assert event_probability("bracket", Decimal("86"), Decimal("87"), cdf) != Decimal(
        str(cdf.prob_range(86.0, 87.0))
    )


def test_the_full_miami_ladder_sums_to_one() -> None:
    total = ladder_sum(
        event_ladder(MIAMI_GEOMETRY), class_cdf(GEOMETRY_HIGH, BUCKET_SIGMA), event=MIAMI_EVENT
    )

    assert total.rungs == 6
    assert total.event == MIAMI_EVENT
    assert total.total == Decimal("1.00000000000000000")
    assert abs(total.total - Decimal(1)) <= Decimal("1e-9")
    assert total.ok


def test_the_below_row_is_the_lowest_t_row_and_carries_its_strike_in_strike_lo() -> None:
    ladder = event_ladder(MIAMI_GEOMETRY)

    assert [(rung.ticker, rung.kind) for rung in ladder] == [
        ("KXHIGHMIA-24OCT24-B80.5", "bracket"),
        ("KXHIGHMIA-24OCT24-B82.5", "bracket"),
        ("KXHIGHMIA-24OCT24-B84.5", "bracket"),
        ("KXHIGHMIA-24OCT24-B86.5", "bracket"),
        ("KXHIGHMIA-24OCT24-T80", "below"),
        ("KXHIGHMIA-24OCT24-T87", "above"),
    ]
    below = next(rung for rung in ladder if rung.kind == "below")
    assert below.strike_lo == Decimal("80")
    assert below.strike_hi is None


def test_the_wrong_below_tail_overshoots_by_exactly_the_reclaimed_degree() -> None:
    cdf = class_cdf(GEOMETRY_HIGH, BUCKET_SIGMA)
    right = ladder_sum(event_ladder(MIAMI_GEOMETRY), cdf, event=MIAMI_EVENT).total

    wrong = right - event_probability("below", Decimal("80"), None, cdf)
    wrong += Decimal(str(cdf.prob_range(-math.inf, 80.5)))

    assert wrong == Decimal("1.03887686793347470")
    assert wrong > Decimal(1)
    assert abs((wrong - Decimal(1)) - Decimal(str(cdf.prob_range(79.5, 80.5)))) < Decimal("1e-12")


def test_a_ladder_missing_its_below_row_is_reported_not_renormalised() -> None:
    cdf = class_cdf(GEOMETRY_HIGH, BUCKET_SIGMA)
    full = event_ladder(MIAMI_GEOMETRY)
    without = tuple(rung for rung in full if rung.kind != "below")

    short = ladder_sum(without, cdf, event=MIAMI_EVENT)

    assert short.rungs == 5
    assert short.total == Decimal("0.92950714294037660")
    assert not short.ok
    assert short.event == MIAMI_EVENT
    assert ladder_sum(full, cdf, event=MIAMI_EVENT).total - short.total == event_probability(
        "below", Decimal("80"), None, cdf
    )


def test_a_ladder_assembled_from_survivors_is_not_the_object_the_check_runs_over() -> None:
    cdf = class_cdf(GEOMETRY_HIGH, BUCKET_SIGMA)
    survivors = tuple(t for t in MIAMI_GEOMETRY if t != "KXHIGHMIA-24OCT24-T87")

    assembled = event_ladder(survivors)

    assert {rung.kind for rung in assembled} == {"bracket", "above"}
    assert next(rung for rung in assembled if rung.kind == "above").ticker == (
        "KXHIGHMIA-24OCT24-T80"
    )
    assert assembled != event_ladder(MIAMI_GEOMETRY)[:5]
    assert ladder_sum(assembled, cdf, event=MIAMI_EVENT).total == Decimal("1.44904045655127270")
    assert not ladder_sum(assembled, cdf, event=MIAMI_EVENT).ok


def test_the_one_degree_kernel_floor_is_the_smoothing() -> None:
    cdf = class_cdf(GEOMETRY_HIGH, Decimal("0"))

    assert cdf.cdf(86.0) == 0.5
    assert cdf.cdf(83.0) == pytest.approx(0.0013498980316300922, rel=1e-15)
    assert effective_sigma(Decimal("0")) == Decimal(1)


@pytest.mark.parametrize("sigma", ["0", "2.0", "3.0148021921951225", "4.232898229269733"])
def test_the_effective_sigma_is_the_root_of_sigma_squared_plus_one(sigma: str) -> None:
    stated = Decimal(sigma)

    assert float(effective_sigma(stated)) == pytest.approx(
        math.sqrt(float(stated) ** 2 + 1), rel=1e-15
    )


def test_the_sensitivity_band_is_derived_from_the_effective_sigma_identity() -> None:
    band = sensitivity_band(BUCKET_SIGMA)

    assert effective_sigma(BUCKET_SIGMA) == Decimal("4.349416905673085866394477688")
    assert effective_sigma(BUCKET_SIGMA / 2) == Decimal("2.340802609114811773574974627")
    assert effective_sigma(BUCKET_SIGMA * 2) == Decimal("8.524653053199254071582341580")
    assert band[Decimal("0.5")] == Decimal("0.5381876835172151082472405429")
    assert band[Decimal("1")] == Decimal(1)
    assert band[Decimal("2")] == Decimal("1.959953078326492906240366607")

    early = sensitivity_band(Decimal("3.0148021921951225"))

    assert effective_sigma(Decimal("3.0148021921951225")) == Decimal(
        "3.176323701713116698798731371"
    )
    assert early[Decimal("0.5")] == Decimal("0.5695069364447214526482044378")
    assert early[Decimal("1")] == Decimal(1)
    assert early[Decimal("2")] == Decimal("1.924226441291510784606958017")


def test_the_sensitivity_at_one_is_the_stated_probability() -> None:
    row = probability_for(leg(), record(), CALIBRATION)

    assert tuple(row.sensitivity) == SIGMA_MULTIPLIERS
    assert row.sensitivity[Decimal("1")] == row.class_probability
    assert row.sensitivity[Decimal("0.5")] > row.class_probability
    assert row.sensitivity[Decimal("2")] < row.class_probability


def test_class_b_sigma_comes_from_the_native_xnd() -> None:
    row = probability_for(leg(), record(native_sigma_f=Decimal("2.0")), CALIBRATION)

    assert row.sigma_f == Decimal("2.0")
    assert row.sigma_source == NATIVE_XND
    assert row.effective_sigma_f == effective_sigma(Decimal("2.0"))
    assert row.members_n == 31
    assert row.daily_high_f == GEOMETRY_HIGH


def test_a_null_native_sigma_falls_to_the_external_calibration_with_its_own_tally() -> None:
    native = probability_for(leg(), record(native_sigma_f=Decimal("2.0")), CALIBRATION)
    fallen = probability_for(leg(), record(native_sigma_f=None), CALIBRATION)

    assert fallen.sigma_source == EXTERNAL_CALIBRATION
    assert fallen.sigma_f == BUCKET_SIGMA
    assert fallen.class_probability != native.class_probability

    tally = sigma_source_tally([native, fallen])

    assert len(tally) == 1
    assert tally[0].forecast_class == CLASS_B
    assert tally[0].native_xnd == 1
    assert tally[0].external_calibration == 1


def test_class_a_takes_the_external_calibration_for_every_record() -> None:
    row = probability_for(
        leg(),
        record(
            forecast_class=CLASS_A,
            member="ecmwf_ifs025",
            native_sigma_f=None,
            window_basis=LST_FULL_LESS_LAST_HOUR,
        ),
        CALIBRATION,
    )

    assert row.sigma_source == EXTERNAL_CALIBRATION
    assert row.sigma_f == BUCKET_SIGMA
    assert row.window_basis == LST_FULL_LESS_LAST_HOUR
    assert row.key == ("KXHIGHMIA-24OCT24-B86.5", 24, CLASS_A, "ecmwf_ifs025")


def test_the_seam_carries_the_leg_identity_m4_would_otherwise_rejoin_for() -> None:
    subject = leg(
        series="KXHIGHNY",
        station="KNYC",
        split="holdout",
        kind="below",
        strike_lo=Decimal("80"),
        strike_hi=None,
        result="yes",
    )

    row = probability_for(subject, record(), CALIBRATION)

    assert (
        row.series,
        row.station,
        row.split,
        row.kind,
        row.strike_lo,
        row.strike_hi,
        row.result,
        row.entry_price,
        row.event_date,
    ) == (
        subject.series,
        subject.station,
        subject.split,
        subject.kind,
        subject.strike_lo,
        subject.strike_hi,
        subject.result,
        subject.entry_price,
        subject.event_date,
    )
    assert row.outcome == 1
    assert row.class_probability == event_probability(
        "below", Decimal("80"), None, class_cdf(GEOMETRY_HIGH, Decimal("2.0"))
    )


@pytest.mark.parametrize("lead_hours", [24, 36])
@pytest.mark.parametrize("event_date", [SUMMER, WINTER])
@pytest.mark.parametrize("series", F4_SERIES)
def test_the_decision_lead_is_the_leg_lead_plus_the_sixty_second_close_margin(
    series: str, event_date: date, lead_hours: int
) -> None:
    station = STATIONS[series]
    subject = leg(
        series=series,
        station=station.station,
        tz=station.timezone,
        event_date=event_date,
        lead_hours=lead_hours,
    )

    assert day_end_utc(event_date, station.timezone) - subject.close_time == timedelta(seconds=60)
    assert decision_lead(subject) == timedelta(hours=lead_hours, seconds=60)
    assert lead_bucket_for(decision_lead(subject)) == "24-72h"


@pytest.mark.parametrize("series", F4_SERIES)
def test_a_midnight_close_would_flip_the_bucket_at_the_24h_leg(series: str) -> None:
    timezone_name = STATIONS[series].timezone
    midnight_close = day_end_utc(SUMMER, timezone_name)

    at_midnight = day_end_utc(SUMMER, timezone_name) - (midnight_close - timedelta(hours=24))

    assert at_midnight == timedelta(hours=24)
    assert lead_bucket_for(at_midnight) == "8-24h"


def test_the_two_candidate_lead_definitions_straddle_the_calibration_boundary() -> None:
    assert CALIBRATION.sigma_for(timedelta(hours=24)) == 3.0148021921951225
    assert CALIBRATION.sigma_for(timedelta(hours=24, seconds=60)) == 4.232898229269733


def test_the_settlement_predicate_settles_exactly_one_row_per_observed_high() -> None:
    ladder = event_ladder(MIAMI_GEOMETRY)

    for high in range(78, 90):
        settled = [
            rung.ticker
            for rung in ladder
            if settles_yes(rung.kind, rung.strike_lo, rung.strike_hi, Decimal(high))
        ]
        assert len(settled) == 1, (high, settled)


def test_the_repo_settlement_oracle_disagrees_on_the_upper_integer_of_every_bracket() -> None:
    for ticker, high in (
        ("KXHIGHMIA-24OCT24-B80.5", 81),
        ("KXHIGHMIA-24OCT24-B82.5", 83),
        ("KXHIGHMIA-24OCT24-B84.5", 85),
        ("KXHIGHMIA-24OCT24-B86.5", 87),
    ):
        parsed = parse_ticker(ticker)
        low, upper = parsed.strikes

        assert settles_yes("bracket", low, upper, Decimal(high))
        assert not settle_bracket(parsed, Decimal(high))


def test_the_baseline_is_the_entry_price_exactly() -> None:
    rows = [
        probability_for(leg(entry_price=Decimal("0.31"), result="yes"), record(), CALIBRATION),
        probability_for(leg(entry_price=Decimal("0.07"), result="no"), record(), CALIBRATION),
        probability_for(leg(entry_price=Decimal("0.62"), result="yes"), record(), CALIBRATION),
    ]

    assert [row.outcome for row in rows] == [1, 0, 1]
    assert baseline_brier(rows) == brier_score(
        [Decimal("0.31"), Decimal("0.07"), Decimal("0.62")], [1, 0, 1]
    )
    assert baseline_brier(rows) == Decimal("0.208467")
    assert class_brier(rows) == Decimal("0.337321")
    assert brier_skill(class_brier(rows), baseline_brier(rows)) == Decimal("-0.618103")


def test_the_brier_skill_score_is_the_ratio_against_the_baseline() -> None:
    assert brier_skill(Decimal("0.05"), Decimal("0.20")) == Decimal("0.750000")
    assert brier_skill(Decimal("0.20"), Decimal("0.20")) == Decimal("0.000000")
    assert brier_skill(Decimal("0.30"), Decimal("0.20")) == Decimal("-0.500000")


@pytest.mark.parametrize("prec", AMBIENT_PRECISIONS)
def test_the_probability_reads_the_same_figure_at_every_ambient_precision(prec: int) -> None:
    with localcontext(prec=prec):
        row = probability_for(leg(), record(native_sigma_f=None), CALIBRATION)
        total = ladder_sum(
            event_ladder(MIAMI_GEOMETRY), class_cdf(GEOMETRY_HIGH, BUCKET_SIGMA), event=MIAMI_EVENT
        )

    assert str(row.class_probability) == "0.17251565134877883"
    assert str(row.sigma_f) == "4.232898229269733"
    assert str(row.effective_sigma_f) == "4.349416905673085866394477688"
    assert str(row.sensitivity[Decimal("0.5")]) == "0.3136659407972427"
    assert str(row.sensitivity[Decimal("2")]) == "0.0886021268570425"
    assert str(total.total) == "1.00000000000000000"


@needs_corpus
def test_the_realised_sigma_source_split_over_the_frozen_corpus() -> None:
    legs = {
        (row.station, row.event_date, row.lead_hours): row for row in read_sample_freeze(SAMPLE)
    }

    split = {}
    for name, path in ((CLASS_A, CLASS_A_FREEZE), (CLASS_B, CLASS_B_FREEZE)):
        rows = read_class_freeze(path)
        split[name] = Counter(
            sigma_for(row, legs[(row.station, row.event_date, row.lead_hours)], CALIBRATION)[1]
            for row in rows
        )

    assert split[CLASS_A] == Counter({EXTERNAL_CALIBRATION: 12490})
    assert split[CLASS_B] == Counter({NATIVE_XND: 4505})


@needs_corpus
def test_a_spring_forward_leg_is_priced_off_the_earlier_bucket_sigma() -> None:
    legs = [
        row
        for row in read_sample_freeze(SAMPLE)
        if row.event_date == date(2025, 3, 9) and row.lead_hours == 24
    ]
    records = {
        (row.station, row.event_date, row.lead_hours): row
        for row in read_class_freeze(CLASS_A_FREEZE)
    }

    rows = [
        probability_for(row, records[(row.station, row.event_date, row.lead_hours)], CALIBRATION)
        for row in legs
    ]

    assert len(rows) == 37
    assert {row.sigma_f for row in rows} == {Decimal("3.0148021921951225")}
    assert {row.sigma_source for row in rows} == {EXTERNAL_CALIBRATION}
    assert {row.effective_sigma_f for row in rows} == {Decimal("3.176323701713116698798731371")}


@needs_corpus
def test_the_frozen_corpus_does_not_land_in_one_lead_bucket() -> None:
    legs = read_sample_freeze(SAMPLE)

    buckets = Counter((row.lead_hours, lead_bucket_for(decision_lead(row))) for row in legs)
    early = {row.event_date for row in legs if lead_bucket_for(decision_lead(row)) == "8-24h"}

    assert buckets == Counter({(24, "24-72h"): 12394, (36, "24-72h"): 8351, (24, "8-24h"): 37})
    assert early == {date(2025, 3, 9)}
    assert Counter(decision_lead(row) for row in legs if row.event_date == date(2025, 3, 9)) == (
        Counter({timedelta(hours=23, seconds=60): 37, timedelta(hours=35, seconds=60): 33})
    )


@needs_corpus
def test_the_window_basis_carries_through_from_the_class_freeze() -> None:
    legs = {
        (row.station, row.event_date, row.lead_hours): row for row in read_sample_freeze(SAMPLE)
    }
    class_a = read_class_freeze(CLASS_A_FREEZE)
    class_b = read_class_freeze(CLASS_B_FREEZE)
    sliced = class_a[:40] + class_a[-40:] + class_b[:40] + class_b[-40:]

    rows = [
        probability_for(legs[(row.station, row.event_date, row.lead_hours)], row, CALIBRATION)
        for row in sliced
    ]

    assert {(row.forecast_class, row.lead_hours, row.window_basis) for row in rows} == {
        (CLASS_A, 24, LST_FULL_LESS_LAST_HOUR),
        (CLASS_A, 36, LST_FULL),
        (CLASS_B, 24, UTC_12Z_00Z),
        (CLASS_B, 36, UTC_12Z_00Z),
    }


@needs_corpus
def test_every_probability_over_a_corpus_slice_is_a_decimal_on_the_unit_interval() -> None:
    legs = read_sample_freeze(SAMPLE)[:300]
    keys = {(row.station, row.event_date, row.lead_hours) for row in legs}
    records = [
        row
        for row in read_class_freeze(CLASS_B_FREEZE)
        if (row.station, row.event_date, row.lead_hours) in keys
    ]

    rows = probabilities(legs, records, CALIBRATION)
    tally = sigma_source_tally(rows)

    assert len(rows) == len(legs)
    assert len({row.key for row in rows}) == len(rows)
    assert all(isinstance(row.class_probability, Decimal) for row in rows)
    assert all(Decimal(0) <= row.class_probability <= Decimal(1) for row in rows)
    assert [(t.forecast_class, t.native_xnd, t.external_calibration) for t in tally] == [
        (CLASS_B, len(rows), 0)
    ]
