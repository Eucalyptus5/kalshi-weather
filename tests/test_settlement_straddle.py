from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from bot.lag.placement_grid import CloseSidecar, MarketClose, read_sidecar
from bot.lag.settlement_straddle import (
    cell_edges,
    city_event_days,
    event_ticker_of,
    settling_row,
    straddle_of,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEN = REPO_ROOT / "data" / "tape_studies" / "closes_v2" / "KXHIGHDEN.json"
NY = REPO_ROOT / "data" / "tape_studies" / "closes_v2" / "KXHIGHNY.json"

pytestmark = pytest.mark.skipif(
    not (DEN.exists() and NY.exists()), reason="the recorded tape is not on this host"
)

ROOT = "KXHIGHDEN"
STATION = "KDEN"
ZONE = "America/Denver"
EVENT_DATE = date(2026, 8, 1)
EVENT_TICKER = "KXHIGHDEN-26AUG01"


@pytest.fixture(scope="module")
def sidecar() -> CloseSidecar:
    return read_sidecar(DEN)


@pytest.fixture(scope="module")
def ny_sidecar() -> CloseSidecar:
    return read_sidecar(NY)


def test_the_event_day_lists_six_rows(sidecar: CloseSidecar) -> None:
    rows = {
        ticker: market
        for ticker, market in sidecar.markets.items()
        if market.event_ticker == EVENT_TICKER
    }
    assert sorted(rows) == [
        "KXHIGHDEN-26AUG01-B88.5",
        "KXHIGHDEN-26AUG01-B90.5",
        "KXHIGHDEN-26AUG01-B92.5",
        "KXHIGHDEN-26AUG01-B94.5",
        "KXHIGHDEN-26AUG01-T88",
        "KXHIGHDEN-26AUG01-T95",
    ]
    assert all(market.status == "finalized" for market in rows.values())


def test_the_cell_edges_are_the_ladders_own(sidecar: CloseSidecar) -> None:
    assert cell_edges(sidecar, EVENT_TICKER) == (87, 89, 91, 93, 95)


def test_a_floor_strike_union_has_nothing_to_union_on_the_less_row(sidecar: CloseSidecar) -> None:
    less = sidecar.markets["KXHIGHDEN-26AUG01-T88"]
    assert less.strike_type == "less"
    assert less.floor_strike is None
    assert less.cap_strike == 88
    greater = sidecar.markets["KXHIGHDEN-26AUG01-T95"]
    assert greater.strike_type == "greater"
    assert greater.floor_strike == 95
    assert greater.cap_strike is None


def test_the_event_ticker_is_the_venues_own_spelling() -> None:
    assert event_ticker_of(ROOT, EVENT_DATE) == EVENT_TICKER


def test_two_readings_either_side_of_one_cell_edge_straddle_it(sidecar: CloseSidecar) -> None:
    record = straddle_of(
        sidecar,
        root=ROOT,
        station=STATION,
        event_date=EVENT_DATE,
        timezone=ZONE,
        observed_f=Decimal("93"),
        acis_f=Decimal("94"),
    )
    assert record is not None
    assert record.separating_strikes == (93,)
    assert record.cell_edges == (87, 89, 91, 93, 95)
    assert record.root == ROOT
    assert record.station == STATION
    assert record.event_date == EVENT_DATE
    assert record.extreme == "max"
    assert record.timezone == ZONE
    assert record.observed_f == Decimal("93")
    assert record.acis_f == Decimal("94")

    settled = settling_row(sidecar, EVENT_TICKER, record.acis_f)
    assert settled.ticker == "KXHIGHDEN-26AUG01-B94.5"
    assert settled.result == "yes"


def test_swap_detection_rests_on_the_settled_row_not_the_strikes(sidecar: CloseSidecar) -> None:
    record = straddle_of(
        sidecar,
        root=ROOT,
        station=STATION,
        event_date=EVENT_DATE,
        timezone=ZONE,
        observed_f=Decimal("94"),
        acis_f=Decimal("93"),
    )
    assert record is not None
    assert record.separating_strikes == (93,)

    settled = settling_row(sidecar, EVENT_TICKER, record.acis_f)
    assert settled.ticker == "KXHIGHDEN-26AUG01-B92.5"
    assert settled.result == "no"


def test_a_one_degree_disagreement_inside_one_bracket_is_not_a_straddle(
    sidecar: CloseSidecar,
) -> None:
    record = straddle_of(
        sidecar,
        root=ROOT,
        station=STATION,
        event_date=EVENT_DATE,
        timezone=ZONE,
        observed_f=Decimal("94"),
        acis_f=Decimal("95"),
    )
    assert record is None


def test_two_readings_that_settle_the_same_row_are_counted_not_raised(
    sidecar: CloseSidecar,
) -> None:
    observed_f = Decimal("88")
    acis_f = Decimal("89")
    assert settling_row(sidecar, EVENT_TICKER, observed_f).ticker == "KXHIGHDEN-26AUG01-B88.5"
    assert settling_row(sidecar, EVENT_TICKER, acis_f).ticker == "KXHIGHDEN-26AUG01-B88.5"
    assert (
        straddle_of(
            sidecar,
            root=ROOT,
            station=STATION,
            event_date=EVENT_DATE,
            timezone=ZONE,
            observed_f=observed_f,
            acis_f=acis_f,
        )
        is None
    )


@pytest.mark.parametrize(
    "reading, ticker",
    [
        (Decimal("80"), "KXHIGHDEN-26AUG01-T88"),
        (Decimal("87"), "KXHIGHDEN-26AUG01-T88"),
        (Decimal("88"), "KXHIGHDEN-26AUG01-B88.5"),
        (Decimal("89"), "KXHIGHDEN-26AUG01-B88.5"),
        (Decimal("90"), "KXHIGHDEN-26AUG01-B90.5"),
        (Decimal("93"), "KXHIGHDEN-26AUG01-B92.5"),
        (Decimal("95"), "KXHIGHDEN-26AUG01-B94.5"),
        (Decimal("96"), "KXHIGHDEN-26AUG01-T95"),
    ],
)
def test_the_rows_partition_the_line(sidecar: CloseSidecar, reading: Decimal, ticker: str) -> None:
    assert settling_row(sidecar, EVENT_TICKER, reading).ticker == ticker


def test_the_boundary_reading_belongs_to_the_between_row(sidecar: CloseSidecar) -> None:
    assert sidecar.markets["KXHIGHDEN-26AUG01-T88"].cap_strike == 88
    assert sidecar.markets["KXHIGHDEN-26AUG01-B88.5"].floor_strike == 88
    assert settling_row(sidecar, EVENT_TICKER, Decimal("88")).ticker == "KXHIGHDEN-26AUG01-B88.5"

    assert sidecar.markets["KXHIGHDEN-26AUG01-T95"].floor_strike == 95
    assert sidecar.markets["KXHIGHDEN-26AUG01-B94.5"].cap_strike == 95
    assert settling_row(sidecar, EVENT_TICKER, Decimal("95")).ticker == "KXHIGHDEN-26AUG01-B94.5"


def test_two_rows_claiming_one_reading_are_refused(sidecar: CloseSidecar) -> None:
    stretched = replace(sidecar.markets["KXHIGHDEN-26AUG01-T88"], cap_strike=90)
    doubled = CloseSidecar(
        root=sidecar.root,
        markets={**sidecar.markets, stretched.ticker: stretched},
        voided=sidecar.voided,
        sha256=sidecar.sha256,
    )
    with pytest.raises(ValueError) as refused:
        settling_row(doubled, EVENT_TICKER, Decimal("89"))
    assert "KXHIGHDEN-26AUG01-B88.5" in str(refused.value)
    assert "KXHIGHDEN-26AUG01-T88" in str(refused.value)
    assert "89" in str(refused.value)


@pytest.mark.parametrize(
    "event_ticker",
    ["KXHIGHDEN-26DEC25", "KXHIGHDEN-26AUG01-B94.5", "KXHIGHNY-26AUG01"],
)
def test_an_event_day_the_sidecar_does_not_name_is_refused(
    sidecar: CloseSidecar, event_ticker: str
) -> None:
    with pytest.raises(ValueError):
        settling_row(sidecar, event_ticker, Decimal("93"))
    with pytest.raises(ValueError):
        cell_edges(sidecar, event_ticker)


def test_an_event_day_outside_the_sidecar_refuses_a_straddle(sidecar: CloseSidecar) -> None:
    with pytest.raises(ValueError):
        straddle_of(
            sidecar,
            root=ROOT,
            station=STATION,
            event_date=date(2026, 12, 25),
            timezone=ZONE,
            observed_f=Decimal("93"),
            acis_f=Decimal("94"),
        )


def test_two_separating_edges_in_one_city_event_day_count_once(sidecar: CloseSidecar) -> None:
    record = straddle_of(
        sidecar,
        root=ROOT,
        station=STATION,
        event_date=EVENT_DATE,
        timezone=ZONE,
        observed_f=Decimal("88"),
        acis_f=Decimal("92"),
    )
    assert record is not None
    assert record.separating_strikes == (89, 91)
    assert city_event_days([record]) == 1

    other = straddle_of(
        sidecar,
        root=ROOT,
        station=STATION,
        event_date=date(2026, 8, 2),
        timezone=ZONE,
        observed_f=Decimal("95"),
        acis_f=Decimal("96"),
    )
    assert other is not None
    assert other.cell_edges == (95, 97, 99, 101, 103)
    assert city_event_days([record, other]) == 2
    assert city_event_days([]) == 0


def test_one_city_event_day_twice_over_still_counts_once(sidecar: CloseSidecar) -> None:
    first = straddle_of(
        sidecar,
        root=ROOT,
        station=STATION,
        event_date=date(2026, 8, 2),
        timezone=ZONE,
        observed_f=Decimal("96"),
        acis_f=Decimal("100"),
    )
    second = straddle_of(
        sidecar,
        root=ROOT,
        station=STATION,
        event_date=date(2026, 8, 2),
        timezone=ZONE,
        observed_f=Decimal("100"),
        acis_f=Decimal("96"),
    )
    assert first is not None
    assert second is not None
    assert first != second
    assert first.separating_strikes == (97, 99)
    assert second.separating_strikes == (97, 99)
    assert city_event_days([first, second]) == 1


def test_one_date_across_two_cities_counts_twice(
    sidecar: CloseSidecar, ny_sidecar: CloseSidecar
) -> None:
    den = straddle_of(
        sidecar,
        root=ROOT,
        station=STATION,
        event_date=date(2026, 8, 2),
        timezone=ZONE,
        observed_f=Decimal("96"),
        acis_f=Decimal("100"),
    )
    ny = straddle_of(
        ny_sidecar,
        root="KXHIGHNY",
        station="KNYC",
        event_date=date(2026, 8, 2),
        timezone="America/New_York",
        observed_f=Decimal("82"),
        acis_f=Decimal("84"),
    )
    assert den is not None
    assert ny is not None
    assert den.event_date == ny.event_date
    assert ny.cell_edges == (79, 81, 83, 85, 87)
    assert ny.separating_strikes == (83,)
    assert city_event_days([den, ny]) == 2


def test_the_seam_carries_integral_edges_and_decimal_readings(sidecar: CloseSidecar) -> None:
    record = straddle_of(
        sidecar,
        root=ROOT,
        station=STATION,
        event_date=EVENT_DATE,
        timezone=ZONE,
        observed_f=Decimal("93"),
        acis_f=Decimal("94"),
    )
    assert record is not None
    assert isinstance(record.observed_f, Decimal)
    assert isinstance(record.acis_f, Decimal)
    assert all(isinstance(edge, int) for edge in record.cell_edges)
    assert all(isinstance(strike, int) for strike in record.separating_strikes)
    assert isinstance(settling_row(sidecar, EVENT_TICKER, record.acis_f), MarketClose)
