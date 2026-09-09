from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from bot.lag import forecast_entry
from bot.lag.fee_floor import TICK_CENTS, economic_bar_cents_per_contract, published_taker_fee
from bot.lag.forecast_entry import (
    NO,
    SCREEN_RULE,
    SIZE,
    TICK_RULE,
    WALKED_TICK_REPORTED_ONLY,
    WALKED_TICK_RULE,
    YES,
    TradedLeg,
    depth_distribution,
    depth_ok,
    entry_counts,
    entry_of,
    entry_side,
    fee_cents_per_contract,
    quantile,
    screen_depth,
)
from bot.lag.forecast_sample import SampleLeg, read_sample_freeze


BAR_PRICE = Decimal("0.50")
FEE_PER_CONTRACT = Decimal("1.769230769230769230769230769")
PAID_PROFIT_CENTS = Decimal("47.23076923076923076923076923")
UNPAID_PROFIT_CENTS = Decimal("-52.76923076923076923076923077")
WALKED_PAID_PROFIT_CENTS = Decimal("46.23076923076923076923076923")
WALKED_UNPAID_PROFIT_CENTS = Decimal("-53.76923076923076923076923077")
ONE_DAY_IN_THREE = Decimal("0.3333333333333333333333333333")
AMBIENT_PRECISIONS = (20, 28, 50)
REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "data" / "tape_studies" / "f4_inputs" / "sample.jsonl"

needs_tape = pytest.mark.skipif(
    not SAMPLE.exists(),
    reason="the recorded tape is not on this host",
)


def leg(
    *,
    entry_price: Decimal,
    result: str,
    event_date: date = date(2025, 8, 1),
    trailing_prints: int = 40,
    trailing_contracts: Decimal = Decimal(400),
) -> SampleLeg:
    close_time = datetime(2025, 8, 2, 4, tzinfo=timezone.utc)
    return SampleLeg(
        ticker=f"KXHIGHDEN-{event_date:%y%b%d}-B94.5".upper(),
        series="KXHIGHDEN",
        station="KDEN",
        timezone="America/Denver",
        event_date=event_date,
        split="discovery",
        lead_hours=24,
        close_time=close_time,
        as_of=close_time - timedelta(hours=24),
        entry_price=entry_price,
        staleness_minutes=Decimal("3.5"),
        era="0016",
        trailing_prints=trailing_prints,
        trailing_contracts=trailing_contracts,
        strike_lo=Decimal("94.5"),
        strike_hi=None,
        kind="above",
        result=result,
    )


def thin_day_legs() -> list[SampleLeg]:
    return [
        leg(entry_price=BAR_PRICE, result="yes", event_date=date(2025, 8, 1)),
        leg(entry_price=BAR_PRICE, result="yes", event_date=date(2025, 8, 2)),
        leg(
            entry_price=BAR_PRICE,
            result="yes",
            event_date=date(2025, 8, 3),
            trailing_contracts=Decimal(4),
        ),
    ]


def traded_at(prec: int, result: str) -> TradedLeg:
    with localcontext(prec=prec):
        return entry_of(leg(entry_price=BAR_PRICE, result=result), Decimal("0.80"))


def lost_at(prec: int) -> Decimal:
    with localcontext(prec=prec):
        return screen_depth(thin_day_legs()).event_days_lost


@pytest.fixture(scope="module")
def sample_legs() -> list[SampleLeg]:
    return read_sample_freeze(SAMPLE)


def test_the_published_fee_and_the_share_a_single_contract_carries() -> None:
    assert published_taker_fee(SIZE, BAR_PRICE) == Decimal("0.46")
    assert fee_cents_per_contract(SIZE, BAR_PRICE) == FEE_PER_CONTRACT


def test_the_charged_entry_cost_is_the_bar_the_statistic_declines_to_charge_twice() -> None:
    charged = fee_cents_per_contract(SIZE, BAR_PRICE) + TICK_CENTS

    assert charged == economic_bar_cents_per_contract(Decimal(26), BAR_PRICE)


@pytest.mark.parametrize(
    ("result", "expected"),
    [("yes", PAID_PROFIT_CENTS), ("no", UNPAID_PROFIT_CENTS)],
)
def test_a_yes_entry_pays_the_settlement_less_its_own_entry_cost(
    result: str, expected: Decimal
) -> None:
    traded = entry_of(leg(entry_price=BAR_PRICE, result=result), Decimal("0.80"))

    assert traded.traded is True
    assert traded.side == YES
    assert traded.probability == Decimal("0.80")
    assert traded.executed_price == BAR_PRICE
    assert traded.entry_fee_cents == FEE_PER_CONTRACT
    assert traded.entry_tick_cents == TICK_CENTS
    assert traded.size == SIZE
    assert traded.trailing_prints == 40
    assert traded.trailing_contracts == Decimal(400)
    assert traded.net_profit_cents == expected


@pytest.mark.parametrize(
    ("result", "expected"),
    [("yes", WALKED_PAID_PROFIT_CENTS), ("no", WALKED_UNPAID_PROFIT_CENTS)],
)
def test_the_walked_entry_reprices_its_fee_and_still_pays_the_tick(
    result: str, expected: Decimal
) -> None:
    traded = entry_of(leg(entry_price=BAR_PRICE, result=result), Decimal("0.80"))

    assert traded.walked_executed_price == Decimal("0.51")
    assert traded.walked_entry_fee_cents == FEE_PER_CONTRACT
    assert traded.walked_net_profit_cents == expected


def test_a_no_entry_executes_against_the_complement_of_the_quoted_price() -> None:
    traded = entry_of(leg(entry_price=Decimal("0.30"), result="no"), Decimal("0.20"))

    assert traded.side == NO
    assert traded.executed_price == Decimal("0.70")
    assert published_taker_fee(SIZE, traded.executed_price) == Decimal("0.39")
    assert traded.entry_fee_cents == Decimal("1.50")
    assert traded.net_profit_cents == Decimal("27.50")


def test_the_walked_fee_is_priced_at_the_walked_price_not_the_executed_one() -> None:
    traded = entry_of(leg(entry_price=Decimal("0.30"), result="no"), Decimal("0.20"))

    assert traded.walked_executed_price == Decimal("0.71")
    assert traded.entry_fee_cents == Decimal("1.50")
    assert traded.walked_entry_fee_cents == Decimal("1.461538461538461538461538462")
    assert traded.walked_net_profit_cents == Decimal("26.53846153846153846153846154")


def test_the_walk_at_the_ceiling_prices_a_zero_fee() -> None:
    traded = entry_of(leg(entry_price=Decimal("0.99"), result="yes"), Decimal("0.995"))

    assert traded.executed_price == Decimal("0.99")
    assert traded.walked_executed_price == Decimal("1.00")
    assert published_taker_fee(SIZE, traded.walked_executed_price) == Decimal("0.00")
    assert traded.walked_entry_fee_cents == Decimal("0.00")
    assert traded.walked_net_profit_cents == Decimal("-1.00")


def test_a_price_the_forecast_agrees_with_is_carried_untraded() -> None:
    traded = entry_of(leg(entry_price=BAR_PRICE, result="yes"), BAR_PRICE)

    assert traded.traded is False
    assert traded.side is None
    assert traded.executed_price is None
    assert traded.entry_fee_cents is None
    assert traded.net_profit_cents is None
    assert traded.walked_executed_price is None
    assert traded.walked_entry_fee_cents is None
    assert traded.walked_net_profit_cents is None
    assert traded.entry_tick_cents == TICK_CENTS
    assert traded.size == SIZE
    assert traded.probability == BAR_PRICE
    assert traded.entry_price == BAR_PRICE
    assert traded.trailing_prints == 40
    assert traded.trailing_contracts == Decimal(400)


def test_the_untraded_legs_are_tallied_apart_from_the_traded_ones() -> None:
    counts = entry_counts(
        [
            entry_of(leg(entry_price=BAR_PRICE, result="yes"), Decimal("0.80")),
            entry_of(leg(entry_price=BAR_PRICE, result="no"), Decimal("0.20")),
            entry_of(leg(entry_price=BAR_PRICE, result="yes"), BAR_PRICE),
        ]
    )

    assert counts.n == 3
    assert counts.traded_n == 2
    assert counts.untraded_n == 1
    assert counts.traded_n + counts.untraded_n == counts.n


def test_a_priced_leg_cannot_be_rewritten_after_it_is_built() -> None:
    traded = entry_of(leg(entry_price=BAR_PRICE, result="yes"), Decimal("0.80"))

    with pytest.raises(FrozenInstanceError):
        traded.side = NO


@pytest.mark.parametrize("prec", AMBIENT_PRECISIONS)
def test_the_entry_reads_the_same_figures_at_every_ambient_precision(prec: int) -> None:
    paid = traded_at(prec, "yes")
    unpaid = traded_at(prec, "no")

    assert paid.entry_fee_cents == FEE_PER_CONTRACT
    assert str(paid.entry_fee_cents) == str(FEE_PER_CONTRACT)
    assert paid.net_profit_cents == PAID_PROFIT_CENTS
    assert str(paid.net_profit_cents) == str(PAID_PROFIT_CENTS)
    assert unpaid.net_profit_cents == UNPAID_PROFIT_CENTS
    assert str(unpaid.net_profit_cents) == str(UNPAID_PROFIT_CENTS)
    assert paid.walked_net_profit_cents == WALKED_PAID_PROFIT_CENTS
    assert str(paid.walked_net_profit_cents) == str(WALKED_PAID_PROFIT_CENTS)
    assert unpaid.walked_net_profit_cents == WALKED_UNPAID_PROFIT_CENTS
    assert str(unpaid.walked_net_profit_cents) == str(WALKED_UNPAID_PROFIT_CENTS)


@pytest.mark.parametrize("prec", AMBIENT_PRECISIONS)
def test_the_event_days_lost_reads_the_same_figure_at_every_ambient_precision(prec: int) -> None:
    lost = lost_at(prec)

    assert lost == ONE_DAY_IN_THREE
    assert str(lost) == str(ONE_DAY_IN_THREE)


def test_the_screen_counts_the_legs_and_the_event_days_it_drops() -> None:
    screened = screen_depth(thin_day_legs())

    assert screened.candidates == 3
    assert screened.kept == 2
    assert screened.dropped == 1
    assert screened.event_days == 3
    assert screened.event_days_kept == 2


def test_a_leg_with_no_trailing_prints_reaches_the_report_in_its_own_tally() -> None:
    unprinted = leg(
        entry_price=Decimal("0.42"),
        result="yes",
        trailing_prints=0,
        trailing_contracts=Decimal(0),
    )
    traded = entry_of(unprinted, Decimal("0.60"))
    distribution = depth_distribution([unprinted])

    assert traded.traded is True
    assert traded.trailing_prints == 0
    assert traded.trailing_contracts == Decimal(0)
    assert distribution.prints_at_zero == 1
    assert distribution.contracts_at_zero == 1
    assert distribution.prints_below_size == 1
    assert distribution.contracts_below_size == 1


@pytest.mark.parametrize(
    ("contracts", "kept"),
    [(Decimal(26), True), (Decimal(25), False), (Decimal(27), True)],
)
def test_the_depth_screen_keeps_a_leg_that_traded_exactly_the_size(
    contracts: Decimal, kept: bool
) -> None:
    screened = leg(entry_price=BAR_PRICE, result="yes", trailing_contracts=contracts)

    assert depth_ok(screened) is kept


def test_the_quantile_truncates_the_index_rather_than_rounding_it() -> None:
    values = [0, 10, 20, 30, 40, 50, 60]

    assert quantile(values, Decimal("0.10")) == 0
    assert quantile(values, Decimal("0.25")) == 10
    assert quantile(values, Decimal("1")) == 60


@pytest.mark.parametrize(
    ("probability", "price", "side"),
    [
        (Decimal("0.60"), Decimal("0.40"), YES),
        (Decimal("0.40"), Decimal("0.60"), NO),
        (Decimal("0.50"), Decimal("0.50"), None),
        (Decimal("0.5000000000000000000000000001"), Decimal("0.50"), YES),
        (Decimal("0.4999999999999999999999999999"), Decimal("0.50"), NO),
    ],
)
def test_entry_side_takes_the_forecast_side_and_skips_the_tie(
    probability: Decimal, price: Decimal, side: str | None
) -> None:
    assert entry_side(probability, price) == side


def test_the_tick_charged_is_the_published_constant() -> None:
    assert forecast_entry.TICK_CENTS is TICK_CENTS
    assert forecast_entry.TICK_CENTS == Decimal("1")


def test_the_fee_charged_is_the_oracle_the_fee_module_is_checked_against() -> None:
    assert forecast_entry.published_taker_fee is published_taker_fee


def test_the_size_is_the_externally_stated_one() -> None:
    assert SIZE == Decimal(26)


def test_the_artifact_labels_the_readings_it_was_built_under() -> None:
    assert SCREEN_RULE == "trailing_contracts_ge_26"
    assert TICK_RULE == "one_tick_constant"
    assert WALKED_TICK_RULE == "one_tick_constant_entry_walked_one_tick"
    assert "gates nothing" in WALKED_TICK_REPORTED_ONLY
    assert "no multiplicity correction" in WALKED_TICK_REPORTED_ONLY


@needs_tape
def test_the_frozen_sample_reads_its_measured_depth_distributions(
    sample_legs: list[SampleLeg],
) -> None:
    distribution = depth_distribution([row for row in sample_legs if row.lead_hours == 24])

    assert distribution.prints_p10 == 3
    assert distribution.prints_p25 == 9
    assert distribution.prints_p50 == 23
    assert distribution.prints_p75 == 47
    assert distribution.prints_p90 == 86
    assert distribution.prints_max == 761
    assert distribution.prints_at_zero == 245
    assert distribution.prints_below_size == 6725
    assert distribution.contracts_p10 == Decimal(56)
    assert distribution.contracts_p25 == Decimal(226)
    assert distribution.contracts_p50 == Decimal(671)
    assert distribution.contracts_p75 == Decimal(1468)
    assert distribution.contracts_p90 == Decimal(2811)
    assert distribution.contracts_max == Decimal(74255)
    assert distribution.contracts_below_size == 779


@needs_tape
def test_the_frozen_sample_reads_what_the_depth_screen_costs_at_each_lead(
    sample_legs: list[SampleLeg],
) -> None:
    at_24h = screen_depth([row for row in sample_legs if row.lead_hours == 24])
    at_36h = screen_depth([row for row in sample_legs if row.lead_hours == 36])

    assert at_24h.candidates == 12431
    assert at_24h.kept == 11652
    assert at_24h.dropped == 779
    assert at_24h.event_days == 342
    assert at_24h.event_days_kept == 342
    assert at_24h.event_days_lost == Decimal(0)
    assert at_36h.candidates == 8351
    assert at_36h.kept == 6324
    assert at_36h.dropped == 2027
    assert at_36h.event_days == 336
    assert at_36h.event_days_kept == 335
    assert at_36h.event_days_lost == Decimal("0.002976190476190476190476190476")
