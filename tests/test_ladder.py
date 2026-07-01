from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from bot.replay.ladder import BookEvent, Ladder


UTC = timezone.utc
T0 = datetime(2026, 7, 20, 15, 0, 0, tzinfo=UTC)
T1 = T0 + timedelta(seconds=30)
TICKER = "KXHIGHDEN-26JUL20-T85"


@pytest.fixture
def ladder() -> Ladder:
    return Ladder(TICKER)


def _snapshot(side: str, price: str, size: str, *, seq: int = 1, at: datetime = T0) -> BookEvent:
    return BookEvent(
        received_at=at,
        seq=seq,
        side=side,
        price=price,
        size=size,
        is_snapshot=True,
    )


def _delta(side: str, price: str, size: str, *, seq: int = 2, at: datetime = T1) -> BookEvent:
    return BookEvent(
        received_at=at,
        seq=seq,
        side=side,
        price=price,
        size=size,
        is_snapshot=False,
    )


def _composed(ladder: Ladder) -> Ladder:
    for event in (
        _snapshot("yes", "0.4000", "10.00"),
        _snapshot("yes", "0.3900", "5.00"),
        _snapshot("no", "0.5500", "7.00"),
        _snapshot("no", "0.5400", "3.00"),
        _delta("yes", "0.4100", "2.00", seq=2),
        _delta("yes", "0.3900", "-5.00", seq=3),
        _delta("no", "0.5500", "-2.00", seq=4),
        _delta("no", "0.5300", "8.00", seq=5),
    ):
        ladder.apply(event)
    return ladder


def test_snapshot_plus_deltas_compose_to_pinned_book(ladder: Ladder) -> None:
    _composed(ladder)
    assert ladder.levels == {
        "yes": {
            Decimal("0.4000"): Decimal("10.00"),
            Decimal("0.4100"): Decimal("2.00"),
        },
        "no": {
            Decimal("0.5500"): Decimal("5.00"),
            Decimal("0.5400"): Decimal("3.00"),
            Decimal("0.5300"): Decimal("8.00"),
        },
    }


def test_mid_stream_snapshot_resets_rather_than_merges(ladder: Ladder) -> None:
    ladder.apply(_snapshot("yes", "0.4000", "10.00"))
    ladder.apply(_snapshot("no", "0.5500", "7.00"))
    ladder.apply(_delta("yes", "0.4100", "2.00", seq=2))
    ladder.apply(_snapshot("yes", "0.3000", "1.00", seq=3, at=T1))
    assert ladder.levels == {"yes": {Decimal("0.3000"): Decimal("1.00")}, "no": {}}
    assert ladder.batch_key == (T1, 3)


def test_multi_row_snapshot_batch_keeps_every_row(ladder: Ladder) -> None:
    ladder.apply(_snapshot("yes", "0.2000", "9.00"))
    ladder.apply(_snapshot("yes", "0.1900", "4.00"))
    ladder.apply(_snapshot("no", "0.7000", "6.00"))
    assert ladder.levels == {
        "yes": {Decimal("0.2000"): Decimal("9.00"), Decimal("0.1900"): Decimal("4.00")},
        "no": {Decimal("0.7000"): Decimal("6.00")},
    }


def test_delta_for_absent_level_creates_it(ladder: Ladder) -> None:
    ladder.apply(_snapshot("yes", "0.4000", "10.00"))
    ladder.apply(_delta("yes", "0.4100", "2.00"))
    assert ladder.levels["yes"][Decimal("0.4100")] == Decimal("2.00")


def test_zero_delta_leaves_the_level_untouched(ladder: Ladder) -> None:
    ladder.apply(_snapshot("yes", "0.4000", "10.00"))
    ladder.apply(_delta("yes", "0.4000", "0.00", seq=2))
    ladder.apply(_delta("yes", "0.3300", "0.00", seq=3))
    assert ladder.levels["yes"] == {Decimal("0.4000"): Decimal("10.00")}


def test_level_disappears_when_its_total_reaches_zero(ladder: Ladder) -> None:
    ladder.apply(_snapshot("yes", "0.4000", "10.00"))
    ladder.apply(_snapshot("yes", "0.3900", "5.00"))
    ladder.apply(_delta("yes", "0.4000", "-4.00", seq=2))
    ladder.apply(_delta("yes", "0.4000", "-6.00", seq=3))
    assert Decimal("0.4000") not in ladder.levels["yes"]
    assert ladder.levels["yes"] == {Decimal("0.3900"): Decimal("5.00")}


def test_negative_total_raises_naming_ticker_and_seq(ladder: Ladder) -> None:
    ladder.apply(_snapshot("yes", "0.4000", "10.00"))
    with pytest.raises(ValueError) as exc:
        ladder.apply(_delta("yes", "0.4000", "-11.00", seq=77))
    assert TICKER in str(exc.value)
    assert "seq=77" in str(exc.value)


def test_level_killed_and_reborn_at_another_scale_is_one_key(ladder: Ladder) -> None:
    ladder.apply(_snapshot("yes", "0.1", "3.00"))
    ladder.apply(_delta("yes", "0.1000", "-3.00", seq=2))
    ladder.apply(_delta("yes", "0.1", "4.00", seq=3))
    assert [str(price) for price in ladder.levels["yes"]] == ["0.1000"]
    assert str(ladder.row(T1).yes_bid) == "0.1000"


def test_full_depth_below_the_touch_is_retained(ladder: Ladder) -> None:
    ladder.apply(_snapshot("yes", "0.4000", "10.00"))
    ladder.apply(_snapshot("yes", "0.3900", "5.00"))
    ladder.apply(_snapshot("yes", "0.3800", "2.00"))
    ladder.apply(_snapshot("no", "0.5500", "7.00"))
    assert ladder.row(T0).yes_bid == Decimal("0.4000")
    assert ladder.levels["yes"][Decimal("0.3800")] == Decimal("2.00")
    assert sorted(ladder.levels["yes"], reverse=True) == [
        Decimal("0.4000"),
        Decimal("0.3900"),
        Decimal("0.3800"),
    ]


def test_rest_inversion_invariant_holds(ladder: Ladder) -> None:
    row = _composed(ladder).row(T1)
    assert row.ticker == TICKER
    assert row.snapshot_at == T1
    assert row.yes_bid == Decimal("0.4100")
    assert row.no_bid == Decimal("0.5500")
    assert row.yes_ask == Decimal("1") - row.no_bid
    assert row.no_ask == Decimal("1") - row.yes_bid
    assert row.yes_bid_depth == 2
    assert row.no_bid_depth == 5
    assert row.yes_ask_depth == row.no_bid_depth
    assert row.no_ask_depth == row.yes_bid_depth


def test_empty_side_maps_to_one_at_zero_depth(ladder: Ladder) -> None:
    ladder.apply(_snapshot("yes", "0.4000", "10.00"))
    row = ladder.row(T0)
    assert row.no_bid == Decimal("0")
    assert row.yes_ask == Decimal("1")
    assert row.yes_ask_depth == 0
    assert row.no_bid_depth == 0
    empty = Ladder(TICKER).row(T0)
    assert empty.yes_ask == Decimal("1")
    assert empty.no_ask == Decimal("1")
    assert empty.yes_bid_depth == 0
    assert empty.no_ask_depth == 0


def test_price_that_does_not_fit_four_decimals_raises(ladder: Ladder) -> None:
    with pytest.raises(ValueError) as exc:
        ladder.apply(_delta("yes", "0.10005", "1.00"))
    assert TICKER in str(exc.value)
    assert "0.10005" in str(exc.value)


def test_size_that_does_not_fit_two_decimals_raises(ladder: Ladder) -> None:
    with pytest.raises(ValueError) as exc:
        ladder.apply(_delta("yes", "0.1000", "1.005"))
    assert TICKER in str(exc.value)
    assert "1.005" in str(exc.value)
