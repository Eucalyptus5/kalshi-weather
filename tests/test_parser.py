from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal

import pytest

from bot.markets.parser import (
    ParsedTicker,
    event_id,
    event_yes_sum_ok,
    parse_ticker,
    series_id,
)


@pytest.mark.parametrize(
    "ticker, series, event_date, is_monthly, strikes, kind, is_bracket, is_tail",
    [
        (
            "KXHIGHDEN-26APR28-T70.5-72.5",
            "KXHIGHDEN",
            date(2026, 4, 28),
            False,
            (Decimal("70.5"), Decimal("72.5")),
            "bracket",
            True,
            False,
        ),
        (
            "KXHIGHDEN-26APR28-T70.5",
            "KXHIGHDEN",
            date(2026, 4, 28),
            False,
            (Decimal("70.5"),),
            "above",
            False,
            True,
        ),
        (
            "KXHIGHDEN-26APR28-T96.5",
            "KXHIGHDEN",
            date(2026, 4, 28),
            False,
            (Decimal("96.5"),),
            "above",
            False,
            True,
        ),
        (
            "KXLOWNY-26MAR15-T35.5-37.5",
            "KXLOWNY",
            date(2026, 3, 15),
            False,
            (Decimal("35.5"), Decimal("37.5")),
            "bracket",
            True,
            False,
        ),
        (
            "KXHIGHTBOS-26JUL04-T82.5-84.5",
            "KXHIGHTBOS",
            date(2026, 7, 4),
            False,
            (Decimal("82.5"), Decimal("84.5")),
            "bracket",
            True,
            False,
        ),
        (
            "KXHIGHDEN-26DEC15-T30.5-32.5",
            "KXHIGHDEN",
            date(2026, 12, 15),
            False,
            (Decimal("30.5"), Decimal("32.5")),
            "bracket",
            True,
            False,
        ),
        (
            "KXHIGHDEN-26MAY06-T43",
            "KXHIGHDEN",
            date(2026, 5, 6),
            False,
            (Decimal("43"),),
            "above",
            False,
            True,
        ),
        (
            "KXHIGHDEN-26MAY06-B42.5",
            "KXHIGHDEN",
            date(2026, 5, 6),
            False,
            (Decimal("42"), Decimal("43")),
            "bracket",
            True,
            False,
        ),
        (
            "KXHIGHDEN-26MAY19-B49.5",
            "KXHIGHDEN",
            date(2026, 5, 19),
            False,
            (Decimal("49"), Decimal("50")),
            "bracket",
            True,
            False,
        ),
        (
            "KXHIGHTNOLA-26MAY19-B86.5",
            "KXHIGHTNOLA",
            date(2026, 5, 19),
            False,
            (Decimal("86"), Decimal("87")),
            "bracket",
            True,
            False,
        ),
    ],
)
def test_parse_ticker_golden(
    ticker: str,
    series: str,
    event_date: date,
    is_monthly: bool,
    strikes: tuple[Decimal, ...],
    kind: str,
    is_bracket: bool,
    is_tail: bool,
) -> None:
    parsed = parse_ticker(ticker)
    assert isinstance(parsed, ParsedTicker)
    assert parsed.series == series
    assert parsed.event_date == event_date
    assert parsed.is_monthly is is_monthly
    assert parsed.strikes == strikes
    assert parsed.kind == kind
    assert parsed.is_bracket is is_bracket
    assert parsed.is_tail is is_tail
    assert parsed.raw == ticker


def test_decimal_is_string_constructed_not_float() -> None:
    parsed = parse_ticker("KXHIGHDEN-26APR28-T70.5-72.5")
    assert parsed.strikes[0] == Decimal("70.5")
    assert parsed.strikes[1] == Decimal("72.5")
    assert str(parsed.strikes[0]) == "70.5"


def test_monthly_ticker() -> None:
    parsed = parse_ticker("KXRAINSFOM-26JUN-T1.0")
    assert parsed.is_monthly is True
    assert parsed.event_date == date(2026, 6, 1)
    assert parsed.strikes == (Decimal("1.0"),)
    assert parsed.is_tail is True


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "XXHIGHDEN-26APR28-T70.5-72.5",
        "KXHIGHDEN-26ZZZ28-T70.5-72.5",
        "KXHIGHDEN-26FEB30-T70.5-72.5",
        "KXHIGHDEN-26APR28-70.5-72.5",
        "KXHIGHDEN-26APR28-T72.5-70.5",
        "KXHIGHDEN-26APR28-T70.5-72.5-74.5",
        "KXHIGHDEN-26MAY19-B49",
        "KXHIGHDEN-26MAY19-B49.0",
        "KXHIGHDEN-26MAY19-B49.4",
    ],
)
def test_parse_ticker_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_ticker(bad)


def test_b_form_rejects_two_strikes() -> None:
    with pytest.raises(ValueError, match="B-form must be single-strike"):
        parse_ticker("KXHIGHDEN-26MAY06-B42.5-43.5")


def test_strike_component_unknown_prefix() -> None:
    with pytest.raises(ValueError, match=r"T<number>.*B<number>"):
        parse_ticker("KXHIGHDEN-26MAY06-X70")


def test_event_yes_sum_ok_exact_one() -> None:
    prices = [
        Decimal("0.05"),
        Decimal("0.10"),
        Decimal("0.30"),
        Decimal("0.30"),
        Decimal("0.20"),
        Decimal("0.05"),
    ]
    assert event_yes_sum_ok(prices) is True


def test_event_yes_sum_ok_below_tolerance_warns(caplog: pytest.LogCaptureFixture) -> None:
    prices = [Decimal("0.20")] * 4 + [Decimal("0.06"), Decimal("0.06")]
    assert sum(prices) == Decimal("0.92")
    caplog.set_level(logging.WARNING, logger="bot.markets.parser")
    result = event_yes_sum_ok(prices)
    assert result is False
    assert any("yes sum off" in r.getMessage() for r in caplog.records)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_event_yes_sum_ok_at_upper_edge() -> None:
    prices = [Decimal("0.20")] * 5 + [Decimal("0.04")]
    assert sum(prices) == Decimal("1.04")
    assert event_yes_sum_ok(prices) is True


def test_event_yes_sum_ok_custom_tolerance(caplog: pytest.LogCaptureFixture) -> None:
    prices = [Decimal("0.20")] * 4 + [Decimal("0.06"), Decimal("0.06")]
    caplog.set_level(logging.WARNING, logger="bot.markets.parser")
    result = event_yes_sum_ok(prices, tolerance=Decimal("0.10"))
    assert result is True
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.parametrize(
    "ticker",
    [
        "KXHIGHDEN-26APR28-T70.5-72.5",
        "KXHIGHDEN-26APR28-T70.5",
        "KXLOWNY-26MAR15-T35.5-37.5",
        "KXHIGHDEN-26MAY19-B86.5",
        "KXHIGHTNOLA-26MAY19-B86.5",
        "KXRAINSFOM-26JUN-T1.0",
    ],
)
def test_event_id_matches_two_part_prefix(ticker: str) -> None:
    assert event_id(ticker) == "-".join(ticker.split("-")[:2])


@pytest.mark.parametrize(
    "ticker",
    [
        "KXHIGHDEN-26APR28-T70.5-72.5",
        "KXHIGHDEN-26APR28-T70.5",
        "KXLOWNY-26MAR15-T35.5-37.5",
        "KXHIGHDEN-26MAY19-B86.5",
        "KXHIGHTNOLA-26MAY19-B86.5",
        "KXRAINSFOM-26JUN-T1.0",
    ],
)
def test_series_id_returns_kx_prefix(ticker: str) -> None:
    assert series_id(ticker) == ticker.split("-")[0]


def test_event_id_raises_on_malformed() -> None:
    with pytest.raises(ValueError):
        event_id("NOT-A-REAL-TICKER")


def test_series_id_raises_on_malformed() -> None:
    with pytest.raises(ValueError):
        series_id("NOT-A-REAL-TICKER")
