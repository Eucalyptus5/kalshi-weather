from __future__ import annotations

from datetime import datetime, timedelta
from datetime import timezone as _timezone
from decimal import Decimal

from bot.lag.lock_events import LockEvent, detect_lock_events
from bot.markets.parser import ParsedTicker, parse_ticker
from bot.observations.metar import StationObservation


UTC = _timezone.utc


def _obs(
    station: str,
    t_utc: datetime,
    temp_f: Decimal | str | int,
    *,
    pub_offset_s: int = 0,
) -> StationObservation:
    return StationObservation(
        station=station,
        valid_time=t_utc,
        publication_time=t_utc + timedelta(seconds=pub_offset_s),
        temp_f=Decimal(str(temp_f)),
        is_special=False,
        raw="",
        source="iem_1min_asos_archive",
    )


def test_clean_above_yes_lock() -> None:
    market = parse_ticker("KXHIGHDEN-26JUN17-T85")
    obs = [
        _obs("KDEN", datetime(2026, 6, 17, 18, 0, tzinfo=UTC), 80),
        _obs("KDEN", datetime(2026, 6, 17, 19, 0, tzinfo=UTC), 84),
        _obs("KDEN", datetime(2026, 6, 17, 20, 0, tzinfo=UTC), 87),
    ]

    events = detect_lock_events(market, obs, tz_name="America/Denver")

    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, LockEvent)
    assert ev.ticker == "KXHIGHDEN-26JUN17-T85"
    assert ev.side_locked == "yes"
    assert ev.t0 == obs[2].publication_time
    assert ev.strike == Decimal("85")
    assert ev.crossing_temp_f == Decimal("87")
    assert ev.lock_ambiguous is False


def test_ambiguous_above_yes_lock_pins_first_event() -> None:
    market = parse_ticker("KXHIGHDEN-26JUN17-T85")
    obs = [
        _obs("KDEN", datetime(2026, 6, 17, 18, 0, tzinfo=UTC), 80),
        _obs("KDEN", datetime(2026, 6, 17, 19, 0, tzinfo=UTC), 84),
        _obs("KDEN", datetime(2026, 6, 17, 20, 0, tzinfo=UTC), 85),
        _obs("KDEN", datetime(2026, 6, 17, 21, 0, tzinfo=UTC), 88),
    ]

    events = detect_lock_events(market, obs, tz_name="America/Denver")

    assert len(events) == 1
    ev = events[0]
    assert ev.t0 == obs[2].publication_time
    assert ev.crossing_temp_f == Decimal("85")
    assert ev.lock_ambiguous is True


def test_no_cross_day_returns_empty() -> None:
    market = parse_ticker("KXHIGHDEN-26JUN17-T85")
    obs = [
        _obs("KDEN", datetime(2026, 6, 17, 18, 0, tzinfo=UTC), 60),
        _obs("KDEN", datetime(2026, 6, 17, 19, 0, tzinfo=UTC), 70),
        _obs("KDEN", datetime(2026, 6, 17, 20, 0, tzinfo=UTC), 80),
    ]

    assert detect_lock_events(market, obs, tz_name="America/Denver") == []


def test_bracket_clean_no_lock() -> None:
    market = parse_ticker("KXHIGHCHI-26JUN17-T75-80")
    obs = [
        _obs("KMDW", datetime(2026, 6, 17, 18, 0, tzinfo=UTC), 70),
        _obs("KMDW", datetime(2026, 6, 17, 19, 0, tzinfo=UTC), 76),
        _obs("KMDW", datetime(2026, 6, 17, 20, 0, tzinfo=UTC), 82),
    ]

    events = detect_lock_events(market, obs, tz_name="America/Chicago")

    assert len(events) == 1
    ev = events[0]
    assert ev.side_locked == "no"
    assert ev.strike == Decimal("80")
    assert ev.crossing_temp_f == Decimal("82")
    assert ev.lock_ambiguous is False


def test_bracket_ambiguous_no_lock_not_promoted() -> None:
    market = parse_ticker("KXHIGHCHI-26JUN17-T75-80")
    obs = [
        _obs("KMDW", datetime(2026, 6, 17, 18, 0, tzinfo=UTC), 70),
        _obs("KMDW", datetime(2026, 6, 17, 19, 0, tzinfo=UTC), 76),
        _obs("KMDW", datetime(2026, 6, 17, 20, 0, tzinfo=UTC), "80.5"),
        _obs("KMDW", datetime(2026, 6, 17, 21, 0, tzinfo=UTC), 90),
    ]

    events = detect_lock_events(market, obs, tz_name="America/Chicago")

    assert len(events) == 1
    ev = events[0]
    assert ev.t0 == obs[2].publication_time
    assert ev.crossing_temp_f == Decimal("80.5")
    assert ev.lock_ambiguous is True


def test_bracket_in_bracket_running_max_excluded() -> None:
    market = parse_ticker("KXHIGHCHI-26JUN17-T75-80")
    obs = [
        _obs("KMDW", datetime(2026, 6, 17, 18, 0, tzinfo=UTC), 70),
        _obs("KMDW", datetime(2026, 6, 17, 19, 0, tzinfo=UTC), 76),
        _obs("KMDW", datetime(2026, 6, 17, 20, 0, tzinfo=UTC), 78),
    ]

    assert detect_lock_events(market, obs, tz_name="America/Chicago") == []


def test_bracket_below_low_returns_empty() -> None:
    market = parse_ticker("KXHIGHCHI-26JUN17-T75-80")
    obs = [
        _obs("KMDW", datetime(2026, 6, 17, 18, 0, tzinfo=UTC), 70),
        _obs("KMDW", datetime(2026, 6, 17, 19, 0, tzinfo=UTC), 72),
        _obs("KMDW", datetime(2026, 6, 17, 20, 0, tzinfo=UTC), 74),
    ]

    assert detect_lock_events(market, obs, tz_name="America/Chicago") == []


def test_dst_boundary_window_filters_correctly() -> None:
    market = parse_ticker("KXHIGHDEN-26MAR08-T50")
    in_window = _obs("KDEN", datetime(2026, 3, 9, 6, 30, tzinfo=UTC), 52)
    after_window = _obs("KDEN", datetime(2026, 3, 9, 7, 30, tzinfo=UTC), 60)

    events_in = detect_lock_events(market, [in_window], tz_name="America/Denver")
    assert len(events_in) == 1
    assert events_in[0].t0 == in_window.publication_time
    assert events_in[0].crossing_temp_f == Decimal("52")
    assert events_in[0].lock_ambiguous is False

    events_out = detect_lock_events(market, [after_window], tz_name="America/Denver")
    assert events_out == []


def test_event_decimal_types_strict() -> None:
    market = parse_ticker("KXHIGHDEN-26JUN17-T85")
    obs = [_obs("KDEN", datetime(2026, 6, 17, 20, 0, tzinfo=UTC), 87)]

    events = detect_lock_events(market, obs, tz_name="America/Denver")
    assert len(events) == 1
    ev = events[0]
    assert type(ev.strike) is Decimal
    assert type(ev.crossing_temp_f) is Decimal


def test_empty_obs_list_returns_empty() -> None:
    market = parse_ticker("KXHIGHDEN-26JUN17-T85")
    assert detect_lock_events(market, [], tz_name="America/Denver") == []


def test_all_obs_outside_window_returns_empty() -> None:
    market = parse_ticker("KXHIGHDEN-26JUN17-T85")
    obs = [
        _obs("KDEN", datetime(2026, 6, 17, 6, 0, tzinfo=UTC), 95),
        _obs("KDEN", datetime(2026, 6, 18, 7, 0, tzinfo=UTC), 95),
    ]

    assert detect_lock_events(market, obs, tz_name="America/Denver") == []


def _below_market(strike: str = "80") -> ParsedTicker:
    return parse_ticker(f"KXHIGHDEN-26JUN17-T{strike}").model_copy(update={"kind": "below"})


def test_below_tail_clean_no_lock() -> None:
    market = _below_market("80")
    obs = [
        _obs("KDEN", datetime(2026, 6, 17, 18, 0, tzinfo=UTC), 60),
        _obs("KDEN", datetime(2026, 6, 17, 19, 0, tzinfo=UTC), 75),
        _obs("KDEN", datetime(2026, 6, 17, 20, 0, tzinfo=UTC), 82),
    ]

    events = detect_lock_events(market, obs, tz_name="America/Denver")

    assert len(events) == 1
    ev = events[0]
    assert ev.side_locked == "no"
    assert ev.t0 == obs[2].publication_time
    assert ev.strike == Decimal("80")
    assert ev.crossing_temp_f == Decimal("82")
    assert ev.lock_ambiguous is False


def test_below_tail_ambiguous_no_lock() -> None:
    market = _below_market("80")
    obs = [
        _obs("KDEN", datetime(2026, 6, 17, 18, 0, tzinfo=UTC), 60),
        _obs("KDEN", datetime(2026, 6, 17, 19, 0, tzinfo=UTC), 75),
        _obs("KDEN", datetime(2026, 6, 17, 20, 0, tzinfo=UTC), "80.5"),
        _obs("KDEN", datetime(2026, 6, 17, 21, 0, tzinfo=UTC), 90),
    ]

    events = detect_lock_events(market, obs, tz_name="America/Denver")

    assert len(events) == 1
    ev = events[0]
    assert ev.t0 == obs[2].publication_time
    assert ev.crossing_temp_f == Decimal("80.5")
    assert ev.lock_ambiguous is True


def test_below_tail_no_event_when_max_at_or_below_strike() -> None:
    market = _below_market("80")
    obs = [
        _obs("KDEN", datetime(2026, 6, 17, 18, 0, tzinfo=UTC), 60),
        _obs("KDEN", datetime(2026, 6, 17, 19, 0, tzinfo=UTC), 75),
        _obs("KDEN", datetime(2026, 6, 17, 20, 0, tzinfo=UTC), 80),
    ]

    assert detect_lock_events(market, obs, tz_name="America/Denver") == []


def test_below_tail_decimal_strict() -> None:
    market = _below_market("80")
    obs = [_obs("KDEN", datetime(2026, 6, 17, 20, 0, tzinfo=UTC), 82)]

    events = detect_lock_events(market, obs, tz_name="America/Denver")
    assert len(events) == 1
    ev = events[0]
    assert type(ev.strike) is Decimal
    assert type(ev.crossing_temp_f) is Decimal


def test_single_event_rule_back_to_back_crossings() -> None:
    market = parse_ticker("KXHIGHDEN-26JUN17-T85")
    obs = [
        _obs("KDEN", datetime(2026, 6, 17, 20, 0, tzinfo=UTC), 85),
        _obs("KDEN", datetime(2026, 6, 17, 20, 1, tzinfo=UTC), 87),
    ]

    events = detect_lock_events(market, obs, tz_name="America/Denver")
    assert len(events) == 1
    assert events[0].t0 == obs[0].publication_time
    assert events[0].crossing_temp_f == Decimal("85")
    assert events[0].lock_ambiguous is True
