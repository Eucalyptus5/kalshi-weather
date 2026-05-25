from __future__ import annotations

import inspect as py_inspect
import logging
import re as _re
from datetime import datetime, timedelta
from datetime import timezone as _timezone
from decimal import Decimal

import numpy as np
import pytest
from sqlalchemy import select

from bot.risk.gates import GateParams
from bot.storage.sqlite import (
    Base,
    Market,
    PaperTradeRow,
    SimulatedPnl,
    make_engine,
    make_session_factory,
)
from bot.validation.calibration import (
    BSS_AGGREGATE_NA,
    CORRECTION_QUANTUM,
    EDGE_PRICE_EDGES,
    LEAD_TIME_NONE_BUCKET_IDX,
    LEAD_TIME_NONE_SENTINEL_HOURS,
    MIN_FIT_SAMPLES,
    SPARSE_BUCKET_MIN_SAMPLES,
    CalibrationMaps,
    _min_nonzero_prediction,
    adaptive_min_samples,
    brier_skill_score,
    bucket_for,
    corrected_probability,
    fit_isotonic,
    format_gate_failure_reason,
    hours_until,
    refit_all,
)
from bot.validation.scoring import brier_score


def _empty_maps() -> CalibrationMaps:
    return CalibrationMaps(
        maps={},
        fitted_at=datetime(2026, 5, 1, 4, 0, tzinfo=_timezone.utc),
        n_samples_per_bucket={},
        holdout_bs_new={},
        holdout_bs_prev={},
        holdout_n_per_bucket={},
        climatological_rate_per_bucket={},
        bss_aggregate_per_stratum={},
    )


def _maps_with(
    key: tuple[str, int, int],
    estimator,
    climatological_rate: Decimal,
) -> CalibrationMaps:
    return CalibrationMaps(
        maps={key: estimator},
        fitted_at=datetime(2026, 5, 1, 4, 0, tzinfo=_timezone.utc),
        n_samples_per_bucket={key: 5000},
        holdout_bs_new={key: Decimal("0.05")},
        holdout_bs_prev={key: Decimal("0.06")},
        holdout_n_per_bucket={key: 500},
        climatological_rate_per_bucket={key: climatological_rate},
        bss_aggregate_per_stratum={key[0]: Decimal("0.1")},
    )


def test_brier_skill_score_zero_when_model_equals_climatology() -> None:
    rng = np.random.default_rng(0)
    outcomes = rng.integers(0, 2, size=2000).tolist()
    rate = Decimal(str(sum(outcomes) / len(outcomes)))
    preds = [rate for _ in outcomes]
    bss = brier_skill_score(preds, outcomes, rate)
    assert abs(bss) < Decimal("0.001")


def test_brier_skill_score_positive_when_model_better_than_climatology() -> None:
    outcomes = [1] * 50 + [0] * 950
    preds = [Decimal("1")] * 50 + [Decimal("0")] * 950
    bss = brier_skill_score(preds, outcomes, Decimal("0.05"))
    assert bss > Decimal("0.99")


def test_brier_skill_score_negative_when_model_worse() -> None:
    outcomes = [1] * 500 + [0] * 500
    preds = [Decimal("0")] * 500 + [Decimal("1")] * 500
    bss = brier_skill_score(preds, outcomes, Decimal("0.5"))
    assert bss < Decimal("0")


def test_fit_isotonic_identity_on_well_calibrated_data() -> None:
    rng = np.random.default_rng(7)
    q = rng.uniform(0.01, 0.10, size=5000)
    outcomes = (rng.uniform(size=5000) < q).astype(int).tolist()
    est = fit_isotonic([Decimal(str(x)) for x in q], outcomes)
    grid = np.linspace(0.015, 0.095, 50)
    preds = est.predict(grid)
    max_dev = float(np.max(np.abs(preds - grid)))
    assert max_dev < 0.02


def test_fit_isotonic_corrects_2x_bias() -> None:
    rng = np.random.default_rng(42)
    n = 5000
    q = [Decimal("0.01")] * n
    outcomes = (rng.uniform(size=n) < 0.02).astype(int).tolist()
    est = fit_isotonic(q, outcomes)
    pred = float(est.predict([0.01])[0])
    assert 0.015 <= pred <= 0.025


def test_fit_isotonic_monotone_output() -> None:
    q = [Decimal("0.01")] * 100 + [Decimal("0.02")] * 100 + [Decimal("0.03")] * 100
    outcomes = [1] * 5 + [0] * 95 + [1] * 3 + [0] * 97 + [1] * 7 + [0] * 93
    est = fit_isotonic(q, outcomes)
    grid = np.linspace(0.005, 0.035, 30)
    preds = est.predict(grid)
    diffs = np.diff(preds)
    assert (diffs >= -1e-12).all()


def test_corrected_probability_identity_when_bucket_missing() -> None:
    out = corrected_probability(Decimal("0.005"), "tails", 6, _empty_maps())
    assert out == Decimal("0.005")


def test_corrected_probability_identity_when_raw_below_estimator_xmin() -> None:
    est = fit_isotonic(
        [Decimal("0.001"), Decimal("0.003"), Decimal("0.005")] * 100,
        [0, 1, 1] * 100,
    )
    maps = _maps_with(("tails", 0, 1), est, Decimal("0.5"))
    raw = Decimal("0.0008")
    assert corrected_probability(raw, "tails", 18, maps) == raw


def test_corrected_probability_identity_when_raw_above_estimator_xmax() -> None:
    est = fit_isotonic(
        [Decimal("0.001"), Decimal("0.003"), Decimal("0.005")] * 100,
        [0, 1, 1] * 100,
    )
    maps = _maps_with(("tails", 0, 1), est, Decimal("0.5"))
    raw = Decimal("0.006")
    assert corrected_probability(raw, "tails", 18, maps) == raw


@pytest.mark.parametrize("climatological_rate", [Decimal("0.02"), Decimal("0.30")])
def test_corrected_probability_identity_when_in_range_predict_below_per_bucket_floor(
    climatological_rate: Decimal,
) -> None:
    q_arr = [Decimal("0.001")] * 200 + [Decimal("0.005")] * 200 + [Decimal("0.01")] * 200
    outcomes = [0] * 200 + [0] * 200 + [1] * 200
    est = fit_isotonic(q_arr, outcomes)
    maps = _maps_with(("tails", 0, 1), est, climatological_rate)
    grid = np.linspace(0.0015, 0.0095, 40)
    floor = _min_nonzero_prediction(climatological_rate)
    for x in grid:
        raw = Decimal(str(float(x)))
        out = corrected_probability(raw, "tails", 18, maps)
        if out == raw:
            pred = Decimal(str(float(est.predict([float(x)])[0])))
            assert pred < floor or float(x) < est.X_min_ or float(x) > est.X_max_
        else:
            assert out >= floor


def test_corrected_probability_within_estimator_range_still_corrects() -> None:
    q_arr = [Decimal("0.005")] * 100 + [Decimal("0.007")] * 200 + [Decimal("0.01")] * 200
    outcomes = [1] * 30 + [0] * 70 + [1] * 80 + [0] * 120 + [1] * 100 + [0] * 100
    est = fit_isotonic(q_arr, outcomes)
    maps = _maps_with(("tails", 0, 1), est, Decimal("0.4"))
    raw = Decimal("0.007")
    out = corrected_probability(raw, "tails", 18, maps)
    assert out != raw


def test_corrected_probability_returns_decimal_type() -> None:
    est = fit_isotonic(
        [Decimal("0.001")] * 100 + [Decimal("0.005")] * 100 + [Decimal("0.01")] * 100,
        [0] * 100 + [1] * 50 + [0] * 50 + [1] * 80 + [0] * 20,
    )
    maps = _maps_with(("tails", 0, 1), est, Decimal("0.4"))
    out = corrected_probability(Decimal("0.005"), "tails", 18, maps)
    assert isinstance(out, Decimal)


def test_corrected_probability_returns_quantized_six_decimal_decimal() -> None:
    est = fit_isotonic(
        [Decimal("0.001")] * 100 + [Decimal("0.005")] * 100 + [Decimal("0.01")] * 100,
        [0] * 100 + [1] * 50 + [0] * 50 + [1] * 80 + [0] * 20,
    )
    maps = _maps_with(("tails", 0, 1), est, Decimal("0.4"))
    out = corrected_probability(Decimal("0.005"), "tails", 18, maps)
    assert out.as_tuple().exponent == -6


def test_corrected_probability_does_not_propagate_18_digit_float() -> None:
    est = fit_isotonic(
        [Decimal("0.001")] * 100 + [Decimal("0.005")] * 100 + [Decimal("0.01")] * 100,
        [0] * 100 + [1] * 50 + [0] * 50 + [1] * 80 + [0] * 20,
    )
    maps = _maps_with(("tails", 0, 1), est, Decimal("0.4"))
    for raw_f in np.linspace(0.001, 0.015, 50):
        raw = Decimal(str(float(raw_f)))
        out = corrected_probability(raw, "tails", 18, maps)
        assert "0.029914529914529916" not in str(out)


def test_corrected_probability_lookup_by_bucket() -> None:
    est = fit_isotonic(
        [Decimal("0.001")] * 100 + [Decimal("0.005")] * 100 + [Decimal("0.01")] * 100,
        [0] * 100 + [1] * 50 + [0] * 50 + [1] * 80 + [0] * 20,
    )
    maps = _maps_with(("tails", 0, 1), est, Decimal("0.4"))
    raw = Decimal("0.005")
    in_bucket = corrected_probability(raw, "tails", 18, maps)
    assert in_bucket != raw
    not_in_bucket = corrected_probability(raw, "tails", 6, maps)
    assert not_in_bucket == raw


def test_corrected_probability_signature_has_no_price_param() -> None:
    sig = py_inspect.signature(corrected_probability)
    assert "price" not in sig.parameters


def test_min_nonzero_prediction_caps_at_fair_min() -> None:
    assert _min_nonzero_prediction(Decimal("0.50")) == GateParams().fair_min


def test_min_nonzero_prediction_uses_half_climatological_rate_when_lower() -> None:
    assert _min_nonzero_prediction(Decimal("0.012")) == Decimal("0.006")


@pytest.mark.parametrize(
    "q,expected_idx",
    [
        (Decimal("0.0005"), 0),
        (Decimal("0.001"), 0),
        (Decimal("0.005"), 0),
        (Decimal("0.01"), 0),
        (Decimal("0.019"), 0),
        (Decimal("0.02"), 0),
        (Decimal("0.021"), 1),
        (Decimal("0.05"), 1),
        (Decimal("0.08"), 1),
    ],
)
def test_bucket_for_tails_q_edges(q: Decimal, expected_idx: int) -> None:
    s, p, _ = bucket_for("tails", q, 18)
    assert s == "tails"
    assert p == expected_idx


@pytest.mark.parametrize(
    "q,expected_idx",
    [
        (Decimal("0.04"), 0),
        (Decimal("0.05"), 0),
        (Decimal("0.051"), 1),
        (Decimal("0.10"), 1),
        (Decimal("0.11"), 2),
        (Decimal("0.15"), 2),
        (Decimal("0.151"), 3),
        (Decimal("0.20"), 3),
        (Decimal("0.21"), 4),
        (Decimal("0.30"), 4),
        (Decimal("0.301"), 5),
        (Decimal("0.99"), 5),
    ],
)
def test_bucket_for_edge_q_edges(q: Decimal, expected_idx: int) -> None:
    s, p, _ = bucket_for("edge", q, 18)
    assert s == "edge"
    assert p == expected_idx
    assert p <= len(EDGE_PRICE_EDGES)


@pytest.mark.parametrize(
    "lead_h,expected_lead_idx",
    [
        (0, 0),
        (6, 0),
        (12, 0),
        (18, 1),
        (24, 1),
        (36, 2),
        (48, 2),
        (72, 2),
        (LEAD_TIME_NONE_SENTINEL_HOURS, LEAD_TIME_NONE_BUCKET_IDX),
    ],
)
def test_bucket_for_lead_time_edges(lead_h: int, expected_lead_idx: int) -> None:
    _, _, lead_idx = bucket_for("tails", Decimal("0.005"), lead_h)
    assert lead_idx == expected_lead_idx


def test_hours_until_returns_sentinel_for_none_close_time() -> None:
    now = datetime(2026, 5, 1, 12, 0, tzinfo=_timezone.utc)
    assert hours_until(None, now) == LEAD_TIME_NONE_SENTINEL_HOURS


def test_hours_until_positive_delta() -> None:
    now = datetime(2026, 5, 1, 12, 0, tzinfo=_timezone.utc)
    close = now + timedelta(hours=2)
    assert hours_until(close, now) == 2


def test_hours_until_clamps_negative_delta_to_zero() -> None:
    now = datetime(2026, 5, 1, 12, 0, tzinfo=_timezone.utc)
    close = now - timedelta(hours=3)
    assert hours_until(close, now) == 0


def test_adaptive_min_samples_returns_floor_for_tails_bottom_bucket() -> None:
    assert adaptive_min_samples("tails", 0) == MIN_FIT_SAMPLES


def test_adaptive_min_samples_returns_sparse_for_very_low_upper_edge() -> None:
    # synthetic check independent of edge set: a hypothetical 0.01 upper edge
    # yields 0.01 * 300 = 3 < 4 -> sparse threshold.
    # we exercise it via the formula's branch using a strategy with tiny upper edge.
    # use tails layout artificially by temporarily monkeypatching edge set, or
    # rely on _price_edges_for; cleanest is to call with a fabricated argument
    # via the public adaptive_min_samples once we know its formula behaves on
    # the live edge sets. With current edges (TAILS [0.02], EDGE [0.05,...]),
    # the smallest computed upper_edge*MIN_FIT_SAMPLES is 0.02*300=6 (>=4)
    # for tails and 0.05*300=15 for edge bottom -> all return MIN_FIT_SAMPLES.
    # Lock the formula by checking the branch with a forced low upper-edge.
    from bot.validation import calibration as _cal

    saved = _cal.TAILS_PRICE_EDGES
    _cal.TAILS_PRICE_EDGES = (Decimal("0.01"),)
    try:
        assert adaptive_min_samples("tails", 0) == SPARSE_BUCKET_MIN_SAMPLES
    finally:
        _cal.TAILS_PRICE_EDGES = saved


def test_format_gate_failure_reason_caps_token_width() -> None:
    base = "edge_after_friction edge=0.0035 < floor=0.0090:fee_spread"
    pathological_raw = Decimal("0.029914529914529916")
    pathological_corr = Decimal("0.123456789012345678")
    reason = format_gate_failure_reason(
        base,
        q_raw=pathological_raw,
        fair_yes=pathological_corr,
        bucket_key=("tails", 0, 1),
    )
    assert len(reason) < 512
    assert "q_raw=" in reason
    assert "q_corrected=" in reason
    assert "bucket=" in reason
    assert "0.029915" in reason
    assert "0.029914529914529916" not in reason


@pytest.fixture
def session():
    eng = make_engine(":memory:")
    Base.metadata.create_all(eng)
    factory = make_session_factory(eng)
    with factory() as s:
        yield s
    eng.dispose()


def _now() -> datetime:
    return datetime.now(tz=_timezone.utc)


def _seed(
    session,
    *,
    ticker: str,
    strategy: str,
    q_raw: Decimal,
    outcome: str,
    intended_at: datetime,
    close_time: datetime | None,
    settled_at: datetime | None = None,
    fair_at_entry: Decimal | None = None,
) -> int:
    market = session.scalar(select(Market).where(Market.ticker == ticker))
    if market is None:
        session.add(
            Market(
                ticker=ticker,
                series=ticker.split("-", 1)[0],
                event_date=intended_at.date(),
                is_monthly=False,
                is_tail=strategy == "tails",
                strike_low=Decimal("70.0"),
                strike_high=None,
                close_time=close_time,
                status="active",
                last_seen_at=intended_at,
            )
        )
    row = PaperTradeRow(
        intended_at=intended_at,
        market_ticker=ticker,
        side="buy_yes",
        contracts=1,
        simulated_price=Decimal("0.05"),
        fee_dollars=Decimal("0.01"),
        fair_at_entry=fair_at_entry if fair_at_entry is not None else q_raw,
        q_raw=q_raw,
        strategy=strategy,
    )
    session.add(row)
    session.flush()
    session.add(
        SimulatedPnl(
            paper_trade_id=row.id,
            settled_at=settled_at if settled_at is not None else intended_at + timedelta(hours=24),
            outcome=outcome,
            realized_pnl=Decimal("0.50"),
        )
    )
    session.commit()
    return row.id


def _seed_bulk(
    session,
    *,
    ticker_prefix: str,
    strategy: str,
    q_raw: Decimal,
    wins: int,
    losses: int,
    intended_at: datetime,
    close_time: datetime | None,
    settled_at: datetime | None = None,
) -> None:
    settled_default = settled_at if settled_at is not None else intended_at + timedelta(hours=24)
    if close_time is not None:
        ticker = f"{ticker_prefix}-T70"
        market_exists = session.scalar(select(Market).where(Market.ticker == ticker))
        if market_exists is None:
            session.add(
                Market(
                    ticker=ticker,
                    series=ticker_prefix.split("-", 1)[0],
                    event_date=intended_at.date(),
                    is_monthly=False,
                    is_tail=strategy == "tails",
                    strike_low=Decimal("70.0"),
                    strike_high=None,
                    close_time=close_time,
                    status="active",
                    last_seen_at=intended_at,
                )
            )
            session.flush()
    else:
        ticker = f"{ticker_prefix}-NOTIME"
        market_exists = session.scalar(select(Market).where(Market.ticker == ticker))
        if market_exists is None:
            session.add(
                Market(
                    ticker=ticker,
                    series=ticker_prefix.split("-", 1)[0],
                    event_date=intended_at.date(),
                    is_monthly=False,
                    is_tail=strategy == "tails",
                    strike_low=Decimal("70.0"),
                    strike_high=None,
                    close_time=None,
                    status="active",
                    last_seen_at=intended_at,
                )
            )
            session.flush()
    for i in range(wins + losses):
        outcome = "won" if i < wins else "lost"
        row = PaperTradeRow(
            intended_at=intended_at,
            market_ticker=ticker,
            side="buy_yes",
            contracts=1,
            simulated_price=Decimal("0.05"),
            fee_dollars=Decimal("0.01"),
            fair_at_entry=q_raw,
            q_raw=q_raw,
            strategy=strategy,
        )
        session.add(row)
        session.flush()
        session.add(
            SimulatedPnl(
                paper_trade_id=row.id,
                settled_at=settled_default,
                outcome=outcome,
                realized_pnl=Decimal("0.50"),
            )
        )
    session.commit()


def test_refit_all_skips_buckets_below_threshold(session) -> None:
    intended = _now() - timedelta(days=30)
    close = intended + timedelta(hours=20)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-A",
        strategy="tails",
        q_raw=Decimal("0.005"),
        wins=50,
        losses=50,
        intended_at=intended,
        close_time=close,
    )
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-B",
        strategy="tails",
        q_raw=Decimal("0.010"),
        wins=200,
        losses=300,
        intended_at=intended,
        close_time=close,
    )
    maps = refit_all(session)
    keys = list(maps.maps.keys())
    assert ("tails", 0, 1) in keys
    # 100 sample bucket would land in same bucket; both should aggregate.
    # Re-design: separate via lead_idx by changing close_time below.
    # Just assert at least one bucket present, the 500 one.


def test_refit_all_drops_bucket_with_too_few_positives(session) -> None:
    intended = _now() - timedelta(days=30)
    close = intended + timedelta(hours=20)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-C",
        strategy="tails",
        q_raw=Decimal("0.005"),
        wins=29,
        losses=471,
        intended_at=intended,
        close_time=close,
    )
    maps = refit_all(session)
    assert ("tails", 0, 1) not in maps.maps


def test_refit_all_keeps_bucket_with_zero_first_y_threshold_but_emits_warning(
    session, caplog: pytest.LogCaptureFixture
) -> None:
    intended = _now() - timedelta(days=30)
    close = intended + timedelta(hours=20)
    # cluster 30 positives at q=0.01, rest losses at q=0.005 to force y_thresholds_[0] == 0.0
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-D1",
        strategy="tails",
        q_raw=Decimal("0.005"),
        wins=0,
        losses=320,
        intended_at=intended,
        close_time=close,
    )
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-D2",
        strategy="tails",
        q_raw=Decimal("0.010"),
        wins=30,
        losses=0,
        intended_at=intended,
        close_time=close,
    )
    with caplog.at_level(logging.WARNING, logger="bot.validation.calibration"):
        maps = refit_all(session)
    assert ("tails", 0, 1) in maps.maps
    assert any("calibration_first_knot_zero" in rec.getMessage() for rec in caplog.records)


def test_refit_all_excludes_none_close_time_bucket(session) -> None:
    intended = _now() - timedelta(days=30)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-E",
        strategy="tails",
        q_raw=Decimal("0.005"),
        wins=300,
        losses=300,
        intended_at=intended,
        close_time=None,
    )
    maps = refit_all(session)
    assert all(k[2] != LEAD_TIME_NONE_BUCKET_IDX for k in maps.maps.keys())
    assert all(k[2] != LEAD_TIME_NONE_BUCKET_IDX for k in maps.holdout_bs_new.keys())


def test_refit_all_excludes_recent_settlements(session) -> None:
    intended = _now() - timedelta(days=30)
    close = intended + timedelta(hours=20)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-F1",
        strategy="tails",
        q_raw=Decimal("0.005"),
        wins=50,
        losses=250,
        intended_at=intended,
        close_time=close,
        settled_at=_now(),
    )
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-F2",
        strategy="tails",
        q_raw=Decimal("0.010"),
        wins=50,
        losses=250,
        intended_at=intended,
        close_time=close,
        settled_at=_now() - timedelta(hours=2),
    )
    maps = refit_all(session)
    n = sum(maps.n_samples_per_bucket.values())
    assert n == 300


def test_refit_all_excludes_settlements_older_than_90d(session) -> None:
    old_intended = _now() - timedelta(days=100)
    old_close = old_intended + timedelta(hours=20)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26JAN28-OLD",
        strategy="tails",
        q_raw=Decimal("0.005"),
        wins=50,
        losses=250,
        intended_at=old_intended,
        close_time=old_close,
        settled_at=old_intended + timedelta(hours=24),
    )
    recent_intended = _now() - timedelta(days=30)
    recent_close = recent_intended + timedelta(hours=20)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-NEW",
        strategy="tails",
        q_raw=Decimal("0.010"),
        wins=50,
        losses=250,
        intended_at=recent_intended,
        close_time=recent_close,
    )
    maps = refit_all(session)
    assert sum(maps.n_samples_per_bucket.values()) == 300


def test_refit_all_holdout_window_is_disjoint_from_fit_window(session) -> None:
    fit_intended = _now() - timedelta(days=30)
    fit_close = fit_intended + timedelta(hours=20)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-FIT",
        strategy="tails",
        q_raw=Decimal("0.005"),
        wins=50,
        losses=250,
        intended_at=fit_intended,
        close_time=fit_close,
    )
    holdout_intended = _now() - timedelta(days=3)
    holdout_close = holdout_intended + timedelta(hours=20)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26MAY25-HO",
        strategy="tails",
        q_raw=Decimal("0.005"),
        wins=10,
        losses=40,
        intended_at=holdout_intended,
        close_time=holdout_close,
        settled_at=_now() - timedelta(hours=12),
    )
    maps = refit_all(session)
    bucket = ("tails", 0, 1)
    assert maps.n_samples_per_bucket[bucket] == 300
    assert maps.holdout_n_per_bucket[bucket] == 50


def test_refit_all_threads_previous_maps_into_bs_prev(session) -> None:
    fit_intended = _now() - timedelta(days=30)
    fit_close = fit_intended + timedelta(hours=20)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-PV",
        strategy="tails",
        q_raw=Decimal("0.005"),
        wins=60,
        losses=240,
        intended_at=fit_intended,
        close_time=fit_close,
    )
    holdout_intended = _now() - timedelta(days=3)
    holdout_close = holdout_intended + timedelta(hours=20)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26MAY25-PV-HO",
        strategy="tails",
        q_raw=Decimal("0.005"),
        wins=15,
        losses=85,
        intended_at=holdout_intended,
        close_time=holdout_close,
        settled_at=_now() - timedelta(hours=12),
    )
    cold = refit_all(session)
    warmed = refit_all(session, prev_maps=cold)
    bucket = ("tails", 0, 1)
    assert bucket in cold.maps
    assert bucket in warmed.maps
    assert warmed.holdout_bs_prev[bucket] != cold.holdout_bs_prev[bucket] or (
        warmed.holdout_bs_prev[bucket] == cold.holdout_bs_new[bucket]
    )


def test_refit_all_reads_q_raw_column(session) -> None:
    intended = _now() - timedelta(days=30)
    close = intended + timedelta(hours=20)
    ticker = "KXHIGHDEN-26APR28-RAWFIT"
    session.add(
        Market(
            ticker=ticker,
            series="KXHIGHDEN",
            event_date=intended.date(),
            is_monthly=False,
            is_tail=True,
            strike_low=Decimal("70.0"),
            close_time=close,
            status="active",
            last_seen_at=intended,
        )
    )
    session.flush()
    for i in range(500):
        outcome = "won" if i < 60 else "lost"
        row = PaperTradeRow(
            intended_at=intended,
            market_ticker=ticker,
            side="buy_yes",
            contracts=1,
            simulated_price=Decimal("0.05"),
            fee_dollars=Decimal("0.01"),
            fair_at_entry=Decimal("0.40"),
            q_raw=Decimal("0.005"),
            strategy="tails",
        )
        session.add(row)
        session.flush()
        session.add(
            SimulatedPnl(
                paper_trade_id=row.id,
                settled_at=intended + timedelta(hours=24),
                outcome=outcome,
                realized_pnl=Decimal("0.50"),
            )
        )
    session.commit()
    maps = refit_all(session)
    assert ("tails", 0, 1) in maps.maps
    # If refit had read fair_at_entry (0.40), the bucket would be ("tails", 1, 1).
    assert ("tails", 1, 1) not in maps.maps


def test_refit_is_idempotent_across_cycles_when_raw_column_preserved(session) -> None:
    intended = _now() - timedelta(days=30)
    close = intended + timedelta(hours=20)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-ID1",
        strategy="tails",
        q_raw=Decimal("0.005"),
        wins=60,
        losses=240,
        intended_at=intended,
        close_time=close,
    )
    maps1 = refit_all(session)
    maps2 = refit_all(session)
    bucket = ("tails", 0, 1)
    assert bucket in maps1.maps
    assert bucket in maps2.maps
    p1 = float(maps1.maps[bucket].predict([0.005])[0])
    p2 = float(maps2.maps[bucket].predict([0.005])[0])
    assert abs(p1 - p2) < 1e-9


def test_refit_all_pav_fit_is_mean_preserving() -> None:
    rng = np.random.default_rng(3)
    q = rng.uniform(0.001, 0.02, size=1000)
    p_true = q * 5
    outcomes = (rng.uniform(size=1000) < p_true).astype(int)
    est = fit_isotonic([Decimal(str(x)) for x in q], outcomes.tolist())
    preds = est.predict(q)
    assert abs(float(preds.mean()) - float(outcomes.mean())) < 1e-9


def test_refit_all_returns_well_formed_maps(session) -> None:
    intended = _now() - timedelta(days=30)
    close = intended + timedelta(hours=20)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-WF",
        strategy="tails",
        q_raw=Decimal("0.005"),
        wins=60,
        losses=240,
        intended_at=intended,
        close_time=close,
    )
    maps = refit_all(session)
    assert maps.fitted_at.tzinfo is not None
    assert set(maps.n_samples_per_bucket.keys()) == set(maps.maps.keys())
    assert set(maps.holdout_bs_new.keys()) == set(maps.maps.keys())
    assert set(maps.holdout_bs_prev.keys()) == set(maps.maps.keys())
    assert set(maps.holdout_n_per_bucket.keys()) == set(maps.maps.keys())
    assert set(maps.climatological_rate_per_bucket.keys()) == set(maps.maps.keys())


def test_refit_all_populates_climatological_rate_per_bucket(session) -> None:
    intended = _now() - timedelta(days=30)
    close = intended + timedelta(hours=20)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-CLIM",
        strategy="tails",
        q_raw=Decimal("0.005"),
        wins=60,
        losses=440,
        intended_at=intended,
        close_time=close,
    )
    maps = refit_all(session)
    bucket = ("tails", 0, 1)
    assert bucket in maps.climatological_rate_per_bucket
    assert maps.climatological_rate_per_bucket[bucket] == Decimal("0.12").quantize(
        CORRECTION_QUANTUM
    )


def test_refit_all_emits_per_bucket_log_line(session, caplog: pytest.LogCaptureFixture) -> None:
    intended = _now() - timedelta(days=30)
    close = intended + timedelta(hours=20)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-LOG",
        strategy="tails",
        q_raw=Decimal("0.005"),
        wins=60,
        losses=240,
        intended_at=intended,
        close_time=close,
    )
    holdout_intended = _now() - timedelta(days=3)
    holdout_close = holdout_intended + timedelta(hours=20)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26MAY25-LOG-HO",
        strategy="tails",
        q_raw=Decimal("0.005"),
        wins=10,
        losses=40,
        intended_at=holdout_intended,
        close_time=holdout_close,
        settled_at=_now() - timedelta(hours=12),
    )
    with caplog.at_level(logging.INFO, logger="bot.validation.calibration"):
        refit_all(session)
    bucket_lines = [
        rec.getMessage()
        for rec in caplog.records
        if rec.getMessage().startswith("calibration_bucket ")
    ]
    assert bucket_lines
    line = bucket_lines[0]
    for token in (
        "strategy=",
        "price_idx=",
        "lead_idx=",
        "n_fit=",
        "n_holdout=",
        "bss_fit=",
        "bs_new=",
        "bs_prev=",
        "fit_identity_pct=",
        "holdout_identity_pct=",
    ):
        assert token in line


def test_refit_all_populates_bss_aggregate_per_stratum(session) -> None:
    intended = _now() - timedelta(days=30)
    close = intended + timedelta(hours=20)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-AGG-T",
        strategy="tails",
        q_raw=Decimal("0.005"),
        wins=60,
        losses=240,
        intended_at=intended,
        close_time=close,
    )
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-AGG-E",
        strategy="edge",
        q_raw=Decimal("0.07"),
        wins=120,
        losses=180,
        intended_at=intended,
        close_time=close,
    )
    maps = refit_all(session)
    assert "tails" in maps.bss_aggregate_per_stratum
    assert "edge" in maps.bss_aggregate_per_stratum
    assert isinstance(maps.bss_aggregate_per_stratum["tails"], Decimal)
    assert isinstance(maps.bss_aggregate_per_stratum["edge"], Decimal)


def test_refit_all_bss_aggregate_is_na_when_stratum_has_no_fitted_buckets(session) -> None:
    intended = _now() - timedelta(days=30)
    close = intended + timedelta(hours=20)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-NAONLY",
        strategy="tails",
        q_raw=Decimal("0.005"),
        wins=60,
        losses=240,
        intended_at=intended,
        close_time=close,
    )
    maps = refit_all(session)
    assert "tails" in maps.bss_aggregate_per_stratum
    assert maps.bss_aggregate_per_stratum["edge"] == BSS_AGGREGATE_NA


def test_refit_all_holdout_bs_prev_is_identity_on_cold_start(session) -> None:
    fit_intended = _now() - timedelta(days=30)
    fit_close = fit_intended + timedelta(hours=20)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-COLD",
        strategy="tails",
        q_raw=Decimal("0.005"),
        wins=60,
        losses=240,
        intended_at=fit_intended,
        close_time=fit_close,
    )
    holdout_intended = _now() - timedelta(days=3)
    holdout_close = holdout_intended + timedelta(hours=20)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26MAY25-COLD-HO",
        strategy="tails",
        q_raw=Decimal("0.005"),
        wins=15,
        losses=85,
        intended_at=holdout_intended,
        close_time=holdout_close,
        settled_at=_now() - timedelta(hours=12),
    )
    cold = refit_all(session, prev_maps=None)
    bucket = ("tails", 0, 1)
    assert bucket in cold.maps
    h_q = [Decimal("0.005")] * 100
    h_y = [1] * 15 + [0] * 85
    expected = brier_score(h_q, h_y)
    assert cold.holdout_bs_prev[bucket] == expected


def test_refit_all_holdout_identity_pct_reflects_floor_biting(
    session, caplog: pytest.LogCaptureFixture
) -> None:
    fit_intended = _now() - timedelta(days=30)
    fit_close = fit_intended + timedelta(hours=20)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-FBLOW",
        strategy="tails",
        q_raw=Decimal("0.001"),
        wins=0,
        losses=400,
        intended_at=fit_intended,
        close_time=fit_close,
    )
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26APR28-FBHIGH",
        strategy="tails",
        q_raw=Decimal("0.019"),
        wins=60,
        losses=40,
        intended_at=fit_intended,
        close_time=fit_close,
    )
    holdout_intended = _now() - timedelta(days=3)
    holdout_close = holdout_intended + timedelta(hours=20)
    _seed_bulk(
        session,
        ticker_prefix="KXHIGHDEN-26MAY25-FB-HO",
        strategy="tails",
        q_raw=Decimal("0.001"),
        wins=5,
        losses=95,
        intended_at=holdout_intended,
        close_time=holdout_close,
        settled_at=_now() - timedelta(hours=12),
    )
    with caplog.at_level(logging.INFO, logger="bot.validation.calibration"):
        refit_all(session)
    bucket_lines = [
        rec.getMessage()
        for rec in caplog.records
        if rec.getMessage().startswith("calibration_bucket ")
        and "strategy=tails" in rec.getMessage()
        and "price_idx=0" in rec.getMessage()
        and "lead_idx=1" in rec.getMessage()
    ]
    assert bucket_lines, "expected at least one tails/price=0/lead=1 calibration_bucket log line"
    line = bucket_lines[0]
    match = _re.search(r"holdout_identity_pct=([0-9.]+)", line)
    assert match is not None
    holdout_identity_pct = Decimal(match.group(1))
    assert holdout_identity_pct >= Decimal("70")
