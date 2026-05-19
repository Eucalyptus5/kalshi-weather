from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


log = logging.getLogger(__name__)


class GateMode(Enum):
    PAPER = "paper"
    LIVE = "live"


@dataclass(frozen=True, slots=True)
class GateContext:
    fair_yes: Decimal | None
    model_age_hours: Decimal
    ensemble_spread: Decimal
    edge: Decimal
    order_size_dollars: Decimal
    market_existing_dollars: Decimal
    market_position_cap: Decimal
    event_existing_dollars: Decimal
    event_position_cap: Decimal
    series_existing_dollars: Decimal
    series_position_cap: Decimal
    aggregate_existing_dollars: Decimal
    aggregate_exposure_cap: Decimal
    account_balance: Decimal
    required_cushion: Decimal
    market_status: str
    minutes_to_close: int
    circuit_breakers_armed: bool


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    passed: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class RiskCheck:
    overall_passed: bool
    all_results: tuple[GateResult, ...]
    failures: tuple[GateResult, ...]


@dataclass(frozen=True, slots=True)
class GateParams:
    fair_min: Decimal = Decimal("0.01")
    fair_max: Decimal = Decimal("0.99")
    model_max_age_hours: Decimal = Decimal("6")
    min_ensemble_spread: Decimal = Decimal("1.0")
    min_edge: Decimal = Decimal("0.05")
    min_minutes_to_close: int = 10


DEFAULT_PARAMS: GateParams = GateParams()


TRADEABLE_STATUSES: frozenset[str] = frozenset({"active"})


GATE_NAMES: tuple[str, ...] = (
    "fair_value_sane",
    "model_fresh",
    "ensemble_spread_ok",
    "edge_threshold",
    "within_market_cap",
    "within_event_cap",
    "within_series_cap",
    "within_aggregate_cap",
    "account_cushion",
    "market_open",
    "time_to_close",
    "circuit_breakers_armed",
)


CAP_GATE_NAMES: frozenset[str] = frozenset(
    {
        "within_market_cap",
        "within_event_cap",
        "within_series_cap",
        "within_aggregate_cap",
    }
)


def _ok(name: str) -> GateResult:
    return GateResult(name=name, passed=True, reason=None)


def _fail(name: str, reason: str) -> GateResult:
    return GateResult(name=name, passed=False, reason=reason)


def _check_fair_value(fair_yes: Decimal | None, params: GateParams) -> GateResult:
    name = "fair_value_sane"
    if fair_yes is None:
        return _fail(name, "fair_yes is None")
    if fair_yes < params.fair_min or fair_yes > params.fair_max:
        return _fail(name, f"fair_yes={fair_yes} outside [{params.fair_min}, {params.fair_max}]")
    return _ok(name)


def evaluate(
    ctx: GateContext,
    mode: GateMode,
    params: GateParams = DEFAULT_PARAMS,
) -> RiskCheck:
    results: list[GateResult] = []

    results.append(_check_fair_value(ctx.fair_yes, params))

    if ctx.model_age_hours <= params.model_max_age_hours:
        results.append(_ok("model_fresh"))
    else:
        results.append(
            _fail(
                "model_fresh",
                f"model_age_hours={ctx.model_age_hours} > {params.model_max_age_hours}",
            )
        )

    if ctx.ensemble_spread >= params.min_ensemble_spread:
        results.append(_ok("ensemble_spread_ok"))
    else:
        results.append(
            _fail(
                "ensemble_spread_ok",
                f"ensemble_spread={ctx.ensemble_spread} < {params.min_ensemble_spread}",
            )
        )

    if ctx.edge >= params.min_edge:
        results.append(_ok("edge_threshold"))
    else:
        results.append(_fail("edge_threshold", f"edge={ctx.edge} < {params.min_edge}"))

    market_total = ctx.market_existing_dollars + ctx.order_size_dollars
    if market_total <= ctx.market_position_cap:
        results.append(_ok("within_market_cap"))
    else:
        results.append(
            _fail(
                "within_market_cap",
                f"market_total={market_total} > cap={ctx.market_position_cap}",
            )
        )

    event_total = ctx.event_existing_dollars + ctx.order_size_dollars
    if event_total <= ctx.event_position_cap:
        results.append(_ok("within_event_cap"))
    else:
        results.append(
            _fail(
                "within_event_cap",
                f"event_total={event_total} > cap={ctx.event_position_cap}",
            )
        )

    series_total = ctx.series_existing_dollars + ctx.order_size_dollars
    if series_total <= ctx.series_position_cap:
        results.append(_ok("within_series_cap"))
    else:
        results.append(
            _fail(
                "within_series_cap",
                f"series_total={series_total} > cap={ctx.series_position_cap}",
            )
        )

    aggregate_total = ctx.aggregate_existing_dollars + ctx.order_size_dollars
    if aggregate_total <= ctx.aggregate_exposure_cap:
        results.append(_ok("within_aggregate_cap"))
    else:
        results.append(
            _fail(
                "within_aggregate_cap",
                f"aggregate_total={aggregate_total} > cap={ctx.aggregate_exposure_cap}",
            )
        )

    if ctx.account_balance >= ctx.required_cushion:
        results.append(_ok("account_cushion"))
    else:
        results.append(
            _fail(
                "account_cushion",
                f"balance={ctx.account_balance} < cushion={ctx.required_cushion}",
            )
        )

    if ctx.market_status in TRADEABLE_STATUSES:
        results.append(_ok("market_open"))
    else:
        results.append(_fail("market_open", f"market_status={ctx.market_status}"))
        if ctx.market_status not in {"closed", "settled", "inactive", "pending_settle"}:
            log.warning("market_open_unknown_status status=%s", ctx.market_status)

    if ctx.minutes_to_close >= params.min_minutes_to_close:
        results.append(_ok("time_to_close"))
    else:
        results.append(
            _fail(
                "time_to_close",
                f"minutes_to_close={ctx.minutes_to_close} < {params.min_minutes_to_close}",
            )
        )

    if ctx.circuit_breakers_armed:
        results.append(_ok("circuit_breakers_armed"))
    else:
        results.append(_fail("circuit_breakers_armed", "circuit_breakers_armed=False"))

    failures = tuple(r for r in results if not r.passed)
    for r in failures:
        log.warning(
            "risk gate failed gate=%s reason=%s mode=%s",
            r.name,
            r.reason,
            mode.value,
        )

    overall_passed = True if mode == GateMode.PAPER else len(failures) == 0
    return RiskCheck(
        overall_passed=overall_passed,
        all_results=tuple(results),
        failures=failures,
    )
