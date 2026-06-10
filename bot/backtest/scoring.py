from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal

from sklearn.metrics import log_loss

from bot.backtest.engine import BacktestOrder, ReplayFailure
from bot.backtest.pnl import BacktestFill
from bot.execution.paper import TradeSide
from bot.markets.parser import parse_ticker
from bot.validation.scoring import BRIER_QUANTUM, RATE_QUANTUM, brier_score


@dataclass(frozen=True, slots=True)
class OrderSettlement:
    result: str
    market_mid: Decimal


@dataclass(frozen=True, slots=True)
class MetricRow:
    brier: Decimal
    log_loss: Decimal
    hit_rate: Decimal
    net_pnl: Decimal
    baseline_brier: Decimal
    beats_baseline: bool


@dataclass(frozen=True, slots=True)
class GroupScore:
    city: str
    lead_bucket: str
    month: str
    n_orders: int
    metrics: MetricRow


@dataclass(frozen=True, slots=True)
class CityScore:
    city: str
    n_orders: int
    skips: dict[str, int]
    metrics: MetricRow | None


@dataclass(frozen=True, slots=True)
class ScoreReport:
    groups: list[GroupScore]
    cities: list[CityScore]


@dataclass(frozen=True, slots=True)
class _Row:
    city: str
    lead_bucket: str
    month: str
    prediction: Decimal
    mid: Decimal
    outcome: int
    won: bool
    net_pnl: Decimal


def _skip_label(failure: ReplayFailure) -> str:
    if failure.layer == "routing" and failure.name == "tails_not_invoked_blacklisted":
        return "tails_blacklisted"
    if (
        failure.layer == "strategy"
        and failure.strategy == "edge"
        and failure.reason == "blacklisted"
    ):
        return "edge_blacklisted"
    return f"{failure.layer}:{failure.name}"


def _metric_row(rows: list[_Row]) -> MetricRow:
    predictions = [r.prediction for r in rows]
    mids = [r.mid for r in rows]
    outcomes = [r.outcome for r in rows]
    model_brier = brier_score(predictions, outcomes)
    baseline_brier = brier_score(mids, outcomes)
    raw_log_loss = log_loss(outcomes, [float(p) for p in predictions], labels=[0, 1])
    wins = sum(1 for r in rows if r.won)
    return MetricRow(
        brier=model_brier,
        log_loss=Decimal(str(raw_log_loss)).quantize(BRIER_QUANTUM),
        hit_rate=(Decimal(wins) / Decimal(len(rows))).quantize(RATE_QUANTUM),
        net_pnl=sum((r.net_pnl for r in rows), Decimal("0")),
        baseline_brier=baseline_brier,
        beats_baseline=model_brier < baseline_brier,
    )


def score_run(
    orders: list[BacktestOrder],
    fills: list[BacktestFill],
    settlements: list[OrderSettlement],
    failures: list[ReplayFailure],
) -> ScoreReport:
    if len(fills) != len(orders) or len(settlements) != len(orders):
        raise ValueError(
            f"length mismatch: orders={len(orders)} fills={len(fills)} "
            f"settlements={len(settlements)}"
        )

    rows: list[_Row] = []
    for order, fill, settlement in zip(orders, fills, settlements):
        parsed = parse_ticker(order.market_ticker)
        rows.append(
            _Row(
                city=parsed.series,
                lead_bucket=order.lead_bucket,
                month=f"{parsed.event_date:%Y-%m}",
                prediction=order.fair_yes,
                mid=settlement.market_mid,
                outcome=1 if settlement.result == "yes" else 0,
                won=(settlement.result == "yes") == (order.action is TradeSide.BUY_YES),
                net_pnl=fill.net_pnl,
            )
        )

    by_group: dict[tuple[str, str, str], list[_Row]] = defaultdict(list)
    by_city: dict[str, list[_Row]] = defaultdict(list)
    for row in rows:
        by_group[(row.city, row.lead_bucket, row.month)].append(row)
        by_city[row.city].append(row)

    skips_by_city: dict[str, Counter[str]] = defaultdict(Counter)
    for failure in failures:
        city = parse_ticker(failure.market_ticker).series
        skips_by_city[city][_skip_label(failure)] += 1

    groups = [
        GroupScore(
            city=city,
            lead_bucket=lead_bucket,
            month=month,
            n_orders=len(group_rows),
            metrics=_metric_row(group_rows),
        )
        for (city, lead_bucket, month), group_rows in sorted(by_group.items())
    ]

    cities = [
        CityScore(
            city=city,
            n_orders=len(by_city.get(city, [])),
            skips=dict(skips_by_city.get(city, {})),
            metrics=_metric_row(by_city[city]) if city in by_city else None,
        )
        for city in sorted(by_city.keys() | skips_by_city.keys())
    ]

    return ScoreReport(groups=groups, cities=cities)
