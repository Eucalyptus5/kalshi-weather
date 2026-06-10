from datetime import datetime, timezone
from decimal import Decimal

from bot.backtest.engine import BacktestOrder, ReplayFailure
from bot.backtest.pnl import BacktestFill
from bot.backtest.scoring import OrderSettlement, score_run
from bot.execution.paper import TradeSide
from bot.validation.scoring import brier_score


_AS_OF = datetime(2025, 11, 4, 17, 0, tzinfo=timezone.utc)


def make_order(
    ticker: str,
    *,
    fair_yes: Decimal,
    action: TradeSide = TradeSide.BUY_YES,
    lead_bucket: str = "24-72h",
) -> BacktestOrder:
    return BacktestOrder(
        market_ticker=ticker,
        as_of=_AS_OF,
        strategy="edge",
        action=action,
        contracts=1,
        fair_yes=fair_yes,
        price_per_contract=Decimal("0.50"),
        order_dollars=Decimal("0.50"),
        depth_at_price=3000,
        depth_source="snapshot",
        lead_bucket=lead_bucket,
    )


def make_fill(net_pnl: Decimal) -> BacktestFill:
    return BacktestFill(gross_pnl=net_pnl, fee_dollars=Decimal("0"), net_pnl=net_pnl)


def make_failure(
    ticker: str,
    *,
    strategy: str,
    layer: str,
    name: str,
    reason: str,
) -> ReplayFailure:
    return ReplayFailure(
        market_ticker=ticker,
        as_of=_AS_OF,
        strategy=strategy,
        layer=layer,
        name=name,
        reason=reason,
    )


def golden_run() -> tuple[list[BacktestOrder], list[BacktestFill], list[OrderSettlement]]:
    preds = [Decimal("0.9"), Decimal("0.1"), Decimal("0.8"), Decimal("0.2")]
    results = ["yes", "no", "yes", "no"]
    orders = [make_order(f"KXHIGHDEN-25NOV0{i + 1}-B61.5", fair_yes=p) for i, p in enumerate(preds)]
    fills = [make_fill(Decimal("0")) for _ in orders]
    settlements = [OrderSettlement(result=r, market_mid=Decimal("0.5")) for r in results]
    return orders, fills, settlements


def test_golden_brier_matches_validation_scoring() -> None:
    orders, fills, settlements = golden_run()

    report = score_run(orders, fills, settlements, [])

    assert len(report.groups) == 1
    group = report.groups[0]
    assert group.city == "KXHIGHDEN"
    assert group.lead_bucket == "24-72h"
    assert group.month == "2025-11"
    assert group.n_orders == 4
    assert group.metrics.brier == Decimal("0.025")
    assert group.metrics.brier == brier_score(
        [Decimal("0.9"), Decimal("0.1"), Decimal("0.8"), Decimal("0.2")], [1, 0, 1, 0]
    )
    assert group.metrics.log_loss == Decimal("0.164252")


def test_market_mid_baseline_beats_baseline_flag() -> None:
    orders, fills, settlements = golden_run()

    report = score_run(orders, fills, settlements, [])

    city = report.cities[0]
    assert city.city == "KXHIGHDEN"
    assert city.metrics is not None
    assert city.metrics.baseline_brier == Decimal("0.25")
    assert city.metrics.beats_baseline is True


def test_model_worse_than_baseline_not_flagged() -> None:
    orders = [
        make_order("KXHIGHDEN-25NOV04-B61.5", fair_yes=Decimal("0.4")),
        make_order("KXHIGHDEN-25NOV05-B61.5", fair_yes=Decimal("0.6")),
    ]
    fills = [make_fill(Decimal("0")), make_fill(Decimal("0"))]
    settlements = [
        OrderSettlement(result="yes", market_mid=Decimal("0.9")),
        OrderSettlement(result="no", market_mid=Decimal("0.1")),
    ]

    report = score_run(orders, fills, settlements, [])

    city = report.cities[0]
    assert city.metrics is not None
    assert city.metrics.brier == Decimal("0.36")
    assert city.metrics.baseline_brier == Decimal("0.01")
    assert city.metrics.beats_baseline is False


def test_blacklisted_city_is_structurally_empty() -> None:
    failures = [
        make_failure(
            "KXHIGHLAX-25NOV04-B71.5",
            strategy="edge",
            layer="strategy",
            name="blacklisted",
            reason="blacklisted",
        ),
        make_failure(
            "KXHIGHLAX-25NOV04-T80",
            strategy="tails",
            layer="routing",
            name="tails_not_invoked_blacklisted",
            reason="tails_not_invoked_blacklisted",
        ),
        make_failure(
            "KXHIGHMIA-25NOV04-B88.5",
            strategy="edge",
            layer="strategy",
            name="blacklisted",
            reason="blacklisted",
        ),
        make_failure(
            "KXHIGHMIA-25NOV04-T95",
            strategy="tails",
            layer="routing",
            name="tails_not_invoked_blacklisted",
            reason="tails_not_invoked_blacklisted",
        ),
    ]

    report = score_run([], [], [], failures)

    assert report.groups == []
    assert [c.city for c in report.cities] == ["KXHIGHLAX", "KXHIGHMIA"]
    for city in report.cities:
        assert city.n_orders == 0
        assert city.metrics is None
        assert city.skips["edge_blacklisted"] == 1
        assert city.skips["tails_blacklisted"] == 1


def test_tails_blacklisted_counts_routing_records() -> None:
    failures = [
        make_failure(
            f"KXHIGHMIA-25NOV0{i + 1}-T95",
            strategy="tails",
            layer="routing",
            name="tails_not_invoked_blacklisted",
            reason="tails_not_invoked_blacklisted",
        )
        for i in range(3)
    ]

    report = score_run([], [], [], failures)

    mia = report.cities[0]
    assert mia.skips["tails_blacklisted"] == 3
    assert mia.skips["tails_blacklisted"] > 0


def test_tails_strategy_layer_blacklist_reason_not_rendered_as_tails_blacklisted() -> None:
    failures = [
        make_failure(
            "KXHIGHLAX-25NOV04-T80",
            strategy="tails",
            layer="strategy",
            name="blacklisted",
            reason="blacklisted",
        )
    ]

    report = score_run([], [], [], failures)

    lax = report.cities[0]
    assert "tails_blacklisted" not in lax.skips
    assert "edge_blacklisted" not in lax.skips
    assert lax.skips["strategy:blacklisted"] == 1


def test_groups_split_by_city_and_month() -> None:
    orders = [
        make_order("KXHIGHDEN-25NOV04-B61.5", fair_yes=Decimal("0.9")),
        make_order("KXHIGHDEN-25DEC02-B50.5", fair_yes=Decimal("0.8")),
        make_order("KXHIGHAUS-25NOV04-B80.5", fair_yes=Decimal("0.7"), lead_bucket="8-24h"),
    ]
    fills = [make_fill(Decimal("0")) for _ in orders]
    settlements = [OrderSettlement(result="yes", market_mid=Decimal("0.5")) for _ in orders]

    report = score_run(orders, fills, settlements, [])

    keys = [(g.city, g.lead_bucket, g.month) for g in report.groups]
    assert keys == [
        ("KXHIGHAUS", "8-24h", "2025-11"),
        ("KXHIGHDEN", "24-72h", "2025-11"),
        ("KXHIGHDEN", "24-72h", "2025-12"),
    ]
    assert all(g.n_orders == 1 for g in report.groups)
    assert [c.city for c in report.cities] == ["KXHIGHAUS", "KXHIGHDEN"]
    assert report.cities[1].n_orders == 2


def test_hit_rate_and_net_pnl() -> None:
    orders = [
        make_order("KXHIGHDEN-25NOV04-B61.5", fair_yes=Decimal("0.9"), action=TradeSide.BUY_YES),
        make_order("KXHIGHDEN-25NOV05-T70", fair_yes=Decimal("0.1"), action=TradeSide.SELL_YES),
        make_order("KXHIGHDEN-25NOV06-T70", fair_yes=Decimal("0.1"), action=TradeSide.SELL_YES),
    ]
    fills = [
        make_fill(Decimal("0.10")),
        make_fill(Decimal("-0.30")),
        make_fill(Decimal("0.05")),
    ]
    settlements = [
        OrderSettlement(result="yes", market_mid=Decimal("0.85")),
        OrderSettlement(result="yes", market_mid=Decimal("0.15")),
        OrderSettlement(result="no", market_mid=Decimal("0.15")),
    ]

    report = score_run(orders, fills, settlements, [])

    city = report.cities[0]
    assert city.metrics is not None
    assert city.metrics.hit_rate == Decimal("0.666667")
    assert city.metrics.net_pnl == Decimal("-0.15")
