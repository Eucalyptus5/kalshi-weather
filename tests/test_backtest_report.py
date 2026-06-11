import csv
import re
import subprocess
import sys
import types
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bot.backtest.cli import PASSING_VERDICTS
from bot.backtest.engine import BacktestOrder
from bot.backtest.pnl import BacktestFill
from bot.backtest.report import (
    FidelityInputs,
    FidelitySnapshot,
    ScoredRun,
    SignedFillCount,
    evaluate_runs,
    write_report,
)
from bot.backtest.scoring import CityScore, MetricRow, ScoreReport
from bot.execution.paper import TradeSide

REPO = Path(__file__).resolve().parents[1]
_AS_OF = datetime(2025, 11, 4, 17, 0, tzinfo=timezone.utc)


def make_order(ticker: str) -> BacktestOrder:
    return BacktestOrder(
        market_ticker=ticker,
        as_of=_AS_OF,
        strategy="edge",
        action=TradeSide.BUY_YES,
        contracts=2,
        fair_yes=Decimal("0.62"),
        price_per_contract=Decimal("0.50"),
        order_dollars=Decimal("1.00"),
        depth_at_price=300,
        depth_source="snapshot",
        lead_bucket="24-72h",
    )


def make_fill(net_pnl: str) -> BacktestFill:
    return BacktestFill(
        gross_pnl=Decimal(net_pnl), fee_dollars=Decimal("0"), net_pnl=Decimal(net_pnl)
    )


def metric(net_pnl: str) -> MetricRow:
    return MetricRow(
        brier=Decimal("0.10"),
        log_loss=Decimal("0.30"),
        hit_rate=Decimal("0.50"),
        net_pnl=Decimal(net_pnl),
        baseline_brier=Decimal("0.20"),
        beats_baseline=True,
    )


def city(
    name: str,
    net_pnl: str | None,
    *,
    friction: int = 0,
    skips: dict[str, int] | None = None,
) -> CityScore:
    merged = dict(skips or {})
    if friction:
        merged["gate:edge_after_friction"] = friction
    return CityScore(
        city=name,
        n_orders=0 if net_pnl is None else 1,
        skips=merged,
        metrics=None if net_pnl is None else metric(net_pnl),
    )


def run(
    strategy: str,
    window: str,
    bankroll: str,
    cities: list[CityScore],
    *,
    depth: str = "1.0",
    sigma: str = "1.0",
    gross_before: str | None = None,
) -> ScoredRun:
    return ScoredRun(
        strategy=strategy,
        window=window,
        bankroll=Decimal(bankroll),
        depth_multiplier=Decimal(depth),
        sigma_multiplier=Decimal(sigma),
        report=ScoreReport(groups=[], cities=cities),
        gross_pnl_before_friction_skips=None if gross_before is None else Decimal(gross_before),
    )


def base_runs() -> list[ScoredRun]:
    return [
        run(
            "edge",
            "in_sample",
            "20000",
            [
                city("KXHIGHDEN", "-3.10", friction=2),
                city("KXHIGHAUS", "1.40", friction=1),
                city("KXHIGHLAX", None, skips={"edge_blacklisted": 14}),
            ],
        ),
        run(
            "tails",
            "in_sample",
            "20000",
            [
                city("KXHIGHDEN", "0.80", friction=1),
                city("KXHIGHLAX", None, skips={"tails_blacklisted": 9}),
            ],
        ),
        run(
            "edge",
            "out_of_sample",
            "20000",
            [city("KXHIGHDEN", "2.20", friction=1), city("KXHIGHAUS", "0.30")],
        ),
        run("tails", "out_of_sample", "20000", [city("KXHIGHDEN", "0.10")]),
        run(
            "edge",
            "in_sample",
            "500",
            [city("KXHIGHDEN", "-0.40", friction=5), city("KXHIGHAUS", "0.20", friction=3)],
        ),
        run("tails", "in_sample", "500", [city("KXHIGHDEN", "0.05", friction=1)]),
        run("edge", "out_of_sample", "500", [city("KXHIGHDEN", "0.60", friction=2)]),
        run("tails", "out_of_sample", "500", [city("KXHIGHDEN", "0.02")]),
    ]


def test_returns_summary_path_and_writes_artifacts(tmp_path: Path) -> None:
    orders = [make_order("KXHIGHDEN-25NOV04-B61.5")]
    fills = [make_fill("0.10")]

    path = write_report(base_runs(), orders, fills, tmp_path)

    assert path == tmp_path / "summary.md"
    assert path.exists()
    assert (tmp_path / "orders.csv").exists()
    assert (tmp_path / "orders.parquet").exists()


def test_csv_and_parquet_row_counts_match_orders(tmp_path: Path) -> None:
    orders = [
        make_order("KXHIGHDEN-25NOV04-B61.5"),
        make_order("KXHIGHDEN-25NOV05-B59.5"),
        make_order("KXHIGHAUS-25NOV04-B80.5"),
    ]
    fills = [make_fill("0.10"), make_fill("-0.30"), make_fill("0.05")]

    write_report(base_runs(), orders, fills, tmp_path)

    with (tmp_path / "orders.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(orders)
    assert rows[0]["market_ticker"] == "KXHIGHDEN-25NOV04-B61.5"
    assert rows[1]["net_pnl"] == "-0.30"
    table = pq.read_table(tmp_path / "orders.parquet")
    assert table.num_rows == len(orders)
    assert table.column("fair_yes").to_pylist() == ["0.62", "0.62", "0.62"]


def test_mismatched_fills_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_report(base_runs(), [make_order("KXHIGHDEN-25NOV04-B61.5")], [], tmp_path)


def test_commits_block_lists_head_and_strategy_file_shas(tmp_path: Path) -> None:
    path = write_report(base_runs(), [], [], tmp_path)
    md = path.read_text()

    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert "commits:" in md
    assert f"head: {head}" in md
    for rel in (
        "bot/strategy/__init__.py",
        "bot/strategy/edge.py",
        "bot/strategy/sizing.py",
        "bot/strategy/tails.py",
        "bot/risk/gates.py",
    ):
        assert re.search(rf"{re.escape(rel)}: [0-9a-f]{{40}}", md)


def test_headline_surfaces_counterfactual_flag_inline(tmp_path: Path) -> None:
    md = write_report(base_runs(), [], [], tmp_path).read_text()

    assert "headline $20000 in_sample: net_pnl=-0.90 counterfactual=true" in md
    assert "headline $20000 out_of_sample: net_pnl=2.60 counterfactual=false" in md
    assert "headline $500 in_sample: net_pnl=-0.15 counterfactual=true" in md
    assert "| $20000 | in_sample | counterfactual=true | -0.90 | 4 | no |" in md
    assert "| $500 | in_sample | counterfactual=true | -0.15 | 9 | no |" in md
    assert "| $500 | out_of_sample | counterfactual=false | 0.62 | 2 | no |" in md


def test_per_city_rows_render_per_strategy_bankroll_and_window(tmp_path: Path) -> None:
    md = write_report(base_runs(), [], [], tmp_path).read_text()

    assert "friction_skips" in md
    assert (
        "| KXHIGHDEN | edge | $20000 | in_sample | counterfactual=true | -3.10 | 2 | 0 | 0 | 0 |"
        in md
    )
    assert (
        "| KXHIGHDEN | edge | $20000 | out_of_sample | counterfactual=false | 2.20 | 1 | 0 | 0 | 0 |"
        in md
    )
    assert (
        "| KXHIGHDEN | edge | $500 | in_sample | counterfactual=true | -0.40 | 5 | 0 | 0 | 0 |"
        in md
    )
    assert (
        "| KXHIGHDEN | tails | $20000 | in_sample | counterfactual=true | 0.80 | 1 | 0 | 0 | 0 |"
        in md
    )
    assert (
        "| KXHIGHDEN | tails | $500 | in_sample | counterfactual=true | 0.05 | 1 | 0 | 0 | 0 |"
        in md
    )
    assert (
        "| KXHIGHAUS | edge | $20000 | in_sample | counterfactual=true | 1.40 | 1 | 0 | 0 | 0 |"
        in md
    )


def test_structurally_empty_city_flagged_for_both_strategies(tmp_path: Path) -> None:
    md = write_report(base_runs(), [], [], tmp_path).read_text()

    assert (
        "| KXHIGHLAX | edge | $20000 | in_sample | counterfactual=true | structurally_empty | 0 | 14 | 0 | 0 |"
        in md
    )
    assert (
        "| KXHIGHLAX | tails | $20000 | in_sample | counterfactual=true | structurally_empty | 0 | 0 | 9 | 0 |"
        in md
    )


def test_plots_skipped_note_without_matplotlib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)

    path = write_report(base_runs(), [], [], tmp_path)

    assert "plots skipped (matplotlib not installed)" in path.read_text()


def test_plot_written_with_fake_matplotlib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Fig:
        def savefig(self, path: Path) -> None:
            Path(path).write_bytes(b"png")

    class Ax:
        def bar(self, x: list[str], y: list[float]) -> None:
            pass

        def set_ylabel(self, label: str) -> None:
            pass

    pyplot = types.SimpleNamespace(subplots=lambda: (Fig(), Ax()), close=lambda fig: None)
    fake = types.SimpleNamespace(use=lambda backend: None, pyplot=pyplot)
    monkeypatch.setitem(sys.modules, "matplotlib", fake)

    md = write_report(base_runs(), [], [], tmp_path).read_text()

    assert "plot: pnl_by_city.png" in md
    assert (tmp_path / "pnl_by_city.png").exists()


def depth_runs(*, rung_friction: int, rung_gross_before: str | None) -> list[ScoredRun]:
    return [
        run("edge", "in_sample", "20000", [city("KXHIGHDEN", "2.00", friction=4)]),
        run(
            "edge",
            "in_sample",
            "20000",
            [city("KXHIGHDEN", "-6.00", friction=rung_friction)],
            depth="0.5",
            gross_before=rung_gross_before,
        ),
        run(
            "edge",
            "in_sample",
            "20000",
            [city("KXHIGHDEN", "1.00", friction=6)],
            depth="2.0",
        ),
    ]


def test_depth_rung_friction_bound_true_excluded_from_stop(tmp_path: Path) -> None:
    runs = depth_runs(rung_friction=12, rung_gross_before="1.00")

    verdict = evaluate_runs(runs)
    md = write_report(runs, [], [], tmp_path).read_text()

    rung = next(d for d in verdict.depth if d.multiplier == Decimal("0.5"))
    assert rung.friction_bound is True
    assert rung.counted_in_stop is False
    assert verdict.final == "survive"
    assert "| in_sample | 0.5x | -6.00 | 12 | yes | friction_bound=true | excluded |" in md
    assert "| in_sample | 2.0x | 1.00 | 6 | no | - | - |" in md
    assert "fidelity flag, not a survival stop" in md
    assert "verdict: survive" in md


def test_depth_rung_friction_bound_false_counts_as_stop(tmp_path: Path) -> None:
    runs = depth_runs(rung_friction=5, rung_gross_before="-2.00")

    verdict = evaluate_runs(runs)
    md = write_report(runs, [], [], tmp_path).read_text()

    rung = next(d for d in verdict.depth if d.multiplier == Decimal("0.5"))
    assert rung.friction_bound is False
    assert rung.counted_in_stop is True
    assert verdict.final == "stop"
    assert "| in_sample | 0.5x | -6.00 | 5 | yes | friction_bound=false | counted |" in md
    assert "verdict: stop" in md


def test_sigma_rung_negative_across_all_cities_fails_money_stop(tmp_path: Path) -> None:
    cities = [
        "KXHIGHAUS",
        "KXHIGHCHI",
        "KXHIGHDEN",
        "KXHIGHMIN",
        "KXHIGHNY",
        "KXHIGHPHIL",
        "KXHIGHSEA",
    ]
    runs = [
        run("edge", "in_sample", "20000", [city(c, "1.00") for c in cities]),
        run("edge", "in_sample", "20000", [city(c, "-1.00") for c in cities], sigma="0.75"),
        run("edge", "in_sample", "20000", [city(c, "0.50") for c in cities], sigma="1.25"),
    ]

    verdict = evaluate_runs(runs)
    md = write_report(runs, [], [], tmp_path).read_text()

    assert verdict.final == "stop"
    assert "| in_sample | 0.75 | -7.00 | yes |" in md
    assert "| in_sample | 1.25 | 3.50 | no |" in md
    assert "verdict: stop" in md


def test_friction_floor_flip_labeled_not_edge_failure(tmp_path: Path) -> None:
    runs = [
        run(
            "edge",
            "in_sample",
            "20000",
            [city("KXHIGHDEN", "-3.00", friction=10), city("KXHIGHAUS", "-1.00")],
        ),
        run(
            "edge",
            "in_sample",
            "500",
            [city("KXHIGHDEN", "0.50", friction=1), city("KXHIGHAUS", "-0.20", friction=1)],
        ),
    ]

    verdict = evaluate_runs(runs)
    md = write_report(runs, [], [], tmp_path).read_text()

    assert verdict.window_verdicts["in_sample"] == "friction_floor_driven"
    assert verdict.final == "friction_floor_driven"
    assert "friction-floor-driven" in md
    assert "verdict: friction_floor_driven" in md
    assert "verdict: stop" not in md


def test_windows_disagree_is_strategy_era_flip(tmp_path: Path) -> None:
    runs = [
        run(
            "edge",
            "in_sample",
            "20000",
            [city("KXHIGHDEN", "-3.00", friction=2), city("KXHIGHAUS", "-1.00")],
        ),
        run(
            "edge",
            "out_of_sample",
            "20000",
            [city("KXHIGHDEN", "2.00"), city("KXHIGHAUS", "1.00")],
        ),
    ]

    verdict = evaluate_runs(runs)
    md = write_report(runs, [], [], tmp_path).read_text()

    assert verdict.era_flip is True
    assert verdict.final == "strategy_era_flip"
    assert "strategy-era flip" in md
    assert "verdict: strategy_era_flip" in md
    assert "verdict: stop" not in md


def test_signed_fill_bound_trip_invalidates_verdict(tmp_path: Path) -> None:
    fidelity = FidelityInputs(
        snapshots=[],
        signed_fills=[
            SignedFillCount(city="KXHIGHDEN", strategy="tails", primary=100, crosscheck=140),
            SignedFillCount(city="KXHIGHAUS", strategy="edge", primary=40, crosscheck=45),
            SignedFillCount(city="KXHIGHCHI", strategy="tails", primary=10, crosscheck=-10),
        ],
    )

    verdict = evaluate_runs(base_runs(), fidelity)
    md = write_report(base_runs(), [], [], tmp_path, fidelity=fidelity).read_text()

    assert verdict.invalidated is True
    assert verdict.final == "invalidated"
    assert "| KXHIGHDEN | tails | 100 | 140 | fail |" in md
    assert "| KXHIGHAUS | edge | 40 | 45 | pass |" in md
    assert "| KXHIGHCHI | tails | 10 | -10 | fail |" in md
    assert "survival verdict invalidated" in md
    assert "verdict: invalidated" in md


def test_prob_bound_regime_split(tmp_path: Path) -> None:
    deep = [
        FidelitySnapshot(
            city="KXHIGHDEN",
            lead_bucket="24-72h",
            primary_prob=Decimal("0.01"),
            crosscheck_prob=Decimal("0.02"),
        )
        for _ in range(3)
    ]
    mid = [
        FidelitySnapshot(
            city="KXHIGHDEN",
            lead_bucket="24-72h",
            primary_prob=Decimal("0.05"),
            crosscheck_prob=Decimal(p),
        )
        for p in ("0.054", "0.046", "0.05")
    ]
    outside = [
        FidelitySnapshot(
            city="KXHIGHDEN",
            lead_bucket="24-72h",
            primary_prob=Decimal("0.30"),
            crosscheck_prob=Decimal("0.40"),
        )
    ]
    fidelity = FidelityInputs(snapshots=deep + mid + outside, signed_fills=[])

    verdict = evaluate_runs(base_runs(), fidelity)
    md = write_report(base_runs(), [], [], tmp_path, fidelity=fidelity).read_text()

    assert [(r.regime, r.passed) for r in verdict.prob_rows] == [
        ("deep_tail", False),
        ("mid_tail", True),
    ]
    assert verdict.final == "invalidated"
    assert "| KXHIGHDEN | 24-72h | deep_tail | 3 | 0.01 | 0.005 | fail |" in md
    assert "| KXHIGHDEN | 24-72h | mid_tail | 3 | 0.004 | 0.005 | pass |" in md


def test_fidelity_not_run_is_surfaced(tmp_path: Path) -> None:
    verdict = evaluate_runs(base_runs())
    md = write_report(base_runs(), [], [], tmp_path).read_text()

    assert verdict.fidelity_ran is False
    assert verdict.invalidated is False
    assert "fidelity cross-check not run" in md


def test_zero_order_window_is_insufficient_data_not_survive() -> None:
    runs = [
        run(
            "tails",
            "out_of_sample",
            "20000",
            [city("KXHIGHDEN", None, skips={"strategy:depth_zero_clamp": 6})],
        )
    ]

    verdict = evaluate_runs(runs)

    assert verdict.window_verdicts["out_of_sample"] == "insufficient_data"
    assert verdict.friction_floor["out_of_sample"] is False
    assert verdict.final == "insufficient_data"


def test_zero_order_window_overrides_surviving_window() -> None:
    runs = [
        run(
            "edge",
            "in_sample",
            "20000",
            [city("KXHIGHDEN", "1.50"), city("KXHIGHAUS", "0.40")],
        ),
        run(
            "tails",
            "out_of_sample",
            "20000",
            [city("KXHIGHDEN", None, skips={"strategy:depth_zero_clamp": 6})],
        ),
    ]

    verdict = evaluate_runs(runs)

    assert verdict.window_verdicts["in_sample"] == "survive"
    assert verdict.window_verdicts["out_of_sample"] == "insufficient_data"
    assert verdict.final == "insufficient_data"
    assert verdict.era_flip is False


def test_stop_window_beats_zero_order_window() -> None:
    runs = [
        run(
            "edge",
            "in_sample",
            "20000",
            [city("KXHIGHDEN", "-2.00"), city("KXHIGHAUS", "-1.00")],
        ),
        run(
            "tails",
            "out_of_sample",
            "20000",
            [city("KXHIGHDEN", None, skips={"strategy:depth_zero_clamp": 6})],
        ),
    ]

    verdict = evaluate_runs(runs)

    assert verdict.window_verdicts["in_sample"] == "stop"
    assert verdict.window_verdicts["out_of_sample"] == "insufficient_data"
    assert verdict.final == "stop"
    assert verdict.era_flip is False


def test_insufficient_data_fails_cli_strict_path() -> None:
    assert "insufficient_data" not in PASSING_VERDICTS
