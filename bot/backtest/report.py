from __future__ import annotations

import csv
import statistics
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bot.backtest.engine import BacktestOrder
from bot.backtest.pnl import BacktestFill
from bot.backtest.scoring import CityScore, ScoreReport

REFERENCE_BANKROLL = Decimal("20000")
SENSITIVITY_BANKROLL = Decimal("500")
BASE_MULTIPLIER = Decimal("1.0")
FRICTION_SKIP_KEY = "gate:edge_after_friction"
WINDOWS: tuple[str, str] = ("in_sample", "out_of_sample")
TAILS_REGIME_MAX = Decimal("0.10")
DEEP_TAIL_SPLIT = Decimal("0.02")
DEEP_TAIL_REL = Decimal("0.5")
DEEP_TAIL_FLOOR = Decimal("0.0005")
MID_TAIL_ABS = Decimal("0.005")
MID_TAIL_REL = Decimal("0.25")
SIGNED_FILL_TOLERANCE = Decimal("0.25")

_REPO_ROOT = Path(__file__).resolve().parents[2]

_ORDER_COLUMNS = (
    "market_ticker",
    "as_of",
    "strategy",
    "action",
    "contracts",
    "fair_yes",
    "price_per_contract",
    "order_dollars",
    "depth_at_price",
    "depth_source",
    "lead_bucket",
    "gross_pnl",
    "fee_dollars",
    "net_pnl",
)

_ORDERS_SCHEMA = pa.schema(
    [
        ("market_ticker", pa.string()),
        ("as_of", pa.string()),
        ("strategy", pa.string()),
        ("action", pa.string()),
        ("contracts", pa.int64()),
        ("fair_yes", pa.string()),
        ("price_per_contract", pa.string()),
        ("order_dollars", pa.string()),
        ("depth_at_price", pa.int64()),
        ("depth_source", pa.string()),
        ("lead_bucket", pa.string()),
        ("gross_pnl", pa.string()),
        ("fee_dollars", pa.string()),
        ("net_pnl", pa.string()),
    ]
)


@dataclass(frozen=True, slots=True)
class ScoredRun:
    strategy: str
    window: str
    bankroll: Decimal
    depth_multiplier: Decimal
    sigma_multiplier: Decimal
    report: ScoreReport
    gross_pnl_before_friction_skips: Decimal | None = None


@dataclass(frozen=True, slots=True)
class FidelitySnapshot:
    city: str
    lead_bucket: str
    primary_prob: Decimal
    crosscheck_prob: Decimal


@dataclass(frozen=True, slots=True)
class SignedFillCount:
    city: str
    strategy: str
    primary: int
    crosscheck: int


@dataclass(frozen=True, slots=True)
class FidelityInputs:
    snapshots: list[FidelitySnapshot] = field(default_factory=list)
    signed_fills: list[SignedFillCount] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class HeadlineRow:
    bankroll: Decimal
    window: str
    counterfactual: bool
    net_pnl: Decimal
    friction_skips: int
    all_cities_negative: bool


@dataclass(frozen=True, slots=True)
class DepthRow:
    window: str
    multiplier: Decimal
    net_pnl: Decimal
    friction_skips: int
    all_cities_negative: bool
    friction_bound: bool | None
    counted_in_stop: bool


@dataclass(frozen=True, slots=True)
class SigmaRow:
    window: str
    multiplier: Decimal
    net_pnl: Decimal
    all_cities_negative: bool


@dataclass(frozen=True, slots=True)
class ProbBoundRow:
    city: str
    lead_bucket: str
    regime: str
    n_snapshots: int
    median_delta: Decimal
    threshold: Decimal
    passed: bool


@dataclass(frozen=True, slots=True)
class SignedFillRow:
    city: str
    strategy: str
    primary: int
    crosscheck: int
    passed: bool


@dataclass(frozen=True, slots=True)
class ReportVerdict:
    headline: list[HeadlineRow]
    depth: list[DepthRow]
    sigma: list[SigmaRow]
    window_verdicts: dict[str, str]
    friction_floor: dict[str, bool]
    era_flip: bool
    fidelity_ran: bool
    prob_rows: list[ProbBoundRow]
    fill_rows: list[SignedFillRow]
    invalidated: bool
    final: str


def _windows(runs: list[ScoredRun]) -> list[str]:
    return [w for w in WINDOWS if any(r.window == w for r in runs)]


def _select(
    runs: list[ScoredRun],
    window: str,
    bankroll: Decimal,
    depth: Decimal,
    sigma: Decimal,
) -> list[ScoredRun]:
    return [
        r
        for r in runs
        if r.window == window
        and r.bankroll == bankroll
        and r.depth_multiplier == depth
        and r.sigma_multiplier == sigma
    ]


def _city_pnls(group: list[ScoredRun]) -> dict[str, Decimal]:
    out: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    for run in group:
        for score in run.report.cities:
            if score.metrics is not None:
                out[score.city] += score.metrics.net_pnl
    return dict(out)


def _all_negative(city_pnls: dict[str, Decimal]) -> bool:
    return bool(city_pnls) and all(pnl < 0 for pnl in city_pnls.values())


def _friction_skips(group: list[ScoredRun]) -> int:
    return sum(
        score.skips.get(FRICTION_SKIP_KEY, 0) for run in group for score in run.report.cities
    )


def _gross_before(group: list[ScoredRun]) -> Decimal | None:
    values = [r.gross_pnl_before_friction_skips for r in group]
    if not values or any(v is None for v in values):
        return None
    return sum(values, Decimal("0"))


def _prob_bound_rows(snapshots: list[FidelitySnapshot]) -> list[ProbBoundRow]:
    buckets: dict[tuple[str, str, str], list[FidelitySnapshot]] = defaultdict(list)
    for snap in snapshots:
        if snap.primary_prob > TAILS_REGIME_MAX:
            continue
        regime = "deep_tail" if snap.primary_prob <= DEEP_TAIL_SPLIT else "mid_tail"
        buckets[(snap.city, snap.lead_bucket, regime)].append(snap)

    rows: list[ProbBoundRow] = []
    for (city, lead_bucket, regime), snaps in sorted(buckets.items()):
        median_delta = statistics.median(abs(s.primary_prob - s.crosscheck_prob) for s in snaps)
        median_primary = statistics.median(s.primary_prob for s in snaps)
        if regime == "deep_tail":
            threshold = max(DEEP_TAIL_REL * median_primary, DEEP_TAIL_FLOOR)
        else:
            threshold = min(MID_TAIL_ABS, MID_TAIL_REL * median_primary)
        rows.append(
            ProbBoundRow(
                city=city,
                lead_bucket=lead_bucket,
                regime=regime,
                n_snapshots=len(snaps),
                median_delta=median_delta,
                threshold=threshold,
                passed=median_delta <= threshold,
            )
        )
    return rows


def _signed_fill_rows(counts: list[SignedFillCount]) -> list[SignedFillRow]:
    rows: list[SignedFillRow] = []
    for count in counts:
        tolerance = SIGNED_FILL_TOLERANCE * abs(count.primary)
        rows.append(
            SignedFillRow(
                city=count.city,
                strategy=count.strategy,
                primary=count.primary,
                crosscheck=count.crosscheck,
                passed=abs(count.crosscheck - count.primary) <= tolerance,
            )
        )
    return rows


def evaluate_runs(runs: list[ScoredRun], fidelity: FidelityInputs | None = None) -> ReportVerdict:
    windows = _windows(runs)

    headline: list[HeadlineRow] = []
    for bankroll in (REFERENCE_BANKROLL, SENSITIVITY_BANKROLL):
        for window in windows:
            group = _select(runs, window, bankroll, BASE_MULTIPLIER, BASE_MULTIPLIER)
            if not group:
                continue
            pnls = _city_pnls(group)
            headline.append(
                HeadlineRow(
                    bankroll=bankroll,
                    window=window,
                    counterfactual=window == "in_sample",
                    net_pnl=sum(pnls.values(), Decimal("0")),
                    friction_skips=_friction_skips(group),
                    all_cities_negative=_all_negative(pnls),
                )
            )

    depth_rows: list[DepthRow] = []
    for window in windows:
        base_group = _select(runs, window, REFERENCE_BANKROLL, BASE_MULTIPLIER, BASE_MULTIPLIER)
        base_skips = _friction_skips(base_group) if base_group else None
        multipliers = sorted(
            {
                r.depth_multiplier
                for r in runs
                if r.window == window
                and r.bankroll == REFERENCE_BANKROLL
                and r.sigma_multiplier == BASE_MULTIPLIER
            }
        )
        for multiplier in multipliers:
            group = _select(runs, window, REFERENCE_BANKROLL, multiplier, BASE_MULTIPLIER)
            pnls = _city_pnls(group)
            negative = _all_negative(pnls)
            skips = _friction_skips(group)
            gross_before = _gross_before(group)
            friction_bound: bool | None = None
            if negative:
                friction_bound = (
                    multiplier != BASE_MULTIPLIER
                    and base_skips is not None
                    and skips > 2 * base_skips
                    and gross_before is not None
                    and gross_before >= 0
                )
            depth_rows.append(
                DepthRow(
                    window=window,
                    multiplier=multiplier,
                    net_pnl=sum(pnls.values(), Decimal("0")),
                    friction_skips=skips,
                    all_cities_negative=negative,
                    friction_bound=friction_bound,
                    counted_in_stop=bool(negative and not friction_bound),
                )
            )

    sigma_rows: list[SigmaRow] = []
    for window in windows:
        multipliers = sorted(
            {
                r.sigma_multiplier
                for r in runs
                if r.window == window
                and r.bankroll == REFERENCE_BANKROLL
                and r.depth_multiplier == BASE_MULTIPLIER
            }
        )
        for multiplier in multipliers:
            group = _select(runs, window, REFERENCE_BANKROLL, BASE_MULTIPLIER, multiplier)
            pnls = _city_pnls(group)
            sigma_rows.append(
                SigmaRow(
                    window=window,
                    multiplier=multiplier,
                    net_pnl=sum(pnls.values(), Decimal("0")),
                    all_cities_negative=_all_negative(pnls),
                )
            )

    window_orders = {
        window: sum(score.n_orders for r in runs if r.window == window for score in r.report.cities)
        for window in windows
    }

    by_key = {(h.bankroll, h.window): h for h in headline}
    window_verdicts: dict[str, str] = {}
    friction_floor: dict[str, bool] = {}
    for window in windows:
        if window_orders[window] == 0:
            window_verdicts[window] = "insufficient_data"
            friction_floor[window] = False
            continue
        base = by_key.get((REFERENCE_BANKROLL, window))
        if base is None:
            continue
        sensitivity = by_key.get((SENSITIVITY_BANKROLL, window))
        floor_driven = (
            base.all_cities_negative
            and sensitivity is not None
            and not sensitivity.all_cities_negative
            and base.friction_skips > 2 * sensitivity.friction_skips
        )
        friction_floor[window] = floor_driven
        depth_stop = any(
            d.counted_in_stop
            for d in depth_rows
            if d.window == window and d.multiplier != BASE_MULTIPLIER
        )
        sigma_stop = any(
            s.all_cities_negative
            for s in sigma_rows
            if s.window == window and s.multiplier != BASE_MULTIPLIER
        )
        if not base.all_cities_negative and not depth_stop and not sigma_stop:
            window_verdicts[window] = "survive"
        elif depth_stop or sigma_stop or not floor_driven:
            window_verdicts[window] = "stop"
        else:
            window_verdicts[window] = "friction_floor_driven"

    reference = [by_key.get((REFERENCE_BANKROLL, w)) for w in WINDOWS]
    era_flip = (
        reference[0] is not None
        and reference[1] is not None
        and all(window_orders[w] > 0 for w in WINDOWS)
        and reference[0].all_cities_negative != reference[1].all_cities_negative
    )

    prob_rows = _prob_bound_rows(fidelity.snapshots) if fidelity is not None else []
    fill_rows = _signed_fill_rows(fidelity.signed_fills) if fidelity is not None else []
    invalidated = any(not r.passed for r in prob_rows) or any(not r.passed for r in fill_rows)

    if invalidated:
        final = "invalidated"
    elif era_flip:
        final = "strategy_era_flip"
    elif "stop" in window_verdicts.values():
        final = "stop"
    elif "insufficient_data" in window_verdicts.values():
        final = "insufficient_data"
    elif "friction_floor_driven" in window_verdicts.values():
        final = "friction_floor_driven"
    else:
        final = "survive"

    return ReportVerdict(
        headline=headline,
        depth=depth_rows,
        sigma=sigma_rows,
        window_verdicts=window_verdicts,
        friction_floor=friction_floor,
        era_flip=era_flip,
        fidelity_ran=fidelity is not None,
        prob_rows=prob_rows,
        fill_rows=fill_rows,
        invalidated=invalidated,
        final=final,
    )


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(_REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _commit_lines() -> list[str]:
    lines = ["commits:", f"  head: {_git('rev-parse', 'HEAD')}"]
    paths = sorted((_REPO_ROOT / "bot" / "strategy").glob("*.py")) + [
        _REPO_ROOT / "bot" / "risk" / "gates.py"
    ]
    for path in paths:
        rel = path.relative_to(_REPO_ROOT).as_posix()
        lines.append(f"  {rel}: {_git('log', '-n', '1', '--format=%H', '--', rel)}")
    return lines


def _plot_note(city_pnls: dict[str, Decimal], out_dir: Path) -> str:
    try:
        import matplotlib
    except ImportError:
        return "plots skipped (matplotlib not installed)"
    if not city_pnls:
        return "plots skipped (no scored cities)"
    matplotlib.use("Agg")
    from matplotlib import pyplot

    fig, ax = pyplot.subplots()
    cities = sorted(city_pnls)
    ax.bar(cities, [float(city_pnls[c]) for c in cities])
    ax.set_ylabel("net_pnl")
    fig.savefig(out_dir / "pnl_by_city.png")
    pyplot.close(fig)
    return "plot: pnl_by_city.png"


def _flag(window: str) -> str:
    return "counterfactual=true" if window == "in_sample" else "counterfactual=false"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _order_record(order: BacktestOrder, fill: BacktestFill) -> dict[str, str | int]:
    return {
        "market_ticker": order.market_ticker,
        "as_of": order.as_of.isoformat(),
        "strategy": order.strategy,
        "action": order.action.value,
        "contracts": order.contracts,
        "fair_yes": str(order.fair_yes),
        "price_per_contract": str(order.price_per_contract),
        "order_dollars": str(order.order_dollars),
        "depth_at_price": order.depth_at_price,
        "depth_source": order.depth_source,
        "lead_bucket": order.lead_bucket,
        "gross_pnl": str(fill.gross_pnl),
        "fee_dollars": str(fill.fee_dollars),
        "net_pnl": str(fill.net_pnl),
    }


def _write_orders(records: list[dict[str, str | int]], out_dir: Path) -> None:
    with (out_dir / "orders.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_ORDER_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    pq.write_table(pa.Table.from_pylist(records, schema=_ORDERS_SCHEMA), out_dir / "orders.parquet")


def _render_per_city(runs: list[ScoredRun], windows: list[str], lines: list[str]) -> None:
    base = [
        r
        for r in runs
        if r.depth_multiplier == BASE_MULTIPLIER and r.sigma_multiplier == BASE_MULTIPLIER
    ]
    pairs = sorted({(score.city, r.strategy) for r in base for score in r.report.cities})
    bankrolls = [
        b for b in (REFERENCE_BANKROLL, SENSITIVITY_BANKROLL) if any(r.bankroll == b for r in base)
    ]
    scores: dict[tuple[str, str, Decimal], dict[str, CityScore]] = {
        (r.strategy, r.window, r.bankroll): {score.city: score for score in r.report.cities}
        for r in base
    }

    lines.append("## per-city pnl by strategy")
    lines.append("")
    lines.append(
        "structurally empty cells mean the city produced no scored orders for that strategy"
        " (blacklist skip counts are in their own columns), not a zero pnl result."
    )
    lines.append("")
    header = (
        "| city | strategy | bankroll | window | counterfactual | net_pnl | friction_skips |"
        " edge_blacklisted | tails_blacklisted | other_skips |"
    )
    lines.append(header)
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for city_name, strategy in pairs:
        for bankroll in bankrolls:
            for window in windows:
                score = scores.get((strategy, window, bankroll), {}).get(city_name)
                if score is None:
                    cells = ["-", "-", "-", "-", "-"]
                else:
                    other = sum(
                        v
                        for k, v in score.skips.items()
                        if k not in (FRICTION_SKIP_KEY, "edge_blacklisted", "tails_blacklisted")
                    )
                    pnl = (
                        "structurally_empty"
                        if score.metrics is None
                        else str(score.metrics.net_pnl)
                    )
                    cells = [
                        pnl,
                        str(score.skips.get(FRICTION_SKIP_KEY, 0)),
                        str(score.skips.get("edge_blacklisted", 0)),
                        str(score.skips.get("tails_blacklisted", 0)),
                        str(other),
                    ]
                lines.append(
                    f"| {city_name} | {strategy} | ${bankroll} | {window} | {_flag(window)} | "
                    + " | ".join(cells)
                    + " |"
                )
    lines.append("")


def _render_markdown(runs: list[ScoredRun], verdict: ReportVerdict, plot_note: str) -> str:
    windows = _windows(runs)
    by_key = {(h.bankroll, h.window): h for h in verdict.headline}

    lines: list[str] = ["# backtest report", ""]
    lines.extend(_commit_lines())
    lines.append("")
    for row in verdict.headline:
        lines.append(
            f"headline ${row.bankroll} {row.window}: net_pnl={row.net_pnl} {_flag(row.window)}"
        )
    lines.append("")
    lines.append(
        "in_sample rows are counterfactual=true: that window predates the strategy commits"
        " listed above, so the caps and friction gate are an overlay applied to pre-era"
        " prices, not a live-strategy backtest."
    )
    lines.append("")
    lines.append(plot_note)
    lines.append("")

    lines.append("## headline")
    lines.append("")
    lines.append(
        "| bankroll | window | counterfactual | net_pnl | friction_skips | all_cities_negative |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in verdict.headline:
        lines.append(
            f"| ${row.bankroll} | {row.window} | {_flag(row.window)} | {row.net_pnl} |"
            f" {row.friction_skips} | {_yes_no(row.all_cities_negative)} |"
        )
    lines.append("")
    for window in windows:
        if not verdict.friction_floor.get(window):
            continue
        base = by_key[(REFERENCE_BANKROLL, window)]
        sensitivity = by_key[(SENSITIVITY_BANKROLL, window)]
        lines.append(
            f"{window}: $20000 all-cities-negative flips at $500 while friction skips collapse"
            f" ({base.friction_skips} -> {sensitivity.friction_skips});"
            " friction-floor-driven, not an edge failure"
        )
        lines.append("")

    _render_per_city(runs, windows, lines)

    lines.append("## depth sensitivity")
    lines.append("")
    lines.append(
        "| window | multiplier | net_pnl | friction_skips | all_cities_negative |"
        " friction_bound | stop_check |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in verdict.depth:
        if row.friction_bound is None:
            bound = "-"
            stop = "-"
        else:
            bound = f"friction_bound={'true' if row.friction_bound else 'false'}"
            stop = "excluded" if row.friction_bound else "counted"
        lines.append(
            f"| {row.window} | {row.multiplier}x | {row.net_pnl} | {row.friction_skips} |"
            f" {_yes_no(row.all_cities_negative)} | {bound} | {stop} |"
        )
    lines.append("")
    for window in windows:
        rows_w = [d for d in verdict.depth if d.window == window]
        base_survives = any(
            d.multiplier == BASE_MULTIPLIER and not d.all_cities_negative for d in rows_w
        )
        for row in rows_w:
            if row.friction_bound and base_survives:
                lines.append(
                    f"{window} {row.multiplier}x negative verdict is friction_bound=true with"
                    " 1.0x survival: fidelity flag, not a survival stop"
                )
                lines.append("")

    lines.append("## sigma sensitivity")
    lines.append("")
    lines.append("| window | sigma | net_pnl | all_cities_negative |")
    lines.append("| --- | --- | --- | --- |")
    for row in verdict.sigma:
        lines.append(
            f"| {row.window} | {row.multiplier} | {row.net_pnl} |"
            f" {_yes_no(row.all_cities_negative)} |"
        )
    lines.append("")

    lines.append("## fidelity cross-check")
    lines.append("")
    if not verdict.fidelity_ran:
        lines.append("fidelity cross-check not run; the survival verdict is not cross-checked")
    else:
        lines.append(
            "| city | lead_bucket | regime | n_snapshots | median_delta | threshold | bound |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for prob in verdict.prob_rows:
            lines.append(
                f"| {prob.city} | {prob.lead_bucket} | {prob.regime} | {prob.n_snapshots} |"
                f" {prob.median_delta} | {prob.threshold} |"
                f" {'pass' if prob.passed else 'fail'} |"
            )
        lines.append("")
        lines.append("| city | strategy | primary_signed_fills | crosscheck_signed_fills | bound |")
        lines.append("| --- | --- | --- | --- | --- |")
        for fill in verdict.fill_rows:
            lines.append(
                f"| {fill.city} | {fill.strategy} | {fill.primary} | {fill.crosscheck} |"
                f" {'pass' if fill.passed else 'fail'} |"
            )
        lines.append("")
        lines.append(
            "fidelity bounds tripped: survival verdict invalidated"
            if verdict.invalidated
            else "fidelity: pass"
        )
    lines.append("")

    lines.append("## verdict")
    lines.append("")
    for window, window_verdict in verdict.window_verdicts.items():
        lines.append(f"{window}: {window_verdict}")
    if verdict.era_flip:
        lines.append(
            "strategy-era flip: survival disagrees between in_sample (counterfactual=true) and"
            " out_of_sample (counterfactual=false); reported as a counterfactual-vs-live"
            " fidelity flag, no single verdict"
        )
    if verdict.fidelity_ran and verdict.invalidated:
        lines.append("fidelity bounds tripped: survival verdict invalidated")
    if not verdict.fidelity_ran:
        lines.append("fidelity cross-check not run")
    lines.append(f"verdict: {verdict.final}")
    lines.append("")
    return "\n".join(lines)


def write_report(
    runs: list[ScoredRun],
    orders: list[BacktestOrder],
    fills: list[BacktestFill],
    out_dir: Path,
    fidelity: FidelityInputs | None = None,
) -> Path:
    if len(fills) != len(orders):
        raise ValueError(f"length mismatch: orders={len(orders)} fills={len(fills)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_orders([_order_record(o, f) for o, f in zip(orders, fills)], out_dir)

    verdict = evaluate_runs(runs, fidelity)
    windows = _windows(runs)
    plot_pnls: dict[str, Decimal] = {}
    if windows:
        plot_pnls = _city_pnls(
            _select(runs, windows[0], REFERENCE_BANKROLL, BASE_MULTIPLIER, BASE_MULTIPLIER)
        )
    markdown = _render_markdown(runs, verdict, _plot_note(plot_pnls, out_dir))
    path = out_dir / "summary.md"
    path.write_text(markdown)
    return path
