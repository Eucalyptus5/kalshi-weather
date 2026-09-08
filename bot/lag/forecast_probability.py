from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

import numpy as np

from bot.backtest.historical_open_meteo import (
    MEMBER_Z,
    MEMBER_Z_STD,
    SpreadCalibration,
    day_end_utc,
)
from bot.forecast.cdf import EnsembleCDF
from bot.lag.fee_floor import BAR_CONTEXT
from bot.lag.forecast_classes import ClassRecord
from bot.lag.forecast_sample import SampleLeg
from bot.markets.parser import parse_ticker, resolve_event_kinds
from bot.validation.scoring import RATE_QUANTUM, brier_score


NATIVE_XND = "native_xnd"
EXTERNAL_CALIBRATION = "external_calibration"
SIGMA_MULTIPLIERS: tuple[Decimal, ...] = (Decimal("0.5"), Decimal("1"), Decimal("2"))
LADDER_SUM_TOLERANCE: Decimal = Decimal("1e-9")
# A declared floor, not a rounding detail: EnsembleCDF convolves the members with norm.cdf(z /
# smoothing), so every class prices the bracket at sqrt(sigma^2 + 1), never at sigma.
SMOOTHING = 1.0

ClassKey = tuple[str, date, int]


@dataclass(frozen=True, slots=True, kw_only=True)
class LadderRung:
    ticker: str
    kind: str
    strike_lo: Decimal
    strike_hi: Decimal | None


@dataclass(frozen=True, slots=True, kw_only=True)
class LadderSum:
    event: str
    rungs: int
    total: Decimal
    ok: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassProbability:
    ticker: str
    series: str
    station: str
    event_date: date
    lead_hours: int
    split: str
    kind: str
    strike_lo: Decimal
    strike_hi: Decimal | None
    entry_price: Decimal
    result: str
    forecast_class: str
    member: str
    window_basis: str
    daily_high_f: Decimal
    sigma_f: Decimal
    effective_sigma_f: Decimal
    sigma_source: str
    members_n: int
    class_probability: Decimal
    outcome: int
    sensitivity: Mapping[Decimal, Decimal]

    @property
    def key(self) -> tuple[str, int, str, str]:
        return (self.ticker, self.lead_hours, self.forecast_class, self.member)


@dataclass(frozen=True, slots=True, kw_only=True)
class SigmaSourceTally:
    forecast_class: str
    native_xnd: int
    external_calibration: int


# scripts/build_spread_calibration.py built the buckets under day_end_utc(valid_date, tz) minus
# issuance, so measuring the lead from close_time instead would query a table never indexed that
# way, and the 60 seconds between the two lands on the 8-24h/24-72h boundary at the 24h leg.
def decision_lead(leg: SampleLeg) -> timedelta:
    return day_end_utc(leg.event_date, leg.timezone) - leg.as_of


def pseudo_members(daily_high: Decimal, sigma: Decimal) -> np.ndarray:
    return float(daily_high) + float(sigma) * MEMBER_Z / MEMBER_Z_STD


def class_cdf(daily_high: Decimal, sigma: Decimal) -> EnsembleCDF:
    return EnsembleCDF.from_members(pseudo_members(daily_high, sigma), smoothing=SMOOTHING)


def sigma_for(
    record: ClassRecord, leg: SampleLeg, calibration: SpreadCalibration
) -> tuple[Decimal, str]:
    if record.native_sigma_f is not None:
        return record.native_sigma_f, NATIVE_XND
    return Decimal(str(calibration.sigma_for(decision_lead(leg)))), EXTERNAL_CALIBRATION


def effective_sigma(sigma: Decimal) -> Decimal:
    return BAR_CONTEXT.sqrt(BAR_CONTEXT.add(BAR_CONTEXT.multiply(sigma, sigma), Decimal(1)))


def sensitivity_band(sigma: Decimal) -> dict[Decimal, Decimal]:
    base = effective_sigma(sigma)
    return {
        multiplier: BAR_CONTEXT.divide(
            effective_sigma(BAR_CONTEXT.multiply(multiplier, sigma)), base
        )
        for multiplier in SIGMA_MULTIPLIERS
    }


def event_probability(
    kind: str, strike_lo: Decimal, strike_hi: Decimal | None, cdf: EnsembleCDF
) -> Decimal:
    low = float(strike_lo)
    if kind == "bracket":
        raw = cdf.prob_range(low - 0.5, float(strike_hi) + 0.5)
    elif kind == "above":
        raw = cdf.prob_range(low + 0.5, math.inf)
    elif kind == "below":
        raw = cdf.prob_range(-math.inf, low - 0.5)
    else:
        raise ValueError(f"unknown ticker kind {kind!r}")
    return Decimal(str(raw))


def settles_yes(
    kind: str, strike_lo: Decimal, strike_hi: Decimal | None, observed_high: Decimal
) -> bool:
    if kind == "bracket":
        return strike_lo <= observed_high <= strike_hi
    if kind == "above":
        return observed_high > strike_lo
    if kind == "below":
        return observed_high < strike_lo
    raise ValueError(f"unknown ticker kind {kind!r}")


def event_ladder(tickers: Sequence[str]) -> tuple[LadderRung, ...]:
    parsed = resolve_event_kinds([parse_ticker(ticker) for ticker in sorted(tickers)])
    return tuple(
        LadderRung(
            ticker=row.raw,
            kind=row.kind,
            strike_lo=row.strikes[0],
            strike_hi=row.strikes[1] if len(row.strikes) == 2 else None,
        )
        for row in parsed
    )


def ladder_sum(ladder: Sequence[LadderRung], cdf: EnsembleCDF, *, event: str) -> LadderSum:
    total = Decimal(0)
    for rung in ladder:
        total = BAR_CONTEXT.add(
            total, event_probability(rung.kind, rung.strike_lo, rung.strike_hi, cdf)
        )
    return LadderSum(
        event=event,
        rungs=len(ladder),
        total=total,
        ok=BAR_CONTEXT.subtract(total, Decimal(1)).copy_abs() <= LADDER_SUM_TOLERANCE,
    )


def probability_for(
    leg: SampleLeg, record: ClassRecord, calibration: SpreadCalibration
) -> ClassProbability:
    sigma, source = sigma_for(record, leg, calibration)
    cdf = class_cdf(record.daily_high_f, sigma)
    sensitivity = {
        multiplier: event_probability(
            leg.kind,
            leg.strike_lo,
            leg.strike_hi,
            class_cdf(record.daily_high_f, BAR_CONTEXT.multiply(multiplier, sigma)),
        )
        for multiplier in SIGMA_MULTIPLIERS
    }
    return ClassProbability(
        ticker=leg.ticker,
        series=leg.series,
        station=leg.station,
        event_date=leg.event_date,
        lead_hours=leg.lead_hours,
        split=leg.split,
        kind=leg.kind,
        strike_lo=leg.strike_lo,
        strike_hi=leg.strike_hi,
        entry_price=leg.entry_price,
        result=leg.result,
        forecast_class=record.forecast_class,
        member=record.member,
        window_basis=record.window_basis,
        daily_high_f=record.daily_high_f,
        sigma_f=sigma,
        effective_sigma_f=effective_sigma(sigma),
        sigma_source=source,
        members_n=int(cdf.members.size),
        class_probability=event_probability(leg.kind, leg.strike_lo, leg.strike_hi, cdf),
        outcome=1 if leg.result == "yes" else 0,
        sensitivity=sensitivity,
    )


def probabilities(
    legs: Sequence[SampleLeg],
    records: Sequence[ClassRecord],
    calibration: SpreadCalibration,
) -> list[ClassProbability]:
    by_key: dict[ClassKey, list[ClassRecord]] = {}
    for record in records:
        by_key.setdefault((record.station, record.event_date, record.lead_hours), []).append(record)
    return [
        probability_for(leg, record, calibration)
        for leg in legs
        for record in by_key.get((leg.station, leg.event_date, leg.lead_hours), ())
    ]


def sigma_source_tally(rows: Sequence[ClassProbability]) -> list[SigmaSourceTally]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = counts.setdefault(row.forecast_class, {NATIVE_XND: 0, EXTERNAL_CALIBRATION: 0})
        bucket[row.sigma_source] += 1
    return [
        SigmaSourceTally(
            forecast_class=forecast_class,
            native_xnd=bucket[NATIVE_XND],
            external_calibration=bucket[EXTERNAL_CALIBRATION],
        )
        for forecast_class, bucket in sorted(counts.items())
    ]


def class_brier(rows: Sequence[ClassProbability]) -> Decimal:
    return brier_score([row.class_probability for row in rows], [row.outcome for row in rows])


def baseline_brier(rows: Sequence[ClassProbability]) -> Decimal:
    return brier_score([row.entry_price for row in rows], [row.outcome for row in rows])


def brier_skill(model: Decimal, baseline: Decimal) -> Decimal:
    return BAR_CONTEXT.quantize(
        BAR_CONTEXT.subtract(Decimal(1), BAR_CONTEXT.divide(model, baseline)), RATE_QUANTUM
    )
