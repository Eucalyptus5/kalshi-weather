import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from types import MappingProxyType

import numpy as np
from scipy.optimize import nnls

from bot.lag.fee_floor import BAR_CONTEXT


DISCOVERY = "discovery"
WEIGHT_QUANTUM: Decimal = Decimal("1E-12")


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassScore:
    ticker: str
    event_date: date
    split: str
    member: str
    probability: Decimal
    outcome: int


@dataclass(frozen=True, slots=True, kw_only=True)
class BlendWeights:
    weights: Mapping[str, Decimal]
    members: tuple[str, ...]
    fitted_on_event_days: int
    fitted_on_split: str
    sha256: str


# Deliberately not imported from bot.replay.run_scope or bot.lag.r0_universe: both carry a copy of
# these three lines already, and bot/forecast should not pull a ladder-validity module to hash.
def freeze_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def fit_weights(records: Sequence[ClassScore]) -> BlendWeights:
    if not records:
        raise ValueError("no records to fit the blend on")
    for record in records:
        if record.split != DISCOVERY:
            raise ValueError(
                f"refusing to fit on a {record.split} record for event day "
                f"{record.event_date.isoformat()}"
            )
    members = tuple(sorted({record.member for record in records}))
    by_ticker: dict[str, list[ClassScore]] = {}
    for record in records:
        by_ticker.setdefault(record.ticker, []).append(record)

    rows: list[list[float]] = []
    outcomes: list[float] = []
    for ticker in sorted(by_ticker):
        leg = sorted(by_ticker[ticker], key=lambda record: record.member)
        # A leg missing a member cannot be blended, and dropping it here would move the member set
        # the weights are recorded against without saying so.
        if tuple(record.member for record in leg) != members:
            raise ValueError(
                f"{ticker} carries members {[record.member for record in leg]}, not {list(members)}"
            )
        rows.append([float(record.probability) for record in leg])
        outcomes.append(float(leg[0].outcome))

    raw, _ = nnls(np.array(rows), np.array(outcomes))
    raw_weights = [Decimal(str(float(value))) for value in raw]
    for member, weight in zip(members, raw_weights, strict=True):
        if weight < 0:
            raise ValueError(f"nnls returned a negative weight {weight} for {member}")
    raw_total = Decimal(0)
    for weight in raw_weights:
        raw_total = BAR_CONTEXT.add(raw_total, weight)
    if raw_total == 0:
        raise ValueError("nnls assigned no weight to any member")

    scaled = {
        member: BAR_CONTEXT.quantize(BAR_CONTEXT.divide(weight, raw_total), WEIGHT_QUANTUM)
        for member, weight in zip(members, raw_weights, strict=True)
    }
    # The residual lands on the largest weight so it cannot be driven negative.
    anchor = min(members, key=lambda member: (-scaled[member], member))
    others = Decimal(0)
    for member in members:
        if member != anchor:
            others = BAR_CONTEXT.add(others, scaled[member])
    weights = dict(scaled)
    weights[anchor] = BAR_CONTEXT.subtract(Decimal(1), others)

    event_days = len({record.event_date for record in records})
    payload = {
        "members": list(members),
        "weights": {member: str(weights[member]) for member in members},
        "fitted_on_event_days": event_days,
        "fitted_on_split": DISCOVERY,
    }
    return BlendWeights(
        weights=MappingProxyType(weights),
        members=members,
        fitted_on_event_days=event_days,
        fitted_on_split=DISCOVERY,
        sha256=freeze_digest(payload),
    )
