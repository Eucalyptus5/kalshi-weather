from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ReadSample:
    sequence: int
    requested_at: datetime
    elapsed_s: float
    ticker: str
    outcome: str
    status_code: int | None
    api_host: str
    endpoint: str
    source_host: str


@dataclass(frozen=True, slots=True)
class RttSummary:
    n_total: int
    n_ok: int
    n_error: int
    p50_s: float | None
    p90_s: float | None
    min_s: float | None
    max_s: float | None
    first_at: datetime | None
    last_at: datetime | None
    hourly: dict[int, int]


def encode_sample(sample: ReadSample) -> str:
    return json.dumps(
        {
            "sequence": sample.sequence,
            "requested_at": sample.requested_at.isoformat(),
            "elapsed_s": sample.elapsed_s,
            "ticker": sample.ticker,
            "outcome": sample.outcome,
            "status_code": sample.status_code,
            "api_host": sample.api_host,
            "endpoint": sample.endpoint,
            "source_host": sample.source_host,
        }
    )


def decode_sample(line: str) -> ReadSample:
    raw = json.loads(line)
    return ReadSample(
        sequence=raw["sequence"],
        requested_at=datetime.fromisoformat(raw["requested_at"]),
        elapsed_s=raw["elapsed_s"],
        ticker=raw["ticker"],
        outcome=raw["outcome"],
        status_code=raw["status_code"],
        api_host=raw["api_host"],
        endpoint=raw["endpoint"],
        source_host=raw["source_host"],
    )


def append_sample(path: Path, sample: ReadSample) -> None:
    with path.open("a") as handle:
        handle.write(encode_sample(sample) + "\n")
        handle.flush()


def load_samples(path: Path) -> list[ReadSample]:
    return [decode_sample(line) for line in path.read_text().splitlines() if line.strip()]


def sample_interval_seconds(target_samples: int, span_s: float) -> float:
    if target_samples < 1:
        raise ValueError(f"target_samples must be >= 1, got {target_samples}")
    if span_s <= 0:
        raise ValueError(f"span_s must be > 0, got {span_s}")
    return span_s / target_samples


def due_at(start: datetime, index: int, interval_s: float) -> datetime:
    return start + timedelta(seconds=index * interval_s)


def reanchor_start(start: datetime, index: int, interval_s: float, now: datetime) -> datetime:
    if now - due_at(start, index, interval_s) <= timedelta(seconds=interval_s):
        return start
    return now - timedelta(seconds=index * interval_s)


def resume_point(samples: list[ReadSample], fallback_start: datetime) -> tuple[datetime, int]:
    if not samples:
        return fallback_start, 0
    return samples[0].requested_at, max(s.sequence for s in samples) + 1


def quantile_seconds(samples: list[ReadSample], q: float) -> float | None:
    if not 0 < q <= 1:
        raise ValueError(f"q must satisfy 0 < q <= 1, got {q}")
    ok = sorted(s.elapsed_s for s in samples if s.outcome == "ok")
    if not ok:
        return None
    return ok[min(math.ceil(q * len(ok)), len(ok)) - 1]


def hourly_counts(samples: list[ReadSample]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for sample in samples:
        hour = sample.requested_at.hour
        counts[hour] = counts.get(hour, 0) + 1
    return dict(sorted(counts.items()))


def summarize(samples: list[ReadSample]) -> RttSummary:
    ok = [s.elapsed_s for s in samples if s.outcome == "ok"]
    stamps = [s.requested_at for s in samples]
    return RttSummary(
        n_total=len(samples),
        n_ok=len(ok),
        n_error=len(samples) - len(ok),
        p50_s=quantile_seconds(samples, 0.5),
        p90_s=quantile_seconds(samples, 0.9),
        min_s=min(ok) if ok else None,
        max_s=max(ok) if ok else None,
        first_at=min(stamps) if stamps else None,
        last_at=max(stamps) if stamps else None,
        hourly=hourly_counts(samples),
    )
