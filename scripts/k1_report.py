from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.lag.fee_floor import PUBLISHED_MAKER_RATE  # noqa: E402
from bot.lag.maker_headroom import (  # noqa: E402
    HALF_TICK_CAPTURE_CENTS,
    NO_MAKER_FEE_RATE,
    HeadroomCell,
    closed_on_arithmetic,
    headroom_grid,
)


PUBLISHED_REGIME = "published_maker_rate"
NO_FEE_REGIME = "no_maker_fee"
REGIMES = (PUBLISHED_REGIME, NO_FEE_REGIME)
CLOSED = "CLOSED"
OPEN = "OPEN"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="read the K1 maker headroom grid against the half tick capture"
    )
    parser.add_argument(
        "--regime",
        required=True,
        choices=REGIMES,
        help="which schedule the venue carries for weather series, and so which grid decides",
    )
    parser.add_argument(
        "--rate",
        type=Decimal,
        default=PUBLISHED_MAKER_RATE,
        help="the rate the published_maker_rate grid runs at",
    )
    return parser


def cell_payload(cell: HeadroomCell) -> dict:
    return {
        "size": str(cell.size),
        "size_kind": cell.size_kind,
        "price": str(cell.price),
        "aggregate_fee": str(cell.aggregate_fee),
        "fee_cents_per_contract": str(cell.fee_cents_per_contract),
        "headroom_cents_per_contract": str(cell.headroom_cents_per_contract),
        "leaves_headroom": cell.leaves_headroom,
    }


# The counts split on the same multiplication the predicate runs on rather than on the rounded
# per-contract figures the cells report, so the three always sum to the cell count.
def cell_counts(cells: Sequence[HeadroomCell]) -> dict:
    return {
        "positive": sum(1 for cell in cells if cell.leaves_headroom),
        "zero": sum(1 for cell in cells if Decimal(200) * cell.aggregate_fee == cell.size),
        "negative": sum(1 for cell in cells if Decimal(200) * cell.aggregate_fee > cell.size),
    }


def regime_payload(rate: Decimal) -> dict:
    cells = headroom_grid(rate)
    return {
        "rate": str(rate),
        "cells": [cell_payload(cell) for cell in cells],
        "counts": cell_counts(cells),
        "closed_on_arithmetic": closed_on_arithmetic(cells),
    }


def report(regime: str, rate: Decimal) -> dict:
    regimes = {
        PUBLISHED_REGIME: regime_payload(rate),
        NO_FEE_REGIME: regime_payload(NO_MAKER_FEE_RATE),
    }
    return {
        "selected_regime": regime,
        "half_tick_capture_cents": str(HALF_TICK_CAPTURE_CENTS),
        "regimes": regimes,
        "verdict": CLOSED if regimes[regime]["closed_on_arithmetic"] else OPEN,
    }


def run(args: argparse.Namespace) -> int:
    print(json.dumps(report(args.regime, args.rate), indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
