from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from bot.lag.rest_agreement import SeriesAgreement, check_rest_agreement
from bot.storage.sqlite import (
    Base,
    OrderbookSnapshot,
    WsBookEvent,
    WsGap,
    make_engine,
    make_session_factory,
)


UTC = timezone.utc
T0 = datetime(2026, 7, 10, 15, 0, 0, tzinfo=UTC)
DEN_TICKER = "KXHIGHDEN-26JUL10-T85"
CHI_TICKER = "KXHIGHCHI-26JUL10-T90"


@pytest.fixture
def db(tmp_path: Path) -> tuple[Path, sessionmaker[Session]]:
    path = tmp_path / "state.db"
    engine = make_engine(path)
    Base.metadata.create_all(engine)
    return path, make_session_factory(engine)


def _snapshot(
    factory: sessionmaker[Session],
    *,
    received_at: datetime,
    seq: int,
    yes: list[tuple[str, str]],
    no: list[tuple[str, str]],
    ticker: str = DEN_TICKER,
) -> None:
    rows = [
        WsBookEvent(
            ticker=ticker,
            received_at=received_at,
            seq=seq,
            side=side,
            price=Decimal(price),
            size=Decimal(size),
            is_snapshot=True,
        )
        for side, levels in (("yes", yes), ("no", no))
        for price, size in levels
    ]
    with factory() as session:
        session.add_all(rows)
        session.commit()


def _gap(
    factory: sessionmaker[Session],
    *,
    detected_at: datetime,
    ticker: str,
    reason: str = "seq_skip",
) -> None:
    with factory() as session:
        session.add(WsGap(ticker=ticker, detected_at=detected_at, last_seq=0, reason=reason))
        session.commit()


def _rest(
    factory: sessionmaker[Session],
    *,
    snapshot_at: datetime,
    yes_bid: Decimal,
    yes_ask: Decimal,
    ticker: str = DEN_TICKER,
    yes_bid_depth: int | None = None,
    yes_ask_depth: int | None = None,
) -> None:
    with factory() as session:
        session.add(
            OrderbookSnapshot(
                ticker=ticker,
                snapshot_at=snapshot_at,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=Decimal("1") - yes_ask,
                no_ask=Decimal("1") - yes_bid,
                yes_bid_depth=yes_bid_depth,
                yes_ask_depth=yes_ask_depth,
            )
        )
        session.commit()


def _seed_base_book(factory: sessionmaker[Session], ticker: str = DEN_TICKER) -> None:
    _snapshot(
        factory,
        received_at=T0,
        seq=10,
        yes=[("0.400000", "50"), ("0.350000", "20")],
        no=[("0.500000", "30"), ("0.450000", "10")],
        ticker=ticker,
    )


def test_matching_rest_row_agrees_with_fraction_one(db) -> None:
    path, factory = db
    _seed_base_book(factory)
    _rest(
        factory,
        snapshot_at=T0 + timedelta(seconds=60),
        yes_bid=Decimal("0.4"),
        yes_ask=Decimal("0.5"),
    )

    report = check_rest_agreement(path, ["KXHIGHDEN"])

    assert len(report.comparisons) == 1
    c = report.comparisons[0]
    assert c.verdict == "agree"
    assert c.ticker == DEN_TICKER
    assert c.snapshot_at == T0 + timedelta(seconds=60)
    assert c.gap_reason is None
    assert c.ws is not None
    assert report.per_series == [
        SeriesAgreement(
            label="KXHIGHDEN",
            aligned_n=1,
            agree_n=1,
            disagree_n=0,
            no_coverage_n=0,
            gap_n=0,
            fraction=Decimal("1"),
        )
    ]
    assert report.pooled.label == "pooled"
    assert report.pooled.aligned_n == 1
    assert report.pooled.fraction == Decimal("1")
    assert report.disagreements == []


def test_mismatched_yes_bid_disagrees_and_is_surfaced(db) -> None:
    path, factory = db
    _seed_base_book(factory)
    _rest(
        factory,
        snapshot_at=T0 + timedelta(seconds=60),
        yes_bid=Decimal("0.39"),
        yes_ask=Decimal("0.5"),
        yes_bid_depth=7,
    )
    _rest(
        factory,
        snapshot_at=T0 + timedelta(seconds=120),
        yes_bid=Decimal("0.4"),
        yes_ask=Decimal("0.5"),
    )

    report = check_rest_agreement(path, ["KXHIGHDEN"])

    assert [c.verdict for c in report.comparisons] == ["disagree", "agree"]
    assert len(report.disagreements) == 1
    d = report.disagreements[0]
    assert d is report.comparisons[0]
    assert d.rest.yes_bid == Decimal("0.39")
    assert d.rest.no_ask == Decimal("0.61")
    assert d.rest.yes_bid_depth == 7
    assert d.ws is not None
    assert d.ws.yes_bid == Decimal("0.400000")
    assert d.ws.yes_bid_depth == 50
    s = report.per_series[0]
    assert s.aligned_n == 2
    assert s.agree_n == 1
    assert s.disagree_n == 1
    assert s.fraction == Decimal("0.5")


def test_rest_row_before_ws_coverage_is_excluded(db) -> None:
    path, factory = db
    _seed_base_book(factory)
    _rest(
        factory,
        snapshot_at=T0 - timedelta(seconds=60),
        yes_bid=Decimal("0.4"),
        yes_ask=Decimal("0.5"),
    )

    report = check_rest_agreement(path, ["KXHIGHDEN"])

    assert [c.verdict for c in report.comparisons] == ["no_coverage"]
    assert report.comparisons[0].ws is None
    s = report.per_series[0]
    assert s.no_coverage_n == 1
    assert s.aligned_n == 0
    assert s.fraction is None
    assert report.pooled.fraction is None


def test_gap_window_excluded_without_diluting(db) -> None:
    path, factory = db
    _seed_base_book(factory)
    _gap(factory, detected_at=T0 + timedelta(seconds=10), ticker=DEN_TICKER)
    _rest(
        factory,
        snapshot_at=T0 + timedelta(seconds=30),
        yes_bid=Decimal("0.4"),
        yes_ask=Decimal("0.5"),
    )
    _snapshot(
        factory,
        received_at=T0 + timedelta(seconds=60),
        seq=20,
        yes=[("0.400000", "50")],
        no=[("0.500000", "30")],
    )
    _rest(
        factory,
        snapshot_at=T0 + timedelta(seconds=90),
        yes_bid=Decimal("0.4"),
        yes_ask=Decimal("0.5"),
    )

    report = check_rest_agreement(path, ["KXHIGHDEN"])

    assert [c.verdict for c in report.comparisons] == ["gap", "agree"]
    gap = report.comparisons[0]
    assert gap.ws is None
    assert gap.gap_reason
    assert "seq_skip" in gap.gap_reason
    s = report.per_series[0]
    assert s.gap_n == 1
    assert s.aligned_n == 1
    assert s.agree_n == 1
    assert s.fraction == Decimal("1")


def test_decimal_scale_differences_still_agree(db) -> None:
    path, factory = db
    _snapshot(factory, received_at=T0, seq=10, yes=[("0.500000", "12")], no=[("0.500000", "9")])
    _rest(
        factory,
        snapshot_at=T0 + timedelta(seconds=30),
        yes_bid=Decimal("0.5"),
        yes_ask=Decimal("0.5"),
    )

    report = check_rest_agreement(path, ["KXHIGHDEN"])

    assert report.comparisons[0].verdict == "agree"
    assert report.per_series[0].fraction == Decimal("1")


def test_depth_mismatch_does_not_affect_verdict(db) -> None:
    path, factory = db
    _seed_base_book(factory)
    _rest(
        factory,
        snapshot_at=T0 + timedelta(seconds=60),
        yes_bid=Decimal("0.4"),
        yes_ask=Decimal("0.5"),
        yes_bid_depth=999,
        yes_ask_depth=1,
    )

    report = check_rest_agreement(path, ["KXHIGHDEN"])

    c = report.comparisons[0]
    assert c.verdict == "agree"
    assert c.rest.yes_bid_depth == 999
    assert c.rest.yes_ask_depth == 1
    assert c.ws is not None
    assert c.ws.yes_bid_depth == 50
    assert c.ws.yes_ask_depth == 30


def test_series_aggregate_separately_and_unrequested_excluded(db) -> None:
    path, factory = db
    _seed_base_book(factory)
    _seed_base_book(factory, ticker=CHI_TICKER)
    _rest(
        factory,
        snapshot_at=T0 + timedelta(seconds=60),
        yes_bid=Decimal("0.4"),
        yes_ask=Decimal("0.5"),
    )
    _rest(
        factory,
        snapshot_at=T0 + timedelta(seconds=60),
        yes_bid=Decimal("0.3"),
        yes_ask=Decimal("0.5"),
        ticker=CHI_TICKER,
    )
    _rest(
        factory,
        snapshot_at=T0 + timedelta(seconds=60),
        yes_bid=Decimal("0.4"),
        yes_ask=Decimal("0.5"),
        ticker="KXHIGHNY-26JUL10-T90",
    )

    report = check_rest_agreement(path, ["KXHIGHDEN", "KXHIGHCHI"])

    assert [s.label for s in report.per_series] == ["KXHIGHCHI", "KXHIGHDEN"]
    assert all(not c.ticker.startswith("KXHIGHNY") for c in report.comparisons)
    chi, den = report.per_series
    assert chi.aligned_n == 1
    assert chi.disagree_n == 1
    assert chi.fraction == Decimal("0")
    assert den.aligned_n == 1
    assert den.agree_n == 1
    assert den.fraction == Decimal("1")
    assert report.pooled.aligned_n == 2
    assert report.pooled.agree_n == 1
    assert report.pooled.fraction == Decimal("0.5")
    assert len(report.disagreements) == 1
    assert report.disagreements[0].ticker == CHI_TICKER


def test_empty_no_side_yes_ask_one_agrees(db) -> None:
    path, factory = db
    _snapshot(factory, received_at=T0, seq=10, yes=[("0.400000", "50")], no=[])
    _rest(
        factory,
        snapshot_at=T0 + timedelta(seconds=30),
        yes_bid=Decimal("0.4"),
        yes_ask=Decimal("1"),
    )

    report = check_rest_agreement(path, ["KXHIGHDEN"])

    c = report.comparisons[0]
    assert c.verdict == "agree"
    assert c.ws is not None
    assert c.ws.yes_ask == Decimal("1")
    assert c.ws.yes_ask_depth == 0
