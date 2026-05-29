from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from sqlalchemy import func, select

from bot.config import get_settings
from bot.execution.order_reconciler import (
    KalshiPosition,
    local_position_aggregate,
    poll_fills,
    poll_open_orders,
    poll_positions,
    reconcile_fills_into_demo_orders,
    upsert_exchange_record,
)
from bot.kalshi_client import KalshiDemoClient
from bot.storage.sqlite import DemoOrder as DemoOrderRow
from bot.storage.sqlite import make_engine, make_session_factory


logger = logging.getLogger(__name__)


def _find_orphans(session_factory, positions: list[KalshiPosition]) -> list[KalshiPosition]:
    orphans: list[KalshiPosition] = []
    with session_factory() as session:
        for pos in positions:
            if pos.position_fp == 0:
                continue
            local = local_position_aggregate(session, pos.ticker)
            if local != pos.position_fp:
                orphans.append(pos)
    return orphans


async def _backfill_ticker(
    client: KalshiDemoClient,
    session_factory,
    ticker: str,
) -> tuple[int, int]:
    orders = await poll_open_orders(client, watermark=None, ticker=ticker)
    fills = await poll_fills(client, watermark=None, ticker=ticker)

    rows_before: int
    rows_after: int
    with session_factory() as session:
        rows_before = (
            session.scalar(
                select(func.count(DemoOrderRow.id)).where(DemoOrderRow.market_ticker == ticker)
            )
            or 0
        )
        for record in orders:
            upsert_exchange_record(session, record)
        reconcile_fills_into_demo_orders(session, fills, orders)
        session.commit()
        rows_after = (
            session.scalar(
                select(func.count(DemoOrderRow.id)).where(DemoOrderRow.market_ticker == ticker)
            )
            or 0
        )
    return len(orders), max(0, rows_after - rows_before)


async def run(*, dry_run: bool, db_path: Path) -> int:
    settings = get_settings()
    if not settings.kalshi_demo_key_id or settings.kalshi_demo_private_key_path is None:
        sys.stderr.write(
            "backfill_from_kalshi: KALSHI_DEMO_KEY_ID and KALSHI_DEMO_PRIVATE_KEY_PATH must be set\n"
        )
        return 2
    if not settings.kalshi_demo_private_key_path.exists():
        sys.stderr.write(
            f"backfill_from_kalshi: private key not found at {settings.kalshi_demo_private_key_path}\n"
        )
        return 2

    engine = make_engine(str(db_path))
    session_factory = make_session_factory(engine)
    client = KalshiDemoClient(settings)
    await client.aopen()
    try:
        positions = await poll_positions(client)
        orphans = _find_orphans(session_factory, positions)

        if dry_run:
            print(f"dry_run orphans={len(orphans)}")
            for pos in orphans:
                with session_factory() as session:
                    local = local_position_aggregate(session, pos.ticker)
                print(f"  ticker={pos.ticker} kalshi={pos.position_fp} local={local}")
            return 0

        reconciled = 0
        rows_inserted = 0
        for pos in orphans:
            orders_count, inserted = await _backfill_ticker(client, session_factory, pos.ticker)
            print(f"backfilled ticker={pos.ticker} orders={orders_count} rows_inserted={inserted}")
            reconciled += 1
            rows_inserted += inserted

        remaining = _find_orphans(session_factory, await poll_positions(client))
        print(
            f"summary tickers_reconciled={reconciled} rows_inserted={rows_inserted} "
            f"remaining_mismatches={len(remaining)}"
        )
        for pos in remaining:
            with session_factory() as session:
                local = local_position_aggregate(session, pos.ticker)
            print(f"  unresolved ticker={pos.ticker} kalshi={pos.position_fp} local={local}")
        return 0 if not remaining else 1
    finally:
        await client.aclose()
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="scripts.reconcile.backfill_from_kalshi")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--db-path", type=Path, default=Path("data/state.db"))
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    exit_code = asyncio.run(run(dry_run=args.dry_run, db_path=args.db_path))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
