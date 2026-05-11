from __future__ import annotations

import asyncio
import logging
import sys
from collections import defaultdict

from bot.config import get_settings
from bot.kalshi_client import KalshiDemoClient


logger = logging.getLogger(__name__)

SERIES: tuple[str, ...] = ("KXHIGHDEN",)


async def run() -> None:
    settings = get_settings()

    if not settings.kalshi_demo_key_id or not settings.kalshi_demo_private_key_path:
        sys.stderr.write(
            "smoke_demo: KALSHI_DEMO_KEY_ID and KALSHI_DEMO_PRIVATE_KEY_PATH must be set\n"
        )
        sys.exit(2)

    if not settings.kalshi_demo_private_key_path.exists():
        sys.stderr.write(
            f"smoke_demo: private key file not found at {settings.kalshi_demo_private_key_path}\n"
        )
        sys.exit(2)

    client = KalshiDemoClient(settings)
    await client.aopen()
    try:
        for series in SERIES:
            logger.info("fetching open %s markets", series)
            markets = await client.list_open_markets_for_series(series)
            logger.info("fetched %d %s markets", len(markets), series)

            by_event: dict[str, list] = defaultdict(list)
            for m in markets:
                by_event[m.event_ticker].append(m)

            print(f"series {series}")
            for event_ticker in list(by_event)[:5]:
                print(f"  event {event_ticker}")
                for market in by_event[event_ticker]:
                    print(
                        f"    {market.ticker}  yes_bid={market.yes_bid}  "
                        f"yes_ask={market.yes_ask}  close={market.close_time}"
                    )
    finally:
        await client.aclose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(run())
