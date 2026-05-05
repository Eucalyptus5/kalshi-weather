from __future__ import annotations

import asyncio
import logging
import sys
from collections import defaultdict

from kalshi_python_async import ApiClient, Configuration, EventsApi, KalshiAuth, MarketApi

from bot.config import get_settings

logger = logging.getLogger(__name__)


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

    private_key_pem = settings.kalshi_demo_private_key_path.read_text()
    config = Configuration(host=settings.kalshi_demo_api_base)

    async with ApiClient(configuration=config) as api_client:
        api_client.kalshi_auth = KalshiAuth(
            key_id=settings.kalshi_demo_key_id,
            private_key_pem=private_key_pem,
        )

        markets_api = MarketApi(api_client)
        events_api = EventsApi(api_client)

        logger.info("fetching open markets")
        response = await markets_api.get_markets(status="open", limit=1000)
        markets = [m for m in response.markets if m.ticker.startswith("KXHIGH")]
        logger.info("fetched %d KXHIGH markets", len(markets))

        by_event: dict[str, list] = defaultdict(list)
        for m in markets:
            by_event[m.event_ticker].append(m)

        for event_ticker in list(by_event)[:5]:
            event_response = await events_api.get_event(event_ticker, with_nested_markets=True)
            event = event_response.event
            print(f"event {event_ticker}: {getattr(event, 'title', '')}")
            for market in by_event[event_ticker]:
                print(
                    f"  {market.ticker}  "
                    f"yes_sub_title={getattr(market, 'yes_sub_title', '')}  "
                    f"yes_bid={getattr(market, 'yes_bid', '')}  "
                    f"yes_ask={getattr(market, 'yes_ask', '')}"
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(run())
