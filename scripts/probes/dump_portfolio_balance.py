from __future__ import annotations

import asyncio
import json
import logging
import sys

from bot.config import get_settings
from bot.kalshi_client import KalshiDemoClient


logger = logging.getLogger(__name__)


async def run() -> None:
    settings = get_settings()
    if not settings.kalshi_demo_key_id or not settings.kalshi_demo_private_key_path:
        sys.stderr.write(
            "dump_portfolio_balance: KALSHI_DEMO_KEY_ID and KALSHI_DEMO_PRIVATE_KEY_PATH must be set\n"
        )
        sys.exit(2)
    if not settings.kalshi_demo_private_key_path.exists():
        sys.stderr.write(
            f"dump_portfolio_balance: private key file not found at {settings.kalshi_demo_private_key_path}\n"
        )
        sys.exit(2)

    client = KalshiDemoClient(settings)
    await client.aopen()
    try:
        response = await client.get_signed("/portfolio/balance")
        response.raise_for_status()
        payload = response.json()
    finally:
        await client.aclose()

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(run())
