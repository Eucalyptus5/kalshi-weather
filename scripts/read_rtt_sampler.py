from __future__ import annotations

import argparse
import asyncio
import logging
import socket
import sys
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.config import Settings, get_settings  # noqa: E402
from bot.kalshi_client import KalshiReadClient  # noqa: E402
from bot.lag.read_rtt import (  # noqa: E402
    ReadSample,
    append_sample,
    due_at,
    load_samples,
    reanchor_start,
    resume_point,
    sample_interval_seconds,
    summarize,
)

logger = logging.getLogger(__name__)

DEFAULT_TARGET = 200
DEFAULT_SPAN_HOURS = 24.0
DEFAULT_SERIES: tuple[str, ...] = ("KXHIGHDEN",)
DEFAULT_REFRESH_MINUTES = 60.0
API_PREFIX = "/trade-api/v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="signed market-data read round trip sampler")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--span-hours", type=float, default=DEFAULT_SPAN_HOURS)
    parser.add_argument(
        "--series",
        type=lambda s: tuple(p for p in s.split(",") if p),
        default=DEFAULT_SERIES,
    )
    parser.add_argument("--refresh-minutes", type=float, default=DEFAULT_REFRESH_MINUTES)
    return parser


async def open_tickers(client: KalshiReadClient, series: tuple[str, ...]) -> list[str]:
    tickers: set[str] = set()
    for s in series:
        tickers.update(m.ticker for m in await client.list_open_markets_for_series(s))
    return sorted(tickers)


async def sample_once(
    client: KalshiReadClient,
    ticker: str,
    sequence: int,
    api_host: str,
    source_host: str,
) -> ReadSample:
    requested_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    try:
        await client.get_orderbook(ticker)
    except httpx.HTTPStatusError as exc:
        outcome, status_code = "http_status", exc.response.status_code
    except httpx.HTTPError:
        outcome, status_code = "transport", None
    else:
        outcome, status_code = "ok", None
    return ReadSample(
        sequence=sequence,
        requested_at=requested_at,
        elapsed_s=time.perf_counter() - started,
        ticker=ticker,
        outcome=outcome,
        status_code=status_code,
        api_host=api_host,
        endpoint=f"GET {API_PREFIX}/markets/{ticker}/orderbook",
        source_host=source_host,
    )


async def run(args: argparse.Namespace, client: KalshiReadClient, settings: Settings) -> int:
    api_host = urllib.parse.urlparse(settings.kalshi_prod_api_base).hostname or ""
    source_host = socket.gethostname()
    interval_s = sample_interval_seconds(args.target, args.span_hours * 3600)
    refresh = timedelta(minutes=args.refresh_minutes)

    existing = load_samples(args.out) if args.out.exists() else []
    start, index = resume_point(existing, datetime.now(timezone.utc))

    try:
        tickers = await open_tickers(client, args.series)
    except httpx.HTTPError as exc:
        logger.error("read_rtt_ticker_fetch_failed series=%s error=%s", ",".join(args.series), exc)
        return 1
    if not tickers:
        logger.error("read_rtt_no_open_tickers series=%s", ",".join(args.series))
        return 1
    refreshed_at = datetime.now(timezone.utc)

    logger.info(
        "read_rtt_start api_host=%s source_host=%s target=%d interval_s=%.1f "
        "resume_index=%d tickers=%d out=%s",
        api_host,
        source_host,
        args.target,
        interval_s,
        index,
        len(tickers),
        args.out,
    )

    samples: list[ReadSample] = []
    while index < args.target:
        now = datetime.now(timezone.utc)
        # a restart or a stalled read must not replay every missed slot back to back against
        # the read path the recorder shares, so the cadence re-anchors instead of catching up
        start = reanchor_start(start, index, interval_s, now)
        wait_s = (due_at(start, index, interval_s) - now).total_seconds()
        if wait_s > 0:
            await asyncio.sleep(wait_s)
        if datetime.now(timezone.utc) - refreshed_at >= refresh:
            try:
                tickers = await open_tickers(client, args.series) or tickers
            except httpx.HTTPError as exc:
                logger.warning("read_rtt_ticker_refresh_failed error=%s", exc)
            refreshed_at = datetime.now(timezone.utc)

        sample = await sample_once(
            client, tickers[index % len(tickers)], index, api_host, source_host
        )
        append_sample(args.out, sample)
        samples.append(sample)
        logger.info(
            "read_rtt_sample seq=%d ticker=%s outcome=%s status=%s elapsed_s=%.4f",
            sample.sequence,
            sample.ticker,
            sample.outcome,
            sample.status_code,
            sample.elapsed_s,
        )
        index += 1

    summary = summarize(samples)
    logger.info(
        "read_rtt_summary n_total=%d n_ok=%d n_error=%d p50_s=%s p90_s=%s min_s=%s max_s=%s "
        "hourly=%s",
        summary.n_total,
        summary.n_ok,
        summary.n_error,
        summary.p50_s,
        summary.p90_s,
        summary.min_s,
        summary.max_s,
        summary.hourly,
    )
    return 0


async def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = get_settings()
    client = KalshiReadClient(settings)
    await client.aopen()
    try:
        return await run(args, client, settings)
    finally:
        await client.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
