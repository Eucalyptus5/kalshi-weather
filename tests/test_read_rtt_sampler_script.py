from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

from bot.config import Settings
from bot.kalshi_client import KalshiReadClient
from bot.lag.read_rtt import ReadSample, append_sample, load_samples
from scripts.read_rtt_sampler import (
    DEFAULT_REFRESH_MINUTES,
    DEFAULT_SERIES,
    DEFAULT_SPAN_HOURS,
    DEFAULT_TARGET,
    build_parser,
    run,
)


PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"
TICKERS = ("KXHIGHDEN-26AUG11-T90", "KXHIGHDEN-26AUG11-T95")
ORDERBOOK = {"orderbook_fp": {"yes_dollars": [["0.4500", "12"]], "no_dollars": [["0.5300", "9"]]}}


@pytest.fixture(scope="module")
def rsa_pem(tmp_path_factory: pytest.TempPathFactory) -> Path:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption())
    path = tmp_path_factory.mktemp("kalshi") / "prod.pem"
    path.write_bytes(pem)
    return path


def _settings(pem_path: Path) -> Settings:
    return Settings(
        mode="paper",
        kalshi_prod_key_id="prod-key-id",
        kalshi_prod_private_key_path=pem_path,
    )


def _markets_payload() -> dict[str, object]:
    return {
        "markets": [
            {
                "ticker": t,
                "event_ticker": "KXHIGHDEN-26AUG11",
                "status": "open",
                "close_time": "2026-08-12T06:00:00Z",
                "yes_ask_dollars": "0.4700",
                "yes_bid_dollars": "0.4500",
            }
            for t in TICKERS
        ],
        "cursor": "",
    }


def _handler(orderbook: httpx.Response | Exception) -> object:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/orderbook"):
            if isinstance(orderbook, Exception):
                raise orderbook
            return httpx.Response(orderbook.status_code, json=ORDERBOOK)
        return httpx.Response(200, json=_markets_payload())

    return handle


async def _run(
    out: Path,
    pem_path: Path,
    orderbook: httpx.Response | Exception,
    target: int = 2,
    argv_extra: list[str] | None = None,
) -> int:
    argv = ["--out", str(out), "--target", str(target), "--span-hours", "0.0001"]
    args = build_parser().parse_args(argv + (argv_extra or []))
    settings = _settings(pem_path)
    transport = httpx.MockTransport(_handler(orderbook))
    async with httpx.AsyncClient(transport=transport, base_url=PROD_BASE) as http:
        client = KalshiReadClient(settings, http_client=http)
        await client.aopen()
        try:
            return await run(args, client, settings)
        finally:
            await client.aclose()


def test_parser_defaults() -> None:
    args = build_parser().parse_args(["--out", "samples.jsonl"])
    assert args.target == DEFAULT_TARGET
    assert args.span_hours == DEFAULT_SPAN_HOURS
    assert args.series == DEFAULT_SERIES
    assert args.refresh_minutes == DEFAULT_REFRESH_MINUTES


async def test_run_writes_one_record_per_sample(tmp_path: Path, rsa_pem: Path) -> None:
    out = tmp_path / "samples.jsonl"
    assert await _run(out, rsa_pem, httpx.Response(200), target=3) == 0

    samples = load_samples(out)
    assert [s.sequence for s in samples] == [0, 1, 2]
    assert [s.outcome for s in samples] == ["ok", "ok", "ok"]
    assert all(s.api_host == "api.elections.kalshi.com" for s in samples)
    assert all(s.source_host for s in samples)
    assert [s.endpoint for s in samples] == [
        f"GET /trade-api/v2/markets/{t}/orderbook" for t in (TICKERS[0], TICKERS[1], TICKERS[0])
    ]
    assert all(s.elapsed_s >= 0 for s in samples)


async def test_run_records_http_status_failures(tmp_path: Path, rsa_pem: Path) -> None:
    out = tmp_path / "samples.jsonl"
    assert await _run(out, rsa_pem, httpx.Response(503)) == 0

    samples = load_samples(out)
    assert len(samples) == 2
    assert all(s.outcome == "http_status" and s.status_code == 503 for s in samples)


async def test_run_records_transport_failures(tmp_path: Path, rsa_pem: Path) -> None:
    out = tmp_path / "samples.jsonl"
    assert await _run(out, rsa_pem, httpx.ConnectError("no route")) == 0

    samples = load_samples(out)
    assert len(samples) == 2
    assert all(s.outcome == "transport" and s.status_code is None for s in samples)


async def test_run_resumes_an_existing_file(tmp_path: Path, rsa_pem: Path) -> None:
    out = tmp_path / "samples.jsonl"
    seeded = ReadSample(
        sequence=0,
        requested_at=datetime(2026, 8, 11, 18, 0, tzinfo=timezone.utc),
        elapsed_s=0.2,
        ticker=TICKERS[0],
        outcome="ok",
        status_code=None,
        api_host="api.elections.kalshi.com",
        endpoint=f"GET /trade-api/v2/markets/{TICKERS[0]}/orderbook",
        source_host="kalshi-ws",
    )
    append_sample(out, seeded)

    assert await _run(out, rsa_pem, httpx.Response(200)) == 0

    samples = load_samples(out)
    assert [s.sequence for s in samples] == [0, 1]
    assert samples[0] == seeded


async def test_run_exits_non_zero_when_first_ticker_fetch_fails(
    tmp_path: Path, rsa_pem: Path
) -> None:
    out = tmp_path / "samples.jsonl"

    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    args = build_parser().parse_args(["--out", str(out), "--target", "2"])
    settings = _settings(rsa_pem)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url=PROD_BASE) as http:
        client = KalshiReadClient(settings, http_client=http)
        await client.aopen()
        try:
            assert await run(args, client, settings) == 1
        finally:
            await client.aclose()
    assert not out.exists()


async def test_run_exits_non_zero_when_no_markets_are_open(tmp_path: Path, rsa_pem: Path) -> None:
    out = tmp_path / "samples.jsonl"

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"markets": [], "cursor": ""})

    args = build_parser().parse_args(["--out", str(out), "--target", "2"])
    settings = _settings(rsa_pem)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url=PROD_BASE) as http:
        client = KalshiReadClient(settings, http_client=http)
        await client.aopen()
        try:
            assert await run(args, client, settings) == 1
        finally:
            await client.aclose()
    assert not out.exists()
