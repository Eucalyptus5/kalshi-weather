# Kalshi Weather

Paper trading bot for Kalshi weather markets. Reads live Kalshi orderbooks against a probabilistic forecast (Open-Meteo ensemble in MVP, NBM and HRRR later) and persists intended trades plus simulated PnL for offline calibration. Live trading is not wired up.

## Setup

```
uv sync
cp .env.example .env
```

Edit `.env` to point at your demo Kalshi key and PEM file before running anything under `scripts/` or `bot/`.

## Demo credentials

The smoke script and bot both run against `demo-api.kalshi.co`. To get keys:

1. Sign up at <https://demo.kalshi.co>.
2. From the demo dashboard, generate an API key pair. Save the key ID into `.env` as `KALSHI_DEMO_KEY_ID`.
3. Save the private key PEM into `secrets/kalshi_demo.pem` (this directory is gitignored, create it with `mkdir -p secrets`).
4. Verify with `uv run python -m scripts.smoke_demo`. It should print a handful of `KXHIGHDEN` markets with their strike ladders.

## Tests

```
uv run pytest -q
```
