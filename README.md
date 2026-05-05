# Kalshi Weather

Paper trading bot for Kalshi weather markets. Reads live Kalshi orderbooks against a probabilistic forecast (Open-Meteo ensemble in MVP, NBM and HRRR later) and persists intended trades plus simulated PnL for offline calibration. Live trading is not wired up.

## Setup

```
uv sync
cp .env.example .env
```

Edit `.env` to point at your demo Kalshi key and PEM file before running anything under `scripts/` or `bot/`.

## Tests

```
uv run pytest -q
```
