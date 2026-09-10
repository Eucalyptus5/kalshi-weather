# Kalshi Weather

A research program on Kalshi's daily high and low temperature markets. It records the live order book, replays it offline, and tests specific claims about where a tradable edge might exist. Five hypotheses were pre-registered and all five were resolved. None of them became a strategy worth trading, and the section below says why.

Paper and demo only. No live orders and no real capital at any point.

## What it found

Each family got a written pre-registration, a fixed share of a 0.05 alpha budget, and a discovery and holdout split frozen before any tape was read.

**Maker-side economics: PASS, on a fee regime rather than a law.** 0.20 cents per contract over 416 discovery market-days and 0.28 over 152 holdout, both significant. The result rests entirely on weather paying no maker fee. Priced at the published 0.0175 rate, the same fills read negative on both splits. That rate is a live venue configuration readable off a series field, not something the venue promises, and the published fee schedule was already stale against it.

**Settlement source: PASS, on an instant nobody could trade.** 33.5 cents per contract over 39 discovery city event-days and 39.7 over 25 holdout. The entry instant is not identifiable while it is happening, because the official daily extreme does not exist until after the observation window closes. The effect is well defined after the fact and cannot be acted on as measured.

**Forecast source: CLOSED at full power.** Negative on both splits, -1.18 cents over 220 discovery event-days and -1.85 over 122 holdout, on a fifteen month sample. This is the one result that is not seasonal.

**Low-temperature staleness: UNDERPOWERED, no extension spent.** 5 discovery station-days carrying a mid-day clean lock against a floor of 30. An honest projection of a funded extension still missed the floor by a factor of three, so it was stopped instead of extended.

**Cross-series consistency: CLOSED on geometry, before any tape was read.** Across all 60 paired high and low city-days in the window, the gap between the two ladders' tail strikes ran 3F at its narrowest and 24F at its widest, and no pair ever expressed a jointly impossible statement.

An earlier latency study closed the speed lane on its own. Median lag from event to observed book update was 20 seconds against a REST polling floor of 79 seconds, so there was no execution advantage to build on.

Two passes are not two builds. Both carry a condition that decides whether they imply anything, and neither condition is settled by recording more tape.

## How it works

A WebSocket recorder runs continuously on a GCP VM, capturing order book deltas and trades across 20 stations into SQLite, with a health watchdog and alerting. The tape has run unbroken since 2026-07-17 through a full host migration across cloud projects.

Studies never read the live venue. A replay engine reconstructs the book at any instant from the recorded deltas and serves it to the strategy code, so a run is reproducible from the tape alone. Every run writes a provenance manifest first, recording the git commit, a dirty flag, and a hash of every input file, then aborts if the pre-registration file is missing. Bootstrap seeds are recorded and never reused across runs that share evidence.

Forecasts come from Open-Meteo ensembles and the statistical guidance product, with GRIB decoding for the model backtests. Probabilities are calibrated against realised outcomes rather than trusted raw. All prices and fees are `Decimal`, never float.

Blind windows are counted rather than assumed away. Recorder resubscribes drop roughly ten seconds of book each, and candidate fills landing inside a gap, a quiet band, or a resubscribe are excluded and reported as a share of the funnel.

## Layout

```
bot/forecast/       ensemble retrieval, calibration, CDF math
bot/markets/        ticker parsing, ladder geometry, strike math
bot/strategy/       entry rules, edge pricing, fill models
bot/risk/           sizing and exposure limits
bot/execution/      order placement against demo, paper simulator
bot/replay/         book reconstruction from recorded deltas
bot/observations/   station readings and settlement values
bot/lag/            latency measurement
bot/observability/  run manifests, logging, metrics
bot/backtest/       scored runs over frozen samples
bot/storage/        SQLite models and Alembic migrations
bot/validation/     input and response validation
scripts/            study runners, report generators, recorder, watchdog
```

Around 35,000 lines across 167 modules, 160 test files, and a suite of roughly 4,600 tests that must be green before any commit.

## Setup

```
uv sync
cp .env.example .env
```

Edit `.env` to point at your demo Kalshi key and PEM file before running anything under `scripts/` or `bot/`.

GRIB decoding needs the eccodes C library, which is a system package: `conda install -c conda-forge eccodes`, `apt install libeccodes-dev`, or `brew install eccodes`. The Python bindings come in with `uv sync`, and tests that decode GRIB skip when the library is missing.

## Demo credentials

The smoke script and the bot both run against `demo-api.kalshi.co`.

1. Sign up at <https://demo.kalshi.co>.
2. Generate an API key pair from the demo dashboard. Save the key ID into `.env` as `KALSHI_DEMO_KEY_ID`.
3. Save the private key PEM into `secrets/kalshi_demo.pem`, which is gitignored. Create the directory with `mkdir -p secrets`.
4. Verify with `uv run python -m scripts.smoke_demo`. It prints a handful of `KXHIGHDEN` markets with their strike ladders.

## Tests

```
uv run pytest -q
```
