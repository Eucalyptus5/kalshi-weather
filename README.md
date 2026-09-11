# Kalshi Weather

Research program on Kalshi's daily high and low temperature markets. It records the live order book, replays it offline, and tests specific claims about where a tradable edge might exist.

Five hypotheses were pre-registered. All five were resolved. None became a strategy worth trading.

> Paper and demo only. No live orders and no real capital at any point.

## Results

| Family | Verdict | Measured | Why it does not trade |
| --- | --- | --- | --- |
| Maker-side economics | **PASS** | +0.20c / +0.28c | Holds only while weather pays no maker fee |
| Settlement source | **PASS** | +33.5c / +39.7c | Entry instant is not identifiable in real time |
| Forecast source | **CLOSED** | -1.18c / -1.85c | Negative at full power over 15 months |
| Low-temp staleness | **UNDERPOWERED** | 5 of 30 station-days | A funded extension still projects 3x short |
| Cross-series consistency | **CLOSED** | 0 of 60 pairs | Killed on ladder geometry before any tape |
| Execution speed | **CLOSED** | 20s lag vs 79s floor | No latency advantage to build on |

Cent figures are per contract, discovery split first and holdout second. Each family got a written pre-registration, a fixed share of a 0.05 alpha budget, and a discovery and holdout split frozen before any tape was read.

### The catch on the two passes

**Maker-side economics.** Significant on 416 discovery market-days and 152 holdout. Priced at the published 0.0175 maker rate, the same fills read negative on both splits. The venue currently charges no maker fee on weather, which is a live configuration readable off a series field rather than a promise, and the published fee schedule was already stale against it.

**Settlement source.** Significant on 39 discovery city event-days and 25 holdout. The official daily extreme does not exist until after the observation window closes, so nobody standing at the entry instant can know it is happening. The effect is well defined after the fact and cannot be acted on as measured.

Neither condition is settled by recording more tape, so neither pass is a build.

## How it works

**Recording.** A WebSocket recorder runs continuously on a GCP VM, capturing book deltas and trades across 20 stations into SQLite, with a health watchdog and alerting. The tape has run unbroken since 2026-07-17, through a full host migration across cloud projects.

**Replay.** Studies never read the live venue. A replay engine reconstructs the book at any instant from recorded deltas and serves it to the strategy code, so every run is reproducible from the tape alone.

**Provenance.** Each run writes a manifest before any statistic touches the data, recording the git commit, a dirty flag, and a hash of every input. A missing pre-registration file aborts the run. Bootstrap seeds are recorded and never reused across runs that share evidence.

**Forecasts.** Open-Meteo ensembles and the statistical guidance product, with GRIB decoding for model backtests. Probabilities are calibrated against realised outcomes rather than trusted raw. All prices and fees are `Decimal`, never float.

**Blind windows.** Recorder resubscribes drop roughly ten seconds of book each. Candidate fills landing in a gap, a quiet band, or a resubscribe are excluded and reported as a share of the funnel rather than assumed away.

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

Roughly 35,000 lines across 167 modules, with 160 test files and about 4,600 tests that must be green before any commit.

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
