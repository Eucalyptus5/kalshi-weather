import json
import re
import subprocess
from dataclasses import fields, replace
from datetime import date, datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from bot.execution import fees
from bot.lag.fee_floor import (
    MAKER_RATE_SOURCE,
    PUBLISHED_MAKER_RATE,
    economic_bar_cents_per_contract,
    fee_source,
)
from bot.lag.fee_regime import PLAIN_REGIME, FeeRegimeCheck
from bot.lag.r0_universe import (
    Coverage,
    R0Universe,
    freeze_digest,
    freeze_universe,
    universe_payload,
)
from bot.lag.read_rtt import (
    FloorSource,
    InadequateSamples,
    LatencyFloor,
    ReadSample,
    derive_latency_floor,
)
from bot.lag.run_manifest import (
    BOOTSTRAP_RESAMPLES,
    EXEMPTIBLE_FIELDS,
    MANIFEST_NAME,
    SUPPLIED_FIELDS,
    Exemption,
    ExemptionRefused,
    GitState,
    Manifest,
    ManifestIncomplete,
    RunInputs,
    SettlementRecord,
    build_manifest,
    fee_payload,
    git_state,
    manifest_payload,
    resolve_latency_floor,
    write_manifest,
)
from bot.lag.settlement_source import (
    BOUNDARY_SOURCE,
    BoundarySplit,
    SeriesSettlementSource,
    SettlementProvenance,
)
from bot.replay.analysis_stations import HIGH


UTC = timezone.utc
RUN_ID = "2026-08-12-q1"
TICKER = "KXHIGHDEN-26AUG11-T95"
THRESHOLD = Decimal("0.5")
CHARGED_SOURCE = "statistic_charges_its_own_fee"
PASSING = ("KXHIGHCHI", "KXHIGHDEN", "KXHIGHMIA", "KXHIGHNY")
ACCRUAL_START = datetime(2026, 7, 18, tzinfo=UTC)
ACCRUAL_END = datetime(2026, 8, 1, tzinfo=UTC)
ROW_COUNTS = {"ws_book_deltas": 482_113, "ws_trades": 51_204}
SEED = 20260812
FOUR_DP = Decimal("0.0001")
BAR_SIZE = Decimal("26")
BAR_PRICE = Decimal("0.50")
BAR_PRICE_SOURCE = "preregistration"
ZERO_RATE = Decimal("0")
ZERO_RATE_SOURCE = "kalshi_series_metadata"
NO_TAPE = "the statistic reads no ws tape"
NO_READS = "the statistic places no read against the api"
OBSERVED_AT = datetime(2026, 8, 19, 17, 30, tzinfo=UTC)
FROZEN_DIGEST = "2e73ed22bb8e6b90698a84b204da0d54ff332cbb844da78eeaac64ad3a481d41"
EXEMPT_DIGEST = "1a17447f6696469fdbd52072e80e7c5c4ed5c3404c6c9ad8d5a504fe54ad8639"
AMBIENT_PRECISIONS = (20, 28, 50)
FEE_TYPE_KEYS = ("fee_type_check", "fee_type_observed_at", "fee_type_sha256")
SETTLEMENT_KEYS = (
    "settlement_source",
    "settlement_source_url",
    "last_updated_ts",
    "days_before_boundary",
    "days_on_or_after_boundary",
    "boundary_date",
    "boundary_source",
)
MOVED_AT = datetime(2026, 8, 14, 17, 48, 38, tzinfo=UTC)
BOUNDARY_DATE = date(2026, 8, 14)
WEATHER_COMPANY = "The Weather Company"
WEATHER_COMPANY_URL = "https://weather.com/kalshi"
SETTLEMENT_ROOTS = ("KXHIGHNY", "KXHIGHDEN")
FLOOR_KEYS = (
    "latency_floor_source",
    "latency_floor_s",
    "t_persist_s",
    "latency_floor_samples",
)

FIELDS = {
    "run_id",
    "preregistration_path",
    "preregistration_sha256",
    "accrual_start",
    "accrual_end",
    "row_counts",
    "git_head",
    "git_dirty",
    "r0_fraction_invalid_max",
    "r0_universe_sha256",
    "fee_threshold_source",
    "fee_module",
    "fee_module_quantum",
    "fee_module_corrected",
    "fee_maker_rate",
    "fee_maker_rate_source",
    "latency_floor_source",
    "latency_floor_s",
    "t_persist_s",
    "latency_floor_samples",
    "economic_bar_size",
    "economic_bar_price",
    "economic_bar_price_source",
    "economic_bar_cents_per_contract",
    "bootstrap_resamples",
    "bootstrap_seed",
}

# The f1-20260820 run computed this digest on the recorder host before the bar context was pinned.
# Held as a literal so a change to the canonical serialisation freeze_digest performs has to move
# this test rather than silently invalidate a finished run's stored digest. Precision was never in
# play here: that run charged its own fee, so it recorded economic_bar_size "0", took the early
# return and never divided.
F1_DIGEST = "26221572c18f4fd9687ebae8cc3a728eea63b02818a8cc4d0fbf2a2b84b27515"
F1_MANIFEST_PAYLOAD = {
    "accrual_end": "2026-08-16T08:00:00+00:00",
    "accrual_start": "2026-08-02T05:00:00+00:00",
    "bootstrap_resamples": 10000,
    "bootstrap_seed": 20260820,
    "cohort": "high",
    "economic_bar_cents_per_contract": "0",
    "economic_bar_price": "0",
    "economic_bar_price_source": "statistic_charges_its_own_fee",
    "economic_bar_size": "0",
    "fee_maker_rate": "0",
    "fee_maker_rate_source": "series_api_fee_type_quadratic_2026-08-19",
    "fee_module": "bot.execution.fees.taker_fee",
    "fee_module_corrected": True,
    "fee_module_quantum": "0.01",
    "fee_threshold_source": "published_formula",
    "fee_type_check": "plain_quadratic",
    "fee_type_observed_at": "2026-08-20T00:20:16+00:00",
    "fee_type_sha256": "130b0e9dfd69109fca1b044b1b218e42b5a1ddafc9c92f1864d42bc298792ff0",
    "git_dirty": False,
    "git_head": "b20ee297de43e07406800c8be773ea570d1384fb",
    "latency_floor_s": 0.14643257297575474,
    "latency_floor_samples": 200,
    "latency_floor_source": "RTT_read",
    "preregistration_path": "improvements/active/f1_preregistration.md",
    "preregistration_sha256": "b5159a022dea06b191cfe0e44525acc11c1c4d9db841557d6bab39ecf27c9268",
    "r0_fraction_invalid_max": "0.5",
    "r0_universe_sha256": "19840051b99cb602612814105503fdcd2c3b86058eb2087f742215bd6692918c",
    "row_counts": {
        "event_days": 1320,
        "exclusions": 361,
        "ladder": 113284469,
        "touch": 0,
        "trades": 1315459,
    },
    "run_id": "f1-20260820",
    "t_persist_s": 10.0,
}


def _sample(sequence: int, hour: int, elapsed_s: float = 0.24) -> ReadSample:
    return ReadSample(
        sequence=sequence,
        requested_at=datetime(2026, 8, 12, hour, sequence % 60, tzinfo=UTC),
        elapsed_s=elapsed_s,
        ticker=TICKER,
        outcome="ok",
        status_code=200,
        api_host="api.elections.kalshi.com",
        endpoint=f"GET /trade-api/v2/markets/{TICKER}/orderbook",
        source_host="kalshi-ws",
    )


def _adequate_samples() -> list[ReadSample]:
    return [_sample(i, i % 24) for i in range(240)]


def _todays_samples() -> list[ReadSample]:
    return [_sample(i, i % 7) for i in range(58)]


def _universe(threshold: Decimal = THRESHOLD) -> R0Universe:
    return freeze_universe(
        fraction_invalid_max=threshold,
        passing=PASSING,
        coverage=Coverage(cities=PASSING, ladder_widths=(6,), in_scope_city_days=56),
    )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def _seeded_repo(root: Path, seed: str) -> Path:
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "core.hooksPath", str(root / ".git" / "hooks"))
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "config", "user.name", "tape")
    _git(root, "config", "user.email", "tape@example.invalid")
    (root / "seed.txt").write_text(f"{seed}\n")
    _git(root, "add", "seed.txt")
    _git(root, "commit", "-q", "-m", seed)
    return root


def _head(repo: Path) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _seeded_repo(tmp_path / "tree", "seed")


@pytest.fixture
def preregistration(tmp_path: Path) -> Path:
    path = tmp_path / "plan.md"
    path.write_text("alpha = 0.0125\nB = 10000\nD_eval = 14\n")
    return path


@pytest.fixture
def complete(preregistration: Path, repo: Path) -> RunInputs:
    return RunInputs(
        run_id=RUN_ID,
        preregistration=preregistration,
        repo=repo,
        accrual_start=ACCRUAL_START,
        accrual_end=ACCRUAL_END,
        row_counts=dict(ROW_COUNTS),
        universe=_universe(),
        fee=fee_source(maker_rate=PUBLISHED_MAKER_RATE, maker_rate_source=MAKER_RATE_SOURCE),
        floor=derive_latency_floor(_adequate_samples(), FloorSource.SIGNED_READ),
        economic_bar_size=BAR_SIZE,
        economic_bar_price=BAR_PRICE,
        economic_bar_price_source=BAR_PRICE_SOURCE,
        bootstrap_seed=SEED,
    )


def test_a_complete_run_records_every_field_the_section_names(
    tmp_path: Path, complete: RunInputs
) -> None:
    root = tmp_path / "tape_studies"

    digest = write_manifest(root, complete)

    payload = json.loads((root / RUN_ID / MANIFEST_NAME).read_text())
    assert payload.pop("sha256") == digest
    assert freeze_digest(payload) == digest
    assert set(payload) == FIELDS
    assert payload["run_id"] == RUN_ID
    assert payload["preregistration_path"] == str(complete.preregistration)
    assert re.fullmatch(r"[0-9a-f]{64}", payload["preregistration_sha256"])
    assert payload["accrual_start"] == ACCRUAL_START.isoformat()
    assert payload["accrual_end"] == ACCRUAL_END.isoformat()
    assert payload["row_counts"] == ROW_COUNTS
    assert re.fullmatch(r"[0-9a-f]{40}", payload["git_head"])
    assert payload["git_dirty"] is False
    assert payload["r0_fraction_invalid_max"] == str(THRESHOLD)
    assert payload["r0_universe_sha256"] == freeze_digest(universe_payload(_universe()))
    assert payload["fee_threshold_source"] == "published_formula"
    assert payload["fee_module"] == "bot.execution.fees.taker_fee"
    assert payload["fee_module_quantum"] == str(fees.FEE_QUANTUM)
    assert payload["fee_module_corrected"] is True
    assert payload["fee_maker_rate"] == str(PUBLISHED_MAKER_RATE)
    assert payload["fee_maker_rate_source"] == MAKER_RATE_SOURCE
    assert payload["latency_floor_source"] == "RTT_read"
    assert payload["latency_floor_s"] == pytest.approx(0.24)
    assert payload["t_persist_s"] == pytest.approx(10.0)
    assert payload["latency_floor_samples"] == 240
    assert payload["economic_bar_size"] == str(BAR_SIZE)
    assert payload["economic_bar_price"] == str(BAR_PRICE)
    assert payload["economic_bar_price_source"] == BAR_PRICE_SOURCE
    assert Decimal(payload["economic_bar_cents_per_contract"]).quantize(FOUR_DP) == Decimal(
        "2.7692"
    )
    assert payload["bootstrap_resamples"] == 10_000
    assert payload["bootstrap_seed"] == SEED


@pytest.mark.parametrize(
    ("attribute", "field"),
    [
        ("accrual_start", "accrual_start"),
        ("accrual_end", "accrual_end"),
        ("row_counts", "row_counts"),
        ("universe", "r0_fraction_invalid_max"),
        ("fee", "fee_source"),
        ("floor", "latency_floor"),
        ("economic_bar_size", "economic_bar_size"),
        ("economic_bar_price", "economic_bar_price"),
        ("economic_bar_price_source", "economic_bar_price_source"),
        ("bootstrap_seed", "bootstrap_seed"),
    ],
)
def test_a_run_missing_one_field_aborts_naming_that_field(
    tmp_path: Path, complete: RunInputs, attribute: str, field: str
) -> None:
    root = tmp_path / "tape_studies"

    with pytest.raises(ManifestIncomplete) as excinfo:
        write_manifest(root, replace(complete, **{attribute: None}))

    assert excinfo.value.fields == (field,)
    assert field in str(excinfo.value)
    assert not root.exists()


def test_an_abort_names_every_field_the_run_is_short_of(
    tmp_path: Path, complete: RunInputs
) -> None:
    with pytest.raises(ManifestIncomplete) as excinfo:
        write_manifest(
            tmp_path / "tape_studies", replace(complete, fee=None, floor=None, bootstrap_seed=None)
        )

    assert excinfo.value.fields == ("fee_source", "latency_floor", "bootstrap_seed")


def test_a_run_without_an_id_has_no_run_directory_to_write_to(
    tmp_path: Path, complete: RunInputs
) -> None:
    root = tmp_path / "tape_studies"

    with pytest.raises(ManifestIncomplete) as excinfo:
        write_manifest(root, replace(complete, run_id=""))

    assert excinfo.value.fields == ("run_id",)
    assert not root.exists()


@pytest.mark.parametrize("run_id", ["..", ".", "2026/q1", "2026-08-12-q1/"])
def test_a_run_id_that_is_not_one_directory_name_aborts(
    tmp_path: Path, complete: RunInputs, run_id: str
) -> None:
    root = tmp_path / "tape_studies"

    with pytest.raises(ManifestIncomplete) as excinfo:
        write_manifest(root, replace(complete, run_id=run_id))

    assert excinfo.value.fields == ("run_id",)
    assert not root.exists()
    assert not (tmp_path / MANIFEST_NAME).exists()


def test_an_absolute_run_id_would_write_outside_the_run_root_and_aborts(
    tmp_path: Path, complete: RunInputs
) -> None:
    root = tmp_path / "tape_studies"
    escaped = tmp_path / "escaped"

    with pytest.raises(ManifestIncomplete) as excinfo:
        write_manifest(root, replace(complete, run_id=str(escaped)))

    assert excinfo.value.fields == ("run_id",)
    assert not escaped.exists()
    assert not root.exists()


def test_an_empty_row_count_map_records_nothing_and_aborts(
    tmp_path: Path, complete: RunInputs
) -> None:
    with pytest.raises(ManifestIncomplete) as excinfo:
        write_manifest(tmp_path / "tape_studies", replace(complete, row_counts={}))

    assert excinfo.value.fields == ("row_counts",)


def test_a_bootstrap_seed_of_zero_is_a_seed_not_a_missing_field(complete: RunInputs) -> None:
    payload = manifest_payload(build_manifest(replace(complete, bootstrap_seed=0)))

    assert payload["bootstrap_seed"] == 0
    assert payload["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES


def test_the_stated_bar_and_the_bar_it_derives_round_trip_through_the_written_manifest(
    tmp_path: Path, complete: RunInputs
) -> None:
    root = tmp_path / "tape_studies"
    manifest = build_manifest(complete)

    write_manifest(root, complete)

    payload = json.loads((root / RUN_ID / MANIFEST_NAME).read_text())
    assert manifest.economic_bar_size == BAR_SIZE
    assert manifest.economic_bar_price == BAR_PRICE
    assert manifest.economic_bar_price_source == BAR_PRICE_SOURCE
    assert manifest.economic_bar_cents_per_contract.quantize(FOUR_DP) == Decimal("2.7692")
    assert isinstance(payload["economic_bar_size"], str)
    assert isinstance(payload["economic_bar_price"], str)
    assert isinstance(payload["economic_bar_cents_per_contract"], str)
    assert Decimal(payload["economic_bar_size"]).as_tuple() == BAR_SIZE.as_tuple()
    assert Decimal(payload["economic_bar_price"]).as_tuple() == BAR_PRICE.as_tuple()
    assert payload["economic_bar_price_source"] == BAR_PRICE_SOURCE
    assert (
        Decimal(payload["economic_bar_cents_per_contract"])
        == manifest.economic_bar_cents_per_contract
    )


@pytest.mark.parametrize(
    ("price", "bar"),
    [
        (Decimal("0.50"), Decimal("2.7692")),
        (Decimal("0.05"), Decimal("1.3462")),
        (Decimal("0.95"), Decimal("1.3462")),
    ],
)
def test_the_manifest_carries_the_bar_the_stated_price_derives(
    tmp_path: Path, complete: RunInputs, price: Decimal, bar: Decimal
) -> None:
    root = tmp_path / "tape_studies"

    write_manifest(root, replace(complete, economic_bar_price=price))

    payload = json.loads((root / RUN_ID / MANIFEST_NAME).read_text())
    assert payload["economic_bar_size"] == "26"
    assert payload["economic_bar_price"] == str(price)
    assert Decimal(payload["economic_bar_cents_per_contract"]).quantize(FOUR_DP) == bar


def test_both_tails_of_the_price_grid_derive_the_same_bar(complete: RunInputs) -> None:
    low = manifest_payload(build_manifest(replace(complete, economic_bar_price=Decimal("0.05"))))
    high = manifest_payload(build_manifest(replace(complete, economic_bar_price=Decimal("0.95"))))
    middle = manifest_payload(build_manifest(complete))

    assert low["economic_bar_cents_per_contract"] == high["economic_bar_cents_per_contract"]
    assert Decimal(low["economic_bar_cents_per_contract"]) < Decimal(
        middle["economic_bar_cents_per_contract"]
    )


def test_a_size_stated_without_a_price_aborts_and_writes_nothing(
    tmp_path: Path, complete: RunInputs
) -> None:
    root = tmp_path / "tape_studies"

    with pytest.raises(ManifestIncomplete) as excinfo:
        write_manifest(root, replace(complete, economic_bar_price=None))

    assert "economic_bar_price" in excinfo.value.fields
    assert "economic_bar_price" in str(excinfo.value)
    assert not (root / RUN_ID / MANIFEST_NAME).exists()
    assert not root.exists()


def test_a_stated_zero_bar_is_a_bar_not_a_missing_field(
    tmp_path: Path, complete: RunInputs
) -> None:
    root = tmp_path / "tape_studies"
    charged = replace(
        complete,
        economic_bar_size=Decimal("0"),
        economic_bar_price=Decimal("0"),
        economic_bar_price_source=CHARGED_SOURCE,
    )

    write_manifest(root, charged)

    payload = json.loads((root / RUN_ID / MANIFEST_NAME).read_text())
    assert "economic_bar_cents_per_contract" in payload
    assert Decimal(payload["economic_bar_size"]) == 0
    assert Decimal(payload["economic_bar_price"]) == 0
    assert Decimal(payload["economic_bar_cents_per_contract"]) == 0
    assert payload["economic_bar_price_source"] == CHARGED_SOURCE


def test_a_preregistration_file_that_is_not_on_disk_aborts(
    tmp_path: Path, complete: RunInputs
) -> None:
    absent = tmp_path / "gone.md"
    root = tmp_path / "tape_studies"

    with pytest.raises(ManifestIncomplete) as excinfo:
        write_manifest(root, replace(complete, preregistration=absent))

    assert excinfo.value.fields == ("preregistration_sha256",)
    assert str(absent) in str(excinfo.value)
    assert not root.exists()


def test_a_tree_that_is_not_a_repository_aborts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, complete: RunInputs
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    loose = tmp_path / "loose"
    loose.mkdir()
    root = tmp_path / "tape_studies"

    with pytest.raises(ManifestIncomplete) as excinfo:
        write_manifest(root, replace(complete, repo=loose))

    assert excinfo.value.fields == ("git_head",)
    assert "not a git repository" in str(excinfo.value)
    assert not root.exists()


def test_a_subdirectory_of_a_repository_is_not_the_tree_that_produced_the_run(
    tmp_path: Path, complete: RunInputs
) -> None:
    nested = complete.repo / "lag"
    nested.mkdir()
    root = tmp_path / "tape_studies"
    enclosing = _head(complete.repo)

    with pytest.raises(ManifestIncomplete) as excinfo:
        write_manifest(root, replace(complete, repo=nested))

    assert _head(nested) == enclosing
    assert excinfo.value.fields == ("git_head",)
    assert enclosing not in str(excinfo.value)
    assert not root.exists()


def test_a_path_that_resolves_to_the_root_is_the_root(tmp_path: Path, repo: Path) -> None:
    (repo / "lag").mkdir()
    link = tmp_path / "link"
    link.symlink_to(repo)

    assert git_state(link) == git_state(repo)
    assert git_state(repo / "lag" / "..") == git_state(repo)


def test_a_git_dir_in_the_environment_does_not_supply_the_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo: Path
) -> None:
    other = _seeded_repo(tmp_path / "other", "other")
    mine, theirs = _head(repo), _head(other)
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))

    state = git_state(repo)

    assert state.head == mine
    assert state.head != theirs


def test_a_borrowed_index_does_not_dirty_the_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo: Path
) -> None:
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "borrowed.index"))

    assert git_state(repo).dirty is False


def test_todays_read_rtt_samples_refuse_to_supply_a_floor() -> None:
    samples = _todays_samples()

    with pytest.raises(InadequateSamples):
        derive_latency_floor(samples, FloorSource.SIGNED_READ)
    with pytest.raises(ManifestIncomplete) as excinfo:
        resolve_latency_floor(samples, FloorSource.SIGNED_READ)

    assert excinfo.value.fields == ("latency_floor",)
    assert "usable samples 58 short of 200" in str(excinfo.value)


def test_a_run_whose_floor_is_unavailable_writes_no_manifest(
    tmp_path: Path, complete: RunInputs
) -> None:
    root = tmp_path / "tape_studies"

    with pytest.raises(ManifestIncomplete):
        resolve_latency_floor(_todays_samples(), FloorSource.SIGNED_READ)
    with pytest.raises(ManifestIncomplete) as excinfo:
        write_manifest(root, replace(complete, floor=None))

    assert excinfo.value.fields == ("latency_floor",)
    assert not root.exists()


def test_an_adequate_sample_set_resolves_to_the_same_floor_the_module_derives() -> None:
    samples = _adequate_samples()

    assert resolve_latency_floor(samples, FloorSource.SIGNED_READ) == derive_latency_floor(
        samples, FloorSource.SIGNED_READ
    )


@pytest.mark.parametrize(
    ("source", "recorded"), [(FloorSource.DEMO_ORDER, "L"), (FloorSource.SIGNED_READ, "RTT_read")]
)
def test_the_manifest_names_which_source_supplied_the_floor(
    complete: RunInputs, source: FloorSource, recorded: str
) -> None:
    floor = derive_latency_floor(_adequate_samples(), source)

    payload = manifest_payload(build_manifest(replace(complete, floor=floor)))

    assert payload["latency_floor_source"] == recorded
    assert payload["latency_floor_s"] == pytest.approx(floor.floor_s)
    assert payload["t_persist_s"] == pytest.approx(floor.t_persist_s)


def test_the_r0_threshold_round_trips_as_an_exact_decimal(
    tmp_path: Path, complete: RunInputs
) -> None:
    root = tmp_path / "tape_studies"

    write_manifest(root, complete)

    payload = json.loads((root / RUN_ID / MANIFEST_NAME).read_text())
    assert isinstance(payload["r0_fraction_invalid_max"], str)
    assert Decimal(payload["r0_fraction_invalid_max"]).as_tuple() == THRESHOLD.as_tuple()
    assert isinstance(payload["fee_module_quantum"], str)
    assert Decimal(payload["fee_module_quantum"]) == fees.FEE_QUANTUM


def test_a_loosened_r0_threshold_moves_the_recorded_digest(complete: RunInputs) -> None:
    frozen = manifest_payload(build_manifest(complete))

    loosened = manifest_payload(
        build_manifest(replace(complete, universe=_universe(Decimal("0.6"))))
    )

    assert frozen["r0_universe_sha256"] != loosened["r0_universe_sha256"]
    assert loosened["r0_fraction_invalid_max"] == "0.6"


def test_a_corrected_fee_module_is_recorded_as_corrected(complete: RunInputs) -> None:
    source = fee_source(maker_rate=PUBLISHED_MAKER_RATE, maker_rate_source=MAKER_RATE_SOURCE)

    payload = manifest_payload(build_manifest(replace(complete, fee=source)))

    assert payload["fee_module_quantum"] == "0.01"
    assert payload["fee_module_corrected"] is True


def test_an_uncorrected_fee_module_is_recorded_as_uncorrected(
    monkeypatch: pytest.MonkeyPatch, complete: RunInputs
) -> None:
    monkeypatch.setattr(fees, "FEE_QUANTUM", Decimal("0.000001"))
    source = fee_source(maker_rate=PUBLISHED_MAKER_RATE, maker_rate_source=MAKER_RATE_SOURCE)

    payload = manifest_payload(build_manifest(replace(complete, fee=source)))

    assert payload["fee_module_quantum"] == "0.000001"
    assert payload["fee_module_corrected"] is False


def test_the_written_manifest_round_trips_both_maker_fee_keys(
    tmp_path: Path, complete: RunInputs
) -> None:
    root = tmp_path / "tape_studies"

    write_manifest(root, complete)

    payload = json.loads((root / RUN_ID / MANIFEST_NAME).read_text())
    assert payload["fee_maker_rate"] == str(PUBLISHED_MAKER_RATE)
    assert payload["fee_maker_rate_source"] == MAKER_RATE_SOURCE
    assert payload["fee_threshold_source"] == "published_formula"
    assert payload["fee_module"] == "bot.execution.fees.taker_fee"
    assert payload["fee_module_quantum"] == str(fees.FEE_QUANTUM)
    assert payload["fee_module_corrected"] is True


def test_the_regime_a_run_states_is_the_regime_its_manifest_digests(complete: RunInputs) -> None:
    zero = fee_source(maker_rate=ZERO_RATE, maker_rate_source=ZERO_RATE_SOURCE)

    stated = manifest_payload(build_manifest(replace(complete, fee=zero)))
    published = manifest_payload(build_manifest(complete))

    assert stated["fee_maker_rate"] == "0"
    assert stated["fee_maker_rate_source"] == ZERO_RATE_SOURCE
    assert published["fee_maker_rate"] == "0.0175"
    assert published["fee_maker_rate_source"] == MAKER_RATE_SOURCE
    assert freeze_digest(stated) != freeze_digest(published)


def test_the_fee_key_derivation_rule_reproduces_the_published_keys_byte_for_byte(
    complete: RunInputs,
) -> None:
    payload = fee_payload(complete.fee)

    assert set(payload) == {
        "fee_threshold_source",
        "fee_module",
        "fee_module_quantum",
        "fee_module_corrected",
        "fee_maker_rate",
        "fee_maker_rate_source",
    }
    assert payload["fee_threshold_source"] == "published_formula"
    assert payload["fee_module"] == "bot.execution.fees.taker_fee"
    assert payload["fee_module_quantum"] == str(fees.FEE_QUANTUM)
    assert payload["fee_module_corrected"] is True
    assert payload["fee_maker_rate"] == str(PUBLISHED_MAKER_RATE)
    assert payload["fee_maker_rate_source"] == MAKER_RATE_SOURCE


def test_the_caller_cannot_supply_the_head_or_the_dirty_flag() -> None:
    supplied = {field.name for field in fields(RunInputs)}

    assert "repo" in supplied
    assert "git_head" not in supplied
    assert "git_dirty" not in supplied


def test_the_caller_states_the_bar_inputs_but_not_the_bar() -> None:
    supplied = {field.name for field in fields(RunInputs)}

    assert {"economic_bar_size", "economic_bar_price", "economic_bar_price_source"} <= supplied
    assert "economic_bar_cents_per_contract" not in supplied
    assert "economic_bar_cents_per_contract" in {field.name for field in fields(Manifest)}


def test_the_dirty_flag_follows_the_working_tree(tmp_path: Path, complete: RunInputs) -> None:
    root = tmp_path / "tape_studies"
    write_manifest(root, complete)
    clean = json.loads((root / RUN_ID / MANIFEST_NAME).read_text())

    (complete.repo / "seed.txt").write_text("edited\n")
    dirty = manifest_payload(build_manifest(complete))

    assert clean["git_dirty"] is False
    assert dirty["git_dirty"] is True
    assert dirty["git_head"] == clean["git_head"]


def test_an_untracked_file_leaves_the_tree_dirty(repo: Path) -> None:
    assert git_state(repo).dirty is False

    (repo / "scratch.py").write_text("x = 1\n")

    assert git_state(repo).dirty is True


def test_the_hash_moves_when_the_file_changes_between_reads(
    preregistration: Path, complete: RunInputs
) -> None:
    before = manifest_payload(build_manifest(complete))["preregistration_sha256"]

    preregistration.write_text("alpha = 0.05\n")
    after = manifest_payload(build_manifest(complete))["preregistration_sha256"]

    assert before != after


def test_the_hash_is_taken_of_the_file_it_was_handed(tmp_path: Path, complete: RunInputs) -> None:
    other = tmp_path / "other.md"
    other.write_text("alpha = 0.0125\nB = 10000\nD_eval = 28\n")

    mine = manifest_payload(build_manifest(complete))
    theirs = manifest_payload(build_manifest(replace(complete, preregistration=other)))

    assert mine["preregistration_sha256"] != theirs["preregistration_sha256"]
    assert theirs["preregistration_path"] == str(other)


def test_the_manifest_lands_alone_in_the_run_directory(tmp_path: Path, complete: RunInputs) -> None:
    root = tmp_path / "tape_studies"

    write_manifest(root, complete)

    assert (root / RUN_ID / MANIFEST_NAME).is_file()
    assert [path.name for path in (root / RUN_ID).iterdir()] == [MANIFEST_NAME]
    assert [path.name for path in root.iterdir()] == [RUN_ID]


def test_the_writer_refuses_to_overwrite_an_existing_manifest(
    tmp_path: Path, complete: RunInputs
) -> None:
    root = tmp_path / "tape_studies"
    digest = write_manifest(root, complete)
    path = root / RUN_ID / MANIFEST_NAME

    with pytest.raises(FileExistsError, match=re.escape(str(path))):
        write_manifest(root, complete)

    assert json.loads(path.read_text())["sha256"] == digest


def test_an_abort_leaves_no_run_directory_behind(tmp_path: Path, complete: RunInputs) -> None:
    root = tmp_path / "tape_studies"

    with pytest.raises(ManifestIncomplete):
        write_manifest(root, replace(complete, row_counts=None))

    assert not root.exists()


def test_an_abort_writes_nothing_into_a_run_directory_that_already_exists(
    tmp_path: Path, complete: RunInputs
) -> None:
    root = tmp_path / "tape_studies"
    (root / RUN_ID).mkdir(parents=True)

    with pytest.raises(ManifestIncomplete):
        write_manifest(root, replace(complete, bootstrap_seed=None))

    assert list((root / RUN_ID).iterdir()) == []


def test_exactly_two_supplied_fields_are_exemptible() -> None:
    assert EXEMPTIBLE_FIELDS == ("r0_fraction_invalid_max", "latency_floor")
    assert set(EXEMPTIBLE_FIELDS) < {field for _, field in SUPPLIED_FIELDS}


def test_an_exempt_r0_threshold_is_recorded_with_its_reason(
    tmp_path: Path, complete: RunInputs
) -> None:
    root = tmp_path / "tape_studies"
    exempt = replace(
        complete,
        universe=None,
        exemptions=(Exemption(field="r0_fraction_invalid_max", reason=NO_TAPE),),
    )

    digest = write_manifest(root, exempt)

    payload = json.loads((root / RUN_ID / MANIFEST_NAME).read_text())
    assert payload.pop("sha256") == digest
    assert freeze_digest(payload) == digest
    assert payload["exemptions"] == [{"field": "r0_fraction_invalid_max", "reason": NO_TAPE}]
    assert payload["r0_fraction_invalid_max"] is None
    assert payload["r0_universe_sha256"] is None
    assert payload["latency_floor_samples"] == 240


def test_an_exempt_latency_floor_is_recorded_with_its_reason(
    tmp_path: Path, complete: RunInputs
) -> None:
    root = tmp_path / "tape_studies"
    exempt = replace(
        complete, floor=None, exemptions=(Exemption(field="latency_floor", reason=NO_READS),)
    )

    digest = write_manifest(root, exempt)

    payload = json.loads((root / RUN_ID / MANIFEST_NAME).read_text())
    assert payload.pop("sha256") == digest
    assert payload["exemptions"] == [{"field": "latency_floor", "reason": NO_READS}]
    assert [payload[key] for key in FLOOR_KEYS] == [None, None, None, None]
    assert payload["r0_fraction_invalid_max"] == str(THRESHOLD)


def test_a_run_that_reads_no_tape_exempts_both_permitted_fields(
    tmp_path: Path, complete: RunInputs
) -> None:
    root = tmp_path / "tape_studies"
    exempt = replace(
        complete,
        universe=None,
        floor=None,
        exemptions=(
            Exemption(field="r0_fraction_invalid_max", reason=NO_TAPE),
            Exemption(field="latency_floor", reason=NO_READS),
        ),
    )

    digest = write_manifest(root, exempt)

    payload = json.loads((root / RUN_ID / MANIFEST_NAME).read_text())
    assert payload.pop("sha256") == digest
    assert set(payload) == FIELDS | {"exemptions"}
    assert payload["exemptions"] == [
        {"field": "latency_floor", "reason": NO_READS},
        {"field": "r0_fraction_invalid_max", "reason": NO_TAPE},
    ]
    assert payload["r0_fraction_invalid_max"] is None
    assert payload["r0_universe_sha256"] is None
    assert [payload[key] for key in FLOOR_KEYS] == [None, None, None, None]
    assert payload["bootstrap_seed"] == SEED


def _pinned(manifest: Manifest) -> dict:
    return manifest_payload(
        replace(
            manifest,
            preregistration=Path("preregistration.md"),
            preregistration_sha256="0" * 64,
            git=GitState(head="1" * 40, dirty=False),
        )
    )


def test_the_order_two_exemptions_arrive_in_does_not_move_the_recorded_digest(
    complete: RunInputs,
) -> None:
    dropped = replace(complete, universe=None, floor=None)
    declared = (
        Exemption(field="latency_floor", reason=NO_READS),
        Exemption(field="r0_fraction_invalid_max", reason=NO_TAPE),
    )
    ascending = build_manifest(replace(dropped, exemptions=declared))
    descending = build_manifest(replace(dropped, exemptions=declared[::-1]))

    payload = _pinned(ascending)

    assert [item.field for item in ascending.exemptions] == [
        "latency_floor",
        "r0_fraction_invalid_max",
    ]
    assert ascending.exemptions == descending.exemptions
    assert [item["field"] for item in payload["exemptions"]] == [
        "latency_floor",
        "r0_fraction_invalid_max",
    ]
    assert freeze_digest(payload) == EXEMPT_DIGEST
    assert freeze_digest(_pinned(descending)) == EXEMPT_DIGEST


@pytest.mark.parametrize(
    "field",
    [
        "bootstrap_seed",
        "economic_bar_price",
        "fee_source",
        "row_counts",
        "git_head",
        "universe",
        "floor",
    ],
)
def test_a_field_the_preregistration_left_closed_cannot_be_exempted(field: str) -> None:
    with pytest.raises(ExemptionRefused) as excinfo:
        Exemption(field=field, reason=NO_TAPE)

    assert field in str(excinfo.value)
    assert not isinstance(excinfo.value, ManifestIncomplete)


@pytest.mark.parametrize("reason", ["", "   ", "\n"])
def test_an_exemption_with_no_stated_reason_is_not_a_declaration(reason: str) -> None:
    with pytest.raises(ExemptionRefused) as excinfo:
        Exemption(field="latency_floor", reason=reason)

    assert "latency_floor" in str(excinfo.value)


def test_an_exemption_reason_padded_with_whitespace_freezes_the_same_digest(
    tmp_path: Path, complete: RunInputs
) -> None:
    padded = replace(
        complete,
        universe=None,
        exemptions=(Exemption(field="r0_fraction_invalid_max", reason=f"  {NO_TAPE}  "),),
    )
    stripped = replace(
        complete,
        universe=None,
        exemptions=(Exemption(field="r0_fraction_invalid_max", reason=NO_TAPE),),
    )

    padded_digest = write_manifest(tmp_path / "padded", padded)
    stripped_digest = write_manifest(tmp_path / "stripped", stripped)

    assert padded_digest == stripped_digest


def test_an_undeclared_gap_still_aborts_the_run(tmp_path: Path, complete: RunInputs) -> None:
    root = tmp_path / "tape_studies"

    with pytest.raises(ManifestIncomplete) as excinfo:
        write_manifest(root, replace(complete, floor=None, exemptions=()))

    assert excinfo.value.fields == ("latency_floor",)
    assert not root.exists()


def test_an_exemption_narrows_nothing_about_a_field_it_did_not_name(
    tmp_path: Path, complete: RunInputs
) -> None:
    root = tmp_path / "tape_studies"
    half = replace(
        complete,
        universe=None,
        floor=None,
        exemptions=(Exemption(field="latency_floor", reason=NO_READS),),
    )

    with pytest.raises(ManifestIncomplete) as excinfo:
        write_manifest(root, half)

    assert excinfo.value.fields == ("r0_fraction_invalid_max",)
    assert "latency_floor" not in str(excinfo.value)
    assert not root.exists()


def test_a_field_the_run_supplied_cannot_also_be_declared_exempt(
    tmp_path: Path, complete: RunInputs
) -> None:
    root = tmp_path / "tape_studies"

    with pytest.raises(ExemptionRefused) as excinfo:
        write_manifest(
            root, replace(complete, exemptions=(Exemption(field="latency_floor", reason=NO_READS),))
        )

    assert "latency_floor" in str(excinfo.value)
    assert not root.exists()


def test_a_run_declaring_no_exemption_writes_the_payload_it_wrote_before(
    complete: RunInputs,
) -> None:
    declared = manifest_payload(build_manifest(replace(complete, exemptions=())))
    undeclared = manifest_payload(build_manifest(complete))

    assert set(declared) == FIELDS
    assert "exemptions" not in declared
    assert declared == undeclared
    assert freeze_digest(declared) == freeze_digest(undeclared)


def test_a_run_naming_no_cohort_writes_the_payload_it_wrote_before(complete: RunInputs) -> None:
    payload = manifest_payload(build_manifest(complete))

    assert complete.cohort is None
    assert "cohort" not in payload
    assert set(payload) == FIELDS


def test_a_run_naming_a_cohort_records_it_and_moves_the_digest(complete: RunInputs) -> None:
    named = manifest_payload(build_manifest(replace(complete, cohort=HIGH)))

    assert named["cohort"] == HIGH
    assert set(named) == FIELDS | {"cohort"}
    assert freeze_digest(named) != freeze_digest(manifest_payload(build_manifest(complete)))


def _frozen_manifest(fee_type_check: FeeRegimeCheck | None) -> Manifest:
    return Manifest(
        run_id=RUN_ID,
        preregistration=Path("preregistration.md"),
        preregistration_sha256="0" * 64,
        accrual_start=ACCRUAL_START,
        accrual_end=ACCRUAL_END,
        row_counts=dict(ROW_COUNTS),
        git=GitState(head="1" * 40, dirty=False),
        r0_fraction_invalid_max=THRESHOLD,
        r0_universe_sha256="2" * 64,
        fee=fee_source(maker_rate=PUBLISHED_MAKER_RATE, maker_rate_source=MAKER_RATE_SOURCE),
        floor=LatencyFloor(
            source=FloorSource.SIGNED_READ, floor_s=0.24, t_persist_s=10.0, n_usable=240
        ),
        economic_bar_size=BAR_SIZE,
        economic_bar_price=BAR_PRICE,
        economic_bar_price_source=BAR_PRICE_SOURCE,
        economic_bar_cents_per_contract=economic_bar_cents_per_contract(BAR_SIZE, BAR_PRICE),
        bootstrap_seed=SEED,
        cohort=None,
        exemptions=(),
        fee_type_check=fee_type_check,
    )


def test_a_run_clearing_no_tripwire_writes_the_payload_it_wrote_before(
    complete: RunInputs,
) -> None:
    payload = manifest_payload(build_manifest(complete))

    assert complete.fee_type_check is None
    assert set(payload).isdisjoint(FEE_TYPE_KEYS)
    assert set(payload) == FIELDS


def test_the_digest_of_a_run_clearing_no_tripwire_has_not_moved() -> None:
    payload = manifest_payload(_frozen_manifest(None))

    assert set(payload) == FIELDS
    assert freeze_digest(payload) == FROZEN_DIGEST


@pytest.mark.parametrize("prec", AMBIENT_PRECISIONS)
def test_the_recorded_digest_does_not_move_with_the_ambient_precision(prec: int) -> None:
    with localcontext(prec=prec):
        payload = manifest_payload(_frozen_manifest(None))
        digest = freeze_digest(payload)

    assert payload["economic_bar_size"] == "26"
    assert payload["economic_bar_cents_per_contract"] == "2.769230769230769230769230769"
    assert digest == FROZEN_DIGEST


@pytest.mark.parametrize("prec", AMBIENT_PRECISIONS)
def test_the_exemption_digest_does_not_move_with_the_ambient_precision(
    complete: RunInputs, prec: int
) -> None:
    dropped = replace(complete, universe=None, floor=None)
    declared = (
        Exemption(field="latency_floor", reason=NO_READS),
        Exemption(field="r0_fraction_invalid_max", reason=NO_TAPE),
    )

    with localcontext(prec=prec):
        digest = freeze_digest(_pinned(build_manifest(replace(dropped, exemptions=declared))))

    assert digest == EXEMPT_DIGEST


def test_the_completed_run_keeps_the_digest_it_recorded() -> None:
    assert freeze_digest(F1_MANIFEST_PAYLOAD) == F1_DIGEST


def test_a_run_clearing_the_tripwire_records_it_and_moves_the_digest() -> None:
    checked = FeeRegimeCheck(result=PLAIN_REGIME, observed_at=OBSERVED_AT, sha256="3" * 64)

    payload = manifest_payload(_frozen_manifest(checked))

    assert set(payload) == FIELDS | set(FEE_TYPE_KEYS)
    assert payload["fee_type_check"] == PLAIN_REGIME
    assert payload["fee_type_observed_at"] == OBSERVED_AT.isoformat()
    assert payload["fee_type_sha256"] == "3" * 64
    assert freeze_digest(payload) != FROZEN_DIGEST


def test_the_tripwire_is_neither_supplied_by_every_family_nor_exemptible() -> None:
    assert "fee_type_check" not in dict(SUPPLIED_FIELDS)
    assert "fee_type_check" not in dict(SUPPLIED_FIELDS).values()
    assert "fee_type_check" not in EXEMPTIBLE_FIELDS


def test_the_tripwire_is_not_a_fee_source_field(complete: RunInputs) -> None:
    payload = fee_payload(complete.fee)

    assert set(payload).isdisjoint(FEE_TYPE_KEYS)


def _settlement() -> SettlementRecord:
    return SettlementRecord(
        provenance=SettlementProvenance(
            observed_at=OBSERVED_AT,
            observation_source="ACIS",
            series={
                root: SeriesSettlementSource(
                    root=root,
                    settlement_source=WEATHER_COMPANY,
                    settlement_source_url=WEATHER_COMPANY_URL,
                    last_updated_ts=MOVED_AT,
                    important_info="",
                    important_info_id="",
                )
                for root in SETTLEMENT_ROOTS
            },
            sha256="4" * 64,
        ),
        boundary=BoundarySplit(
            boundary_date=BOUNDARY_DATE,
            days_before_boundary=12,
            days_on_or_after_boundary=2,
            boundary_source=BOUNDARY_SOURCE,
        ),
    )


def test_a_run_recording_no_settlement_source_writes_the_payload_it_wrote_before() -> None:
    frozen = _frozen_manifest(None)

    payload = manifest_payload(frozen)

    assert frozen.settlement is None
    assert set(payload).isdisjoint(SETTLEMENT_KEYS)
    assert set(payload) == FIELDS
    assert freeze_digest(payload) == FROZEN_DIGEST


def test_a_run_recording_the_settlement_source_keys_it_by_root_and_moves_the_digest() -> None:
    payload = manifest_payload(replace(_frozen_manifest(None), settlement=_settlement()))

    assert set(payload) == FIELDS | set(SETTLEMENT_KEYS)
    assert payload["settlement_source"] == dict.fromkeys(SETTLEMENT_ROOTS, WEATHER_COMPANY)
    assert payload["settlement_source_url"] == dict.fromkeys(SETTLEMENT_ROOTS, WEATHER_COMPANY_URL)
    assert payload["last_updated_ts"] == dict.fromkeys(SETTLEMENT_ROOTS, MOVED_AT.isoformat())
    assert list(payload["settlement_source"]) == sorted(SETTLEMENT_ROOTS)
    assert list(payload["last_updated_ts"]) == sorted(SETTLEMENT_ROOTS)
    assert payload["boundary_date"] == BOUNDARY_DATE.isoformat()
    assert payload["boundary_source"] == "product_metadata.important_info.markdown"
    assert payload["days_before_boundary"] == 12
    assert payload["days_on_or_after_boundary"] == 2
    assert freeze_digest(payload) != FROZEN_DIGEST


def test_the_settlement_source_is_neither_supplied_by_every_family_nor_exemptible() -> None:
    assert "settlement" not in dict(SUPPLIED_FIELDS)
    assert "settlement" not in dict(SUPPLIED_FIELDS).values()
    assert set(SETTLEMENT_KEYS).isdisjoint(dict(SUPPLIED_FIELDS).values())
    assert "settlement" not in EXEMPTIBLE_FIELDS


def test_a_run_that_supplies_the_settlement_source_carries_it_onto_the_manifest(
    complete: RunInputs,
) -> None:
    record = _settlement()

    manifest = build_manifest(replace(complete, settlement=record))

    assert manifest.settlement == record
    assert build_manifest(complete).settlement is None
