import pytest

from bot.main import RECORDING_SERIES, STATIONS
from bot.replay.analysis_stations import ANALYSIS_STATIONS, LOW_STATIONS, LOW_TO_HIGH


# Read off rules_primary on each open low market, one root at a time. The CLI suffix is a locally
# assigned site id, not the ICAO minus its K, and CLINYC is a COOP station with no ICAO at all.
MEASURED_STATIONS = (
    ("KXLOWTBOS", "KBOS"),
    ("KXLOWTDEN", "KDEN"),
    ("KXLOWTAUS", "KAUS"),
    ("KXLOWTCHI", "KMDW"),
    ("KXLOWTNYC", "KNYC"),
    ("KXLOWTPHIL", "KPHL"),
    ("KXLOWTATL", "KATL"),
    ("KXLOWTDAL", "KDFW"),
    ("KXLOWTDC", "KDCA"),
    ("KXLOWTHOU", "KHOU"),
    ("KXLOWTLV", "KLAS"),
    ("KXLOWTMIN", "KMSP"),
    ("KXLOWTNOLA", "KMSY"),
    ("KXLOWTOKC", "KOKC"),
    ("KXLOWTPHX", "KPHX"),
    ("KXLOWTSATX", "KSAT"),
    ("KXLOWTSEA", "KSEA"),
    ("KXLOWTSFO", "KSFO"),
    ("KXLOWTLAX", "KLAX"),
    ("KXLOWTMIA", "KMIA"),
)


@pytest.mark.parametrize(("root", "station"), MEASURED_STATIONS)
def test_each_low_root_settles_on_the_station_read_off_the_venue(root: str, station: str) -> None:
    assert LOW_STATIONS[root].station == station


def test_the_measured_table_covers_every_low_root_exactly_once() -> None:
    measured = [root for root, _ in MEASURED_STATIONS]

    assert len(measured) == 20
    assert set(measured) == set(LOW_TO_HIGH)
    assert set(measured) == set(LOW_STATIONS)


def test_the_pairing_is_the_recorders_own_low_subscription_list() -> None:
    assert set(LOW_TO_HIGH) == {root for root in RECORDING_SERIES if root.startswith("KXLOW")}
    assert set(LOW_TO_HIGH.values()) == set(STATIONS)
    assert len(STATIONS) == 20


@pytest.mark.parametrize(("low", "high"), sorted(LOW_TO_HIGH.items()))
def test_a_low_root_carries_its_paired_high_geography(low: str, high: str) -> None:
    derived = LOW_STATIONS[low]
    paired = STATIONS[high]

    assert derived.series == low
    assert derived.station == paired.station
    assert derived.timezone == paired.timezone
    assert derived.latitude == paired.latitude
    assert derived.longitude == paired.longitude


def test_the_analysis_map_is_forty_roots_and_carries_no_rain() -> None:
    assert len(ANALYSIS_STATIONS) == 40
    assert set(ANALYSIS_STATIONS) == set(STATIONS) | set(LOW_STATIONS)
    assert not [root for root in ANALYSIS_STATIONS if root.startswith("KXRAIN")]
    assert [root for root in RECORDING_SERIES if root.startswith("KXRAIN")]


def test_the_live_bots_map_is_left_alone() -> None:
    assert all(ANALYSIS_STATIONS[root] is config for root, config in STATIONS.items())
    assert not [root for root in STATIONS if root.startswith("KXLOW")]
