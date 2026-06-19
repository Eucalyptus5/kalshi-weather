from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from urllib.parse import parse_qs, urlparse

import httpx

from bot.observations.metar import MetarClient, StationObservation


def _entry(
    icao: str,
    obs_time: int,
    receipt_time: str,
    temp: object,
    raw_ob: str,
) -> dict[str, object]:
    return {
        "icaoId": icao,
        "obsTime": obs_time,
        "receiptTime": receipt_time,
        "reportTime": "2026-06-17 18:00",
        "temp": temp,
        "rawOb": raw_ob,
    }


async def test_single_station_basic() -> None:
    payload = [
        _entry(
            "KDEN",
            1718640000,
            "2026-06-17T16:05:00Z",
            26.5,
            "METAR KDEN 171600Z 27015KT 10SM CLR 26/01 A2992 RMK T02650011",
        )
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = MetarClient(http_client=http)
        obs = await client.fetch_observations(["KDEN"])

    assert len(obs) == 1
    o = obs[0]
    assert isinstance(o, StationObservation)
    assert o.station == "KDEN"
    assert o.temp_f == Decimal("79.7")
    assert o.is_special is False
    assert o.source == "metar"
    assert o.valid_time.tzinfo is not None
    assert o.valid_time.utcoffset() == timezone.utc.utcoffset(o.valid_time)
    assert o.publication_time.tzinfo is not None
    assert o.publication_time.utcoffset() == timezone.utc.utcoffset(o.publication_time)


async def test_t_group_precision() -> None:
    payload = [
        _entry(
            "KDEN",
            1718640000,
            "2026-06-17T16:05:00Z",
            26.5,
            "METAR KDEN 171600Z 27015KT 10SM CLR 27/01 A2992 RMK T02670011",
        )
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = MetarClient(http_client=http)
        obs = await client.fetch_observations(["KDEN"])

    assert len(obs) == 1
    assert obs[0].temp_f == Decimal("80.06")


async def test_no_t_group_falls_back_to_decoded_temp() -> None:
    payload = [
        _entry(
            "KDEN",
            1718640000,
            "2026-06-17T16:05:00Z",
            25.0,
            "METAR KDEN 171600Z 27015KT 10SM CLR 25/01 A2992",
        )
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = MetarClient(http_client=http)
        obs = await client.fetch_observations(["KDEN"])

    assert len(obs) == 1
    assert obs[0].temp_f == Decimal("77")


async def test_speci_flag_set_from_raw_ob() -> None:
    payload = [
        _entry(
            "KMIA",
            1718640000,
            "2026-06-17T16:05:00Z",
            30.0,
            "SPECI KMIA 171605Z 18012KT 10SM CLR 30/22 A2998",
        ),
        _entry(
            "KDEN",
            1718640000,
            "2026-06-17T16:05:00Z",
            26.0,
            "METAR KDEN 171600Z 27015KT 10SM CLR 26/01 A2992",
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = MetarClient(http_client=http)
        obs = await client.fetch_observations(["KMIA", "KDEN"])

    assert len(obs) == 2
    by_station = {o.station: o for o in obs}
    assert by_station["KMIA"].is_special is True
    assert by_station["KDEN"].is_special is False


async def test_missing_temperature_is_skipped() -> None:
    payload = [
        _entry(
            "KDEN",
            1718640000,
            "2026-06-17T16:05:00Z",
            None,
            "METAR KDEN 171600Z 27015KT 10SM CLR //01 A2992",
        ),
        _entry(
            "KMDW",
            1718640000,
            "2026-06-17T16:05:00Z",
            22.0,
            "METAR KMDW 171600Z 27015KT 10SM CLR 22/10 A2992",
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = MetarClient(http_client=http)
        obs = await client.fetch_observations(["KDEN", "KMDW"])

    assert len(obs) == 1
    assert obs[0].station == "KMDW"


async def test_multi_station_request_uses_comma_join() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["req"] = request
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = MetarClient(http_client=http)
        await client.fetch_observations(["KDEN", "KMDW"])

    req = captured["req"]
    parsed = urlparse(str(req.url))
    assert parsed.netloc == "aviationweather.gov"
    assert parsed.path == "/api/data/metar"
    qs = parse_qs(parsed.query)
    assert qs["ids"] == ["KDEN,KMDW"]
    assert qs["format"] == ["json"]
    assert qs["hours"] == ["2.0"]


async def test_receipt_time_parses_both_iso_forms() -> None:
    payload = [
        _entry(
            "KDEN",
            1718640000,
            "2026-06-17T16:05:00Z",
            26.0,
            "METAR KDEN 171600Z 27015KT 10SM CLR 26/01 A2992",
        ),
        _entry(
            "KMDW",
            1718640000,
            "2026-06-17T16:05:00+00:00",
            22.0,
            "METAR KMDW 171600Z 27015KT 10SM CLR 22/10 A2992",
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = MetarClient(http_client=http)
        obs = await client.fetch_observations(["KDEN", "KMDW"])

    assert len(obs) == 2
    assert obs[0].publication_time == obs[1].publication_time
    assert obs[0].publication_time == datetime(2026, 6, 17, 16, 5, tzinfo=timezone.utc)


async def test_obs_time_epoch_parsed_to_utc() -> None:
    epoch = 1718640000
    expected = datetime.fromtimestamp(epoch, tz=timezone.utc)
    payload = [
        _entry(
            "KDEN",
            epoch,
            "2026-06-17T16:05:00Z",
            26.0,
            "METAR KDEN 171600Z 27015KT 10SM CLR 26/01 A2992",
        )
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = MetarClient(http_client=http)
        obs = await client.fetch_observations(["KDEN"])

    assert obs[0].valid_time == expected


async def test_negative_t_group_temperature() -> None:
    payload = [
        _entry(
            "KDEN",
            1718640000,
            "2026-06-17T16:05:00Z",
            -2.5,
            "METAR KDEN 171600Z 27015KT 10SM CLR M02/M05 A2992 RMK T10250011",
        )
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = MetarClient(http_client=http)
        obs = await client.fetch_observations(["KDEN"])

    assert obs[0].temp_f == Decimal("27.5")


async def test_response_not_a_list_raises_value_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "boom"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = MetarClient(http_client=http)
        raised = False
        try:
            await client.fetch_observations(["KDEN"])
        except ValueError:
            raised = True
        assert raised


async def test_owned_http_client_closed_on_aclose(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)
    real_init = httpx.AsyncClient.__init__

    def patched_init(self: httpx.AsyncClient, *args: object, **kwargs: object) -> None:
        kwargs["transport"] = transport
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

    client = MetarClient()
    await client.fetch_observations(["KDEN"])
    await client.aclose()
    assert client._http.is_closed


async def test_caller_owned_http_client_not_closed_by_aclose() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = MetarClient(http_client=http)
        await client.fetch_observations(["KDEN"])
        await client.aclose()
        assert not http.is_closed
