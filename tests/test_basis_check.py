from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from urllib.parse import parse_qs, urlparse

import httpx

from bot.observations.basis_check import (
    BasisCompareRow,
    fetch_iem_1min_asos_archive,
    _fetch_iowa_asos_archive,
    _integer_tolerant_basis_valid,
    compare_basis,
    summarize_basis,
)
from bot.observations.metar import MetarClient
from bot.validation.reconcile import ACISClient


UTC = timezone.utc


def _acis_resp(day: date, value: object) -> dict[str, object]:
    return {
        "meta": {"name": "DENVER INTL AP", "sids": ["KDEN 1"]},
        "data": [[day.isoformat(), value]],
    }


def _iowa_csv(rows: list[tuple[str, str, str]]) -> str:
    body = "\n".join(f"{s},{v},{t}" for s, v, t in rows)
    return "station,valid,tmpc\n" + body + "\n"


def _multi_route_transport(
    iowa_payload: str | None = None,
    acis_by_date: dict[date, object] | None = None,
    acis_capture: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "mesonet.agron.iastate.edu":
            assert iowa_payload is not None
            return httpx.Response(200, text=iowa_payload)
        if host == "data.rcc-acis.org":
            if acis_capture is not None:
                acis_capture.append(request)
            assert acis_by_date is not None
            qs = parse_qs(urlparse(str(request.url)).query)
            sdate = date.fromisoformat(qs["sdate"][0])
            value = acis_by_date.get(sdate)
            if value is None:
                return httpx.Response(200, json={"meta": {}, "data": []})
            return httpx.Response(200, json=_acis_resp(sdate, value))
        raise AssertionError(f"unexpected host {host}")

    return httpx.MockTransport(handler)


async def test_iowa_archive_agree_golden() -> None:
    csv = _iowa_csv(
        [
            ("KDEN", "2026-01-15 18:00", "22.0"),
            ("KDEN", "2026-01-15 22:00", "25.0"),
            ("KDEN", "2026-01-16 02:00", "20.0"),
        ]
    )
    acis = {date(2026, 1, 15): "77"}
    transport = _multi_route_transport(iowa_payload=csv, acis_by_date=acis)
    async with httpx.AsyncClient(transport=transport) as http:
        acis_client = ACISClient(http_client=http)
        rows = await compare_basis(
            "KDEN",
            date(2026, 1, 15),
            date(2026, 1, 15),
            acis_client,
            None,
            source="iowa_asos_archive",
            http_client=http,
        )

    assert len(rows) == 1
    row = rows[0]
    assert row.station == "KDEN"
    assert row.observation_day == date(2026, 1, 15)
    assert row.metar_max_f == Decimal("77")
    assert row.acis_high_f == Decimal("77")
    assert row.delta_f == Decimal("0")
    assert row.basis_valid is True


async def test_iowa_archive_miss_by_2_4f() -> None:
    csv = _iowa_csv(
        [
            ("KDEN", "2026-01-15 18:00", "20.0"),
            ("KDEN", "2026-01-15 22:00", "23.0"),
        ]
    )
    acis = {date(2026, 1, 15): "71"}
    transport = _multi_route_transport(iowa_payload=csv, acis_by_date=acis)
    async with httpx.AsyncClient(transport=transport) as http:
        acis_client = ACISClient(http_client=http)
        rows = await compare_basis(
            "KDEN",
            date(2026, 1, 15),
            date(2026, 1, 15),
            acis_client,
            None,
            source="iowa_asos_archive",
            http_client=http,
        )

    assert len(rows) == 1
    row = rows[0]
    assert row.metar_max_f == Decimal("73.4")
    assert row.acis_high_f == Decimal("71")
    assert row.delta_f == Decimal("2.4")
    assert row.basis_valid is False


async def test_iowa_archive_parser_skips_m_and_header() -> None:
    csv = (
        "station,valid,tmpc\n"
        "KDEN,2026-01-15 18:00,22.0\n"
        "KDEN,2026-01-15 18:05,M\n"
        "KDEN,2026-01-15 18:10,M\n"
        "KDEN,2026-01-15 18:15,24.5\n"
    )
    transport = _multi_route_transport(iowa_payload=csv)
    async with httpx.AsyncClient(transport=transport) as http:
        obs = await _fetch_iowa_asos_archive(
            "KDEN",
            date(2026, 1, 15),
            date(2026, 1, 15),
            http,
        )
    assert len(obs) == 2
    o0 = obs[0]
    assert o0.station == "KDEN"
    assert o0.temp_f == Decimal("71.6")
    assert type(o0.temp_f) is Decimal
    assert o0.valid_time == datetime(2026, 1, 15, 18, 0, tzinfo=UTC)
    assert o0.publication_time == o0.valid_time
    assert o0.source == "iowa_asos_archive"
    assert o0.is_special is False
    assert o0.raw == ""
    o1 = obs[1]
    assert o1.temp_f == Decimal("24.5") * Decimal("9") / Decimal("5") + Decimal("32")


async def test_iowa_archive_request_params() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["req"] = request
        return httpx.Response(200, text="station,valid,tmpc\n")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        await _fetch_iowa_asos_archive(
            "KDEN",
            date(2026, 1, 15),
            date(2026, 1, 17),
            http,
        )
    req = captured["req"]
    parsed = urlparse(str(req.url))
    assert parsed.netloc == "mesonet.agron.iastate.edu"
    assert parsed.path == "/cgi-bin/request/asos.py"
    qs = parse_qs(parsed.query)
    assert qs["station"] == ["KDEN"]
    assert qs["data"] == ["tmpc"]
    assert qs["report_type"] == ["3", "4"]
    assert qs["year1"] == ["2026"]
    assert qs["month1"] == ["1"]
    assert qs["day1"] == ["15"]
    assert qs["year2"] == ["2026"]
    assert qs["month2"] == ["1"]
    assert qs["day2"] == ["17"]
    assert qs["format"] == ["onlycomma"]


async def test_day_grouping_dst_boundary_kden() -> None:
    csv = (
        "station,valid,tmpc\n"
        "KDEN,2026-01-15 06:30,1.0\n"
        "KDEN,2026-01-15 07:30,2.0\n"
        "KDEN,2026-01-16 06:30,3.0\n"
    )
    acis = {
        date(2026, 1, 14): "33",
        date(2026, 1, 15): "35",
    }
    transport = _multi_route_transport(iowa_payload=csv, acis_by_date=acis)
    async with httpx.AsyncClient(transport=transport) as http:
        acis_client = ACISClient(http_client=http)
        rows = await compare_basis(
            "KDEN",
            date(2026, 1, 14),
            date(2026, 1, 15),
            acis_client,
            None,
            source="iowa_asos_archive",
            http_client=http,
        )

    by_day = {r.observation_day: r for r in rows}
    assert set(by_day.keys()) == {date(2026, 1, 14), date(2026, 1, 15)}
    assert by_day[date(2026, 1, 14)].metar_max_f == Decimal("1.0") * Decimal("9") / Decimal(
        "5"
    ) + Decimal("32")
    assert by_day[date(2026, 1, 15)].metar_max_f == Decimal("3.0") * Decimal("9") / Decimal(
        "5"
    ) + Decimal("32")


async def test_missing_acis_drops_day() -> None:
    csv = "station,valid,tmpc\nKDEN,2026-01-15 18:00,22.0\nKDEN,2026-01-16 18:00,23.0\n"
    acis = {date(2026, 1, 16): "73"}
    transport = _multi_route_transport(iowa_payload=csv, acis_by_date=acis)
    async with httpx.AsyncClient(transport=transport) as http:
        acis_client = ACISClient(http_client=http)
        rows = await compare_basis(
            "KDEN",
            date(2026, 1, 15),
            date(2026, 1, 16),
            acis_client,
            None,
            source="iowa_asos_archive",
            http_client=http,
        )
    assert len(rows) == 1
    assert rows[0].observation_day == date(2026, 1, 16)


async def test_acis_station_code_strips_leading_k() -> None:
    captured: list[httpx.Request] = []
    csv = "station,valid,tmpc\nKDEN,2026-01-15 18:00,22.0\n"
    transport = _multi_route_transport(
        iowa_payload=csv,
        acis_by_date={date(2026, 1, 15): "71"},
        acis_capture=captured,
    )
    async with httpx.AsyncClient(transport=transport) as http:
        acis_client = ACISClient(http_client=http)
        await compare_basis(
            "KDEN",
            date(2026, 1, 15),
            date(2026, 1, 15),
            acis_client,
            None,
            source="iowa_asos_archive",
            http_client=http,
        )

    assert len(captured) == 1
    qs = parse_qs(urlparse(str(captured[0].url)).query)
    assert qs["sid"] == ["DEN"]


async def test_live_metar_path() -> None:
    metar_payload = [
        {
            "icaoId": "KDEN",
            "obsTime": int(datetime(2026, 7, 15, 18, 0, tzinfo=UTC).timestamp()),
            "receiptTime": "2026-07-15T18:05:00Z",
            "reportTime": "2026-07-15 18:00",
            "temp": 30.0,
            "rawOb": "METAR KDEN 151800Z 27015KT 10SM CLR 30/05 A2992",
        },
        {
            "icaoId": "KDEN",
            "obsTime": int(datetime(2026, 7, 15, 22, 0, tzinfo=UTC).timestamp()),
            "receiptTime": "2026-07-15T22:05:00Z",
            "reportTime": "2026-07-15 22:00",
            "temp": 33.0,
            "rawOb": "METAR KDEN 152200Z 27015KT 10SM CLR 33/05 A2992",
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "aviationweather.gov":
            return httpx.Response(200, json=metar_payload)
        if host == "data.rcc-acis.org":
            return httpx.Response(200, json=_acis_resp(date(2026, 7, 15), "90"))
        raise AssertionError(f"unexpected host {host}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        acis_client = ACISClient(http_client=http)
        metar_client = MetarClient(http_client=http)
        rows = await compare_basis(
            "KDEN",
            date(2026, 7, 15),
            date(2026, 7, 15),
            acis_client,
            metar_client,
            source="live_metar",
        )

    assert len(rows) == 1
    row = rows[0]
    assert row.station == "KDEN"
    assert row.observation_day == date(2026, 7, 15)
    assert row.metar_max_f == Decimal("33") * Decimal("9") / Decimal("5") + Decimal("32")
    assert row.acis_high_f == Decimal("90")
    assert row.delta_f == row.metar_max_f - Decimal("90")


def test_summarize_basis_aggregates() -> None:
    deltas = [
        Decimal("-2.5"),
        Decimal("-1.5"),
        Decimal("-0.8"),
        Decimal("-0.3"),
        Decimal("0.0"),
        Decimal("0.2"),
        Decimal("0.5"),
        Decimal("0.7"),
        Decimal("1.2"),
        Decimal("1.8"),
        Decimal("3.0"),
    ]
    rows = [
        BasisCompareRow(
            station="KDEN",
            observation_day=date(2026, 1, 15) + timedelta(days=i),
            metar_max_f=Decimal("70") + d,
            acis_high_f=Decimal("70"),
            delta_f=d,
            basis_valid=_integer_tolerant_basis_valid(Decimal("70") + d, Decimal("70")),
        )
        for i, d in enumerate(deltas)
    ]
    summaries = summarize_basis(rows)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.station == "KDEN"
    assert s.n_days == 11
    assert s.median_delta_f == Decimal("0.2")
    assert s.p10_delta_f == Decimal("-1.5")
    assert s.p90_delta_f == Decimal("1.8")
    assert s.fraction_invalid == Decimal("4") / Decimal("11")
    assert type(s.median_delta_f) is Decimal
    assert type(s.fraction_invalid) is Decimal


def test_summarize_empty_input() -> None:
    assert summarize_basis([]) == []


def test_summarize_groups_by_station() -> None:
    rows = [
        BasisCompareRow(
            station="KDEN",
            observation_day=date(2026, 1, 15),
            metar_max_f=Decimal("70"),
            acis_high_f=Decimal("70"),
            delta_f=Decimal("0"),
            basis_valid=True,
        ),
        BasisCompareRow(
            station="KAUS",
            observation_day=date(2026, 1, 15),
            metar_max_f=Decimal("85"),
            acis_high_f=Decimal("82"),
            delta_f=Decimal("3"),
            basis_valid=False,
        ),
    ]
    summaries = summarize_basis(rows)
    by_station = {s.station: s for s in summaries}
    assert set(by_station) == {"KDEN", "KAUS"}
    assert by_station["KDEN"].fraction_invalid == Decimal("0")
    assert by_station["KAUS"].fraction_invalid == Decimal("1")


async def test_compare_basis_resolves_ksea_timezone() -> None:
    csv = "station,valid,tmpc\nKSEA,2026-07-15 22:00,25.0\n"
    acis = {date(2026, 7, 15): "77"}

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "mesonet.agron.iastate.edu":
            return httpx.Response(200, text=csv)
        if host == "data.rcc-acis.org":
            qs = parse_qs(urlparse(str(request.url)).query)
            sdate = date.fromisoformat(qs["sdate"][0])
            assert qs["sid"] == ["SEA"]
            return httpx.Response(
                200,
                json={
                    "meta": {"name": "SEATTLE", "sids": ["KSEA 1"]},
                    "data": [[sdate.isoformat(), acis[sdate]]],
                },
            )
        raise AssertionError(f"unexpected host {host}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        acis_client = ACISClient(http_client=http)
        rows = await compare_basis(
            "KSEA",
            date(2026, 7, 15),
            date(2026, 7, 15),
            acis_client,
            None,
            source="iowa_asos_archive",
            http_client=http,
        )

    assert len(rows) == 1
    assert rows[0].station == "KSEA"


async def test_delta_f_is_decimal_end_to_end() -> None:
    csv = "station,valid,tmpc\nKDEN,2026-01-15 18:00,22.0\n"
    acis = {date(2026, 1, 15): "70"}
    transport = _multi_route_transport(iowa_payload=csv, acis_by_date=acis)
    async with httpx.AsyncClient(transport=transport) as http:
        acis_client = ACISClient(http_client=http)
        rows = await compare_basis(
            "KDEN",
            date(2026, 1, 15),
            date(2026, 1, 15),
            acis_client,
            None,
            source="iowa_asos_archive",
            http_client=http,
        )

    assert len(rows) == 1
    assert type(rows[0].delta_f) is Decimal
    assert type(rows[0].metar_max_f) is Decimal
    assert type(rows[0].acis_high_f) is Decimal


def _iem_1min_csv(rows: list[tuple[str, str, str, str]]) -> str:
    body = "\n".join(f"{s},{n},{v},{t}" for s, n, v, t in rows)
    return "station,station_name,valid(UTC),tmpf\n" + body + "\n"


async def test_iem_1min_parser_skips_missing_and_bad_timestamps() -> None:
    csv = _iem_1min_csv(
        [
            ("DEN", "DENVER INTL", "2026-05-18 18:00", "80.0"),
            ("DEN", "DENVER INTL", "2026-05-18 18:01", "M"),
            ("DEN", "DENVER INTL", "not-a-date", "75.0"),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "mesonet.agron.iastate.edu"
        return httpx.Response(200, text=csv)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        obs = await fetch_iem_1min_asos_archive(
            "KDEN",
            date(2026, 5, 18),
            date(2026, 5, 18),
            http,
        )

    assert len(obs) == 1
    o = obs[0]
    assert o.station == "KDEN"
    assert o.temp_f == Decimal("80.0")
    assert type(o.temp_f) is Decimal
    assert o.valid_time == datetime(2026, 5, 18, 18, 0, tzinfo=UTC)
    assert o.publication_time == o.valid_time
    assert o.is_special is False
    assert o.raw == ""
    assert o.source == "iem_1min_asos_archive"


async def test_iem_1min_request_shape() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["req"] = request
        return httpx.Response(200, text="station,station_name,valid(UTC),tmpf\n")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        await fetch_iem_1min_asos_archive(
            "KDEN",
            date(2026, 5, 18),
            date(2026, 5, 20),
            http,
        )
    req = captured["req"]
    parsed = urlparse(str(req.url))
    assert parsed.netloc == "mesonet.agron.iastate.edu"
    assert parsed.path == "/cgi-bin/request/asos1min.py"
    qs = parse_qs(parsed.query)
    assert qs["station"] == ["DEN"]
    assert qs["vars"] == ["tmpf"]
    assert qs["sts"] == ["2026-05-18T00:00Z"]
    assert qs["ets"] == ["2026-05-22T00:00Z"]
    assert qs["sample"] == ["1min"]
    assert qs["tz"] == ["UTC"]
    assert qs["format"] == ["onlycomma"]


async def test_iem_1min_unexpected_header_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="foo,bar,baz,qux\n")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        try:
            await fetch_iem_1min_asos_archive(
                "KDEN",
                date(2026, 5, 18),
                date(2026, 5, 18),
                http,
            )
        except ValueError as exc:
            assert "KDEN" in str(exc)
        else:
            raise AssertionError("expected ValueError on unexpected header")


def _iem_route_transport(
    iem_payload: str,
    acis_by_date: dict[date, object],
    iem_capture: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "mesonet.agron.iastate.edu":
            if iem_capture is not None:
                iem_capture.append(request)
            return httpx.Response(200, text=iem_payload)
        if host == "data.rcc-acis.org":
            qs = parse_qs(urlparse(str(request.url)).query)
            sdate = date.fromisoformat(qs["sdate"][0])
            value = acis_by_date.get(sdate)
            if value is None:
                return httpx.Response(200, json={"meta": {}, "data": []})
            return httpx.Response(200, json=_acis_resp(sdate, value))
        raise AssertionError(f"unexpected host {host}")

    return httpx.MockTransport(handler)


async def test_iem_1min_integer_tolerant_equal() -> None:
    csv = _iem_1min_csv(
        [
            ("DEN", "DENVER INTL", "2026-05-18 18:00", "78.0"),
            ("DEN", "DENVER INTL", "2026-05-18 22:00", "80.0"),
        ]
    )
    transport = _iem_route_transport(csv, {date(2026, 5, 18): "80"})
    async with httpx.AsyncClient(transport=transport) as http:
        acis_client = ACISClient(http_client=http)
        rows = await compare_basis(
            "KDEN",
            date(2026, 5, 18),
            date(2026, 5, 18),
            acis_client,
            None,
            source="iem_1min_asos_archive",
            http_client=http,
        )

    assert len(rows) == 1
    row = rows[0]
    assert row.metar_max_f == Decimal("80.0")
    assert row.acis_high_f == Decimal("80")
    assert row.delta_f == Decimal("0.0")
    assert row.basis_valid is True


async def test_iem_1min_integer_tolerant_floor_accepts() -> None:
    csv = _iem_1min_csv(
        [
            ("DEN", "DENVER INTL", "2026-05-18 18:00", "79.0"),
            ("DEN", "DENVER INTL", "2026-05-18 22:00", "81.0"),
        ]
    )
    transport = _iem_route_transport(csv, {date(2026, 5, 18): "80"})
    async with httpx.AsyncClient(transport=transport) as http:
        acis_client = ACISClient(http_client=http)
        rows = await compare_basis(
            "KDEN",
            date(2026, 5, 18),
            date(2026, 5, 18),
            acis_client,
            None,
            source="iem_1min_asos_archive",
            http_client=http,
        )

    assert len(rows) == 1
    row = rows[0]
    assert row.metar_max_f == Decimal("81.0")
    assert row.acis_high_f == Decimal("80")
    assert row.delta_f == Decimal("1.0")
    assert row.basis_valid is True


async def test_iem_1min_integer_tolerant_rejects_above_floor() -> None:
    csv = _iem_1min_csv(
        [
            ("DEN", "DENVER INTL", "2026-05-18 18:00", "79.0"),
            ("DEN", "DENVER INTL", "2026-05-18 22:00", "82.0"),
        ]
    )
    transport = _iem_route_transport(csv, {date(2026, 5, 18): "80"})
    async with httpx.AsyncClient(transport=transport) as http:
        acis_client = ACISClient(http_client=http)
        rows = await compare_basis(
            "KDEN",
            date(2026, 5, 18),
            date(2026, 5, 18),
            acis_client,
            None,
            source="iem_1min_asos_archive",
            http_client=http,
        )

    assert len(rows) == 1
    row = rows[0]
    assert row.metar_max_f == Decimal("82.0")
    assert row.acis_high_f == Decimal("80")
    assert row.delta_f == Decimal("2.0")
    assert row.basis_valid is False


def test_integer_tolerant_basis_valid_half_even_accepts() -> None:
    assert _integer_tolerant_basis_valid(Decimal("80.5"), Decimal("81")) is True


def test_integer_tolerant_basis_valid_half_even_rejects() -> None:
    assert _integer_tolerant_basis_valid(Decimal("82.5"), Decimal("80")) is False


async def test_compare_basis_default_source_is_iem_1min() -> None:
    csv = _iem_1min_csv(
        [
            ("DEN", "DENVER INTL", "2026-05-18 22:00", "80.0"),
        ]
    )
    captured: list[httpx.Request] = []
    transport = _iem_route_transport(
        csv,
        {date(2026, 5, 18): "80"},
        iem_capture=captured,
    )
    async with httpx.AsyncClient(transport=transport) as http:
        acis_client = ACISClient(http_client=http)
        rows = await compare_basis(
            "KDEN",
            date(2026, 5, 18),
            date(2026, 5, 18),
            acis_client,
            None,
            http_client=http,
        )

    assert len(rows) == 1
    assert any(urlparse(str(req.url)).path == "/cgi-bin/request/asos1min.py" for req in captured)
