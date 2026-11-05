import json
from datetime import date

import httpx
import pytest

from src import data_collection as dc

PAYLOAD = {"hourly": {"07130": [
    {"id_station": "07130", "dh_utc": "2024-01-01 00:00:00", "temperature": "5.1", "pression": "1012.0",
     "humidite": "88", "vent_moyen": "12", "vent_rafales": "25", "pluie_1h": "0.4"},
    {"id_station": "07130", "dh_utc": "2024-01-01 01:00:00", "temperature": None, "pression": "1011.0",
     "humidite": "90", "vent_moyen": "11", "vent_rafales": None, "pluie_1h": None}]}}


def make_client(handler, **kw):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return dc.InfoclimatClient("tok", "https://example.test/opendata/", backoff=0, min_interval=0,
                               client=http, **kw)


def test_parse_dict_and_list_forms():
    a = dc.parse_infoclimat_payload(PAYLOAD)
    b = dc.parse_infoclimat_payload({"hourly": PAYLOAD["hourly"]["07130"]})
    assert len(a) == 2 and a.equals(b)
    assert a.loc[0, "temperature_c"] == 5.1 and a.loc[1, "rain_1h_mm"] != a.loc[1, "rain_1h_mm"]  # NaN


def test_parse_empty_and_invalid():
    assert dc.parse_infoclimat_payload({"hourly": {}}).empty
    with pytest.raises(dc.CollectionError):
        dc.parse_infoclimat_payload([1, 2])
    with pytest.raises(dc.CollectionError):
        dc.parse_infoclimat_payload({"hourly": [{"foo": 1}]})


def test_missing_token():
    with pytest.raises(dc.MissingTokenError):
        dc.InfoclimatClient(None, "https://x")


def test_success_sends_token_and_dates():
    seen = {}

    def handler(req):
        seen.update(dict(req.url.params))
        return httpx.Response(200, json=PAYLOAD)
    df = make_client(handler).fetch(["07130"], date(2024, 1, 1), date(2024, 1, 2))
    assert len(df) == 2 and seen["token"] == "tok" and seen["start"] == "2024-01-01"


def test_retry_on_500_then_success():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(500) if calls["n"] < 3 else httpx.Response(200, json=PAYLOAD)
    assert len(make_client(handler).fetch(["07130"], date(2024, 1, 1), date(2024, 1, 1))) == 2
    assert calls["n"] == 3


def test_403_not_retried():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(403)
    with pytest.raises(dc.CollectionError, match="refus"):
        make_client(handler).fetch(["07130"], date(2024, 1, 1), date(2024, 1, 1))
    assert calls["n"] == 1


def test_timeout_exhausts_retries():
    def handler(req):
        raise httpx.ReadTimeout("lent", request=req)
    with pytest.raises(dc.CollectionError, match="tentatives"):
        make_client(handler, max_retries=2).fetch(["07130"], date(2024, 1, 1), date(2024, 1, 1))


def test_invalid_json():
    with pytest.raises(dc.CollectionError, match="JSON"):
        make_client(lambda r: httpx.Response(200, content=b"<html>")).fetch(["07130"], date(2024, 1, 1), date(2024, 1, 1))


def test_period_is_chunked():
    starts = []

    def handler(req):
        starts.append(req.url.params["start"])
        return httpx.Response(200, json={"hourly": {}})
    make_client(handler, chunk_days=30).fetch(["07130"], date(2024, 1, 1), date(2024, 3, 15))
    assert starts == ["2024-01-01", "2024-01-31", "2024-03-01"]
