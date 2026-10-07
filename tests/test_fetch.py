from datetime import date

import pandas as pd
import pytest
import requests

from airq import fetch
from airq.fetch import FetchError, fetch_city, get_hourly, write_csv_gz


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def ok_payload(variables, n=3):
    hourly = {"time": [f"2024-01-01T0{i}:00" for i in range(n)]}
    for v in variables:
        hourly[v] = [float(i) for i in range(n)]
    return {"hourly": hourly}


def queue_responses(monkeypatch, items):
    """Make requests.get return (or raise) each item in turn; return the call log."""
    calls = []
    items = list(items)

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        item = items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    return calls


def test_retries_after_server_error_then_succeeds(monkeypatch):
    calls = queue_responses(monkeypatch, [
        FakeResponse(503, {"error": True, "reason": "busy"}),
        FakeResponse(200, ok_payload(["pm2_5"])),
    ])
    sleeps = []

    df = get_hourly("http://x", {}, ["pm2_5"], sleep=sleeps.append)

    assert len(calls) == 2
    assert sleeps == [1.0]
    assert list(df.columns) == ["time", "pm2_5"]
    assert len(df) == 3


def test_client_error_fails_fast_with_api_reason(monkeypatch):
    calls = queue_responses(monkeypatch, [
        FakeResponse(400, {"error": True, "reason": "Parameter 'hourly' is invalid"}),
    ])

    with pytest.raises(FetchError, match="Parameter 'hourly' is invalid"):
        get_hourly("http://x", {}, ["pm2_5"], sleep=lambda s: None)
    assert len(calls) == 1


def test_network_down_gives_up_after_all_retries(monkeypatch):
    calls = queue_responses(monkeypatch, [requests.ConnectionError("down")] * 4)
    sleeps = []

    with pytest.raises(FetchError, match="after 4 attempts"):
        get_hourly("http://x", {}, ["pm2_5"], retries=3, sleep=sleeps.append)
    assert len(calls) == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_missing_variable_in_response_is_an_error(monkeypatch):
    queue_responses(monkeypatch, [FakeResponse(200, ok_payload(["pm2_5"]))])

    with pytest.raises(FetchError, match="pm10"):
        get_hourly("http://x", {}, ["pm2_5", "pm10"], sleep=lambda s: None)


def test_unequal_array_lengths_are_an_error(monkeypatch):
    payload = ok_payload(["pm2_5"])
    payload["hourly"]["pm2_5"] = payload["hourly"]["pm2_5"][:-1]
    queue_responses(monkeypatch, [FakeResponse(200, payload)])

    with pytest.raises(FetchError, match="length"):
        get_hourly("http://x", {}, ["pm2_5"], sleep=lambda s: None)


def test_failed_weather_download_writes_nothing(monkeypatch, tmp_path):
    from airq.cities import AIR_VARS

    queue_responses(monkeypatch, [
        FakeResponse(200, ok_payload(AIR_VARS)),
        FakeResponse(400, {"error": True, "reason": "bad date"}),
    ])

    with pytest.raises(FetchError):
        fetch_city("surat", date(2024, 1, 1), date(2024, 1, 2), tmp_path, sleep=lambda s: None)
    assert list(tmp_path.iterdir()) == []


def test_fetch_city_writes_air_and_weather_files(monkeypatch, tmp_path):
    from airq.cities import AIR_VARS, WEATHER_VARS

    queue_responses(monkeypatch, [
        FakeResponse(200, ok_payload(AIR_VARS)),
        FakeResponse(200, ok_payload(WEATHER_VARS)),
    ])

    paths = fetch_city("surat", date(2024, 1, 1), date(2024, 1, 2), tmp_path, sleep=lambda s: None)

    assert sorted(p.name for p in paths) == ["surat_air.csv.gz", "surat_weather.csv.gz"]
    air = pd.read_csv(tmp_path / "surat_air.csv.gz")
    assert list(air.columns) == ["time"] + AIR_VARS


def test_same_data_written_twice_is_byte_identical(tmp_path):
    df = pd.DataFrame({"time": ["2024-01-01T00:00"], "pm2_5": [12.5]})

    a = write_csv_gz(df, tmp_path / "a.csv.gz").read_bytes()
    b = write_csv_gz(df, tmp_path / "b.csv.gz").read_bytes()

    assert a == b
    assert not list(tmp_path.glob("*.tmp"))
