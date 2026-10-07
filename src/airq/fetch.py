"""Download hourly air quality and weather from Open-Meteo into gzipped CSVs."""
import gzip
import os
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from airq.cities import AIR_VARS, CITIES, WEATHER_VARS

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEOUT = 60


class FetchError(RuntimeError):
    pass


def get_hourly(url, params, variables, *, retries=3, backoff=1.0, sleep=time.sleep):
    """Fetch one Open-Meteo hourly series.

    Timeouts, connection errors and 5xx responses are retried with exponential
    backoff (1s, 2s, 4s...). A 4xx means the request itself is wrong, so it
    fails straight away with the API's reason.
    """
    query = {**params, "hourly": ",".join(variables)}
    attempts = retries + 1
    last_error = ""

    for attempt in range(attempts):
        if attempt:
            sleep(backoff * 2 ** (attempt - 1))
        try:
            resp = requests.get(url, params=query, timeout=TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as e:
            last_error = str(e)
            continue

        if resp.status_code >= 500:
            last_error = f"HTTP {resp.status_code}"
            continue
        if resp.status_code >= 400:
            raise FetchError(f"HTTP {resp.status_code}: {_reason(resp)}")
        return _to_frame(resp.json(), variables)

    raise FetchError(f"{url} failed after {attempts} attempts: {last_error}")


def _reason(resp):
    try:
        return resp.json().get("reason", "no reason given")
    except ValueError:
        return "no reason given"


def _to_frame(payload, variables):
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise FetchError("response has no hourly data")

    missing = [v for v in variables if v not in hourly]
    if missing:
        raise FetchError(f"response is missing variables: {', '.join(missing)}")

    n = len(hourly["time"])
    bad = [v for v in variables if len(hourly[v]) != n]
    if bad:
        raise FetchError(f"length mismatch with time ({n}) for: {', '.join(bad)}")

    return pd.DataFrame({"time": hourly["time"], **{v: hourly[v] for v in variables}})


def write_csv_gz(df, path):
    """Write atomically (temp file + rename). The gzip header gets no file name
    and a fixed timestamp, so the same data always produces the same bytes."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(df.to_csv(index=False).encode("utf-8"))
    os.replace(tmp, path)
    return path


def fetch_city(city, start: date, end: date, out_dir, **kw):
    """Download air and weather for one city. Both downloads must succeed
    before either file is written."""
    lat, lon = CITIES[city]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "Asia/Kolkata",
    }
    air = get_hourly(AIR_URL, params, AIR_VARS, **kw)
    weather = get_hourly(WEATHER_URL, params, WEATHER_VARS, **kw)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        write_csv_gz(air, out_dir / f"{city}_air.csv.gz"),
        write_csv_gz(weather, out_dir / f"{city}_weather.csv.gz"),
    ]
