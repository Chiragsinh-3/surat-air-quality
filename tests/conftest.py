import numpy as np
import pandas as pd
import pytest

from airq.cities import AIR_VARS, WEATHER_VARS
from airq.fetch import write_csv_gz


def synthetic_raw(city, seed, start="2025-06-01", end="2025-10-31 23:00"):
    """Raw air and weather tables shaped like the Open-Meteo downloads, where
    PM2.5 really does depend on the weather. Covers the regression split
    (2025-08-01) and Diwali 2025."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h")
    n = len(idx)
    daylight = np.clip(np.sin((idx.hour.to_numpy() - 6) / 24 * 2 * np.pi), 0, None)
    blh = 150 + 1200 * daylight + rng.uniform(0, 200, n)
    wind = rng.uniform(1, 25, n)
    rain = rng.exponential(1.0, n) * (rng.random(n) < 0.1)
    wdir = rng.uniform(0, 360, n)
    pm25 = np.exp(5.5 - 0.4 * np.log(blh) - 0.02 * wind - 0.3 * np.log1p(rain)
                  + 0.2 * np.cos(np.deg2rad(wdir)) + rng.normal(0, 0.1, n))
    times = idx.strftime("%Y-%m-%dT%H:%M")
    air = pd.DataFrame({"time": times, "pm2_5": pm25.round(1), "pm10": (pm25 * 1.6).round(1),
                        "nitrogen_dioxide": rng.uniform(5, 30, n).round(1)})
    weather = pd.DataFrame({
        "time": times,
        "temperature_2m": rng.uniform(22, 36, n).round(1),
        "relative_humidity_2m": rng.uniform(40, 95, n).round(0),
        "precipitation": rain.round(1),
        "wind_speed_10m": wind.round(1),
        "wind_direction_10m": wdir.round(0),
        "boundary_layer_height": blh.round(0),
    })
    assert list(air.columns[1:]) == AIR_VARS and list(weather.columns[1:]) == WEATHER_VARS
    return air, weather


@pytest.fixture
def raw_dir(tmp_path):
    out = tmp_path / "raw"
    out.mkdir()
    for seed, city in enumerate(["surat", "ahmedabad"]):
        air, weather = synthetic_raw(city, seed)
        write_csv_gz(air, out / f"{city}_air.csv.gz")
        write_csv_gz(weather, out / f"{city}_weather.csv.gz")
    return out
