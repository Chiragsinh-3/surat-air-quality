"""Turn the raw snapshot into tidy hourly and daily tables."""
from pathlib import Path

import numpy as np
import pandas as pd

from airq.cities import AIR_VARS, CITIES, WEATHER_VARS

MAX_GAP = 3  # hours; longer gaps stay missing
# NO2 at exactly zero for this many hours in a row means the model had an
# outage (real city air always has some NO2). Seen 2022-12-13 to 12-16.
MODEL_OUTAGE_HOURS = 6
MIN_HOURS = 16  # CPCB needs 16 valid hours for a 24-hour average

# IMD seasons
SEASON_OF_MONTH = {
    1: "winter", 2: "winter",
    3: "pre-monsoon", 4: "pre-monsoon", 5: "pre-monsoon",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post-monsoon", 11: "post-monsoon", 12: "post-monsoon",
}
SEASONS = ["winter", "pre-monsoon", "monsoon", "post-monsoon"]

# Daily values: rain is a total, wind direction a vector mean, the rest plain means.
MEAN_COLS = AIR_VARS + [v for v in WEATHER_VARS if v not in ("precipitation", "wind_direction_10m")]


class MissingDataError(FileNotFoundError):
    pass


def season(month: int) -> str:
    return SEASON_OF_MONTH[month]


def read_raw(path, columns):
    path = Path(path)
    if not path.exists():
        raise MissingDataError(f"{path.resolve()} not found - run `python -m airq fetch` first")
    df = pd.read_csv(path, parse_dates=["time"])
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    return df.set_index("time")[columns]


def fill_short_gaps(s: pd.Series, max_gap: int = MAX_GAP) -> pd.Series:
    """Linearly fill runs of at most `max_gap` missing values that sit between
    two real values. Longer runs, and missing values at the edges, stay NaN."""
    missing = s.isna()
    run_id = (~missing).cumsum()  # every missing value shares the id of the value before it
    run_len = missing.groupby(run_id).transform("sum")
    filled = s.interpolate(limit_area="inside")
    return s.where(~(missing & (run_len <= max_gap)), filled)


def _long_runs(mask: pd.Series, min_len: int) -> pd.Series:
    """True where `mask` is part of a run of at least `min_len` Trues."""
    run_id = (mask != mask.shift()).cumsum()
    return mask & (mask.groupby(run_id).transform("size") >= min_len)


def clean_hourly(df):
    """Sanity-check pollutant columns. Returns the cleaned frame and counts."""
    df = df.copy()
    counts = {}

    outage = _long_runs(df["nitrogen_dioxide"] == 0, MODEL_OUTAGE_HOURS)
    counts["model_outage_hours"] = int(outage.sum())
    df.loc[outage, AIR_VARS] = np.nan

    negative = df[AIR_VARS] < 0
    counts["negatives"] = int(negative.sum().sum())
    df[AIR_VARS] = df[AIR_VARS].mask(negative)

    # PM2.5 is part of PM10, so PM2.5 > PM10 can't be real.
    df["pm_flag"] = df["pm2_5"] > df["pm10"]
    counts["pm25_gt_pm10"] = int(df["pm_flag"].sum())

    missing_before = int(df[AIR_VARS].isna().sum().sum())
    for col in AIR_VARS:
        df[col] = fill_short_gaps(df[col])
    counts["filled"] = missing_before - int(df[AIR_VARS].isna().sum().sum())
    counts["missing_pm25"] = int(df["pm2_5"].isna().sum())
    return df, counts


def load_city(city, raw_dir):
    raw_dir = Path(raw_dir)
    air = read_raw(raw_dir / f"{city}_air.csv.gz", AIR_VARS)
    weather = read_raw(raw_dir / f"{city}_weather.csv.gz", WEATHER_VARS)

    df = air.join(weather, how="outer")
    hours = pd.date_range(df.index.min(), df.index.max(), freq="h", name="time")
    df = df.reindex(hours)  # an hour the API skipped becomes an explicit gap

    df, counts = clean_hourly(df)
    df["city"] = city
    return df, counts


def load_all(raw_dir, cities=tuple(CITIES)):
    """All cities in one long hourly table (with a `city` column), plus the
    cleaning counts per city."""
    frames, counts = [], {}
    for city in cities:
        df, counts[city] = load_city(city, raw_dir)
        frames.append(df)
    return pd.concat(frames), counts


def circular_mean(degrees) -> float:
    """Mean wind direction: average the unit vectors, not the angles.
    The plain mean of 350 and 10 is 180 (south); the vector mean is 0 (north)."""
    rad = np.deg2rad(np.asarray(degrees, dtype=float))
    rad = rad[~np.isnan(rad)]
    if rad.size == 0:
        return np.nan
    return float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)


def daily(hourly, min_hours: int = MIN_HOURS):
    """Daily values per city. A day counts only with >= min_hours of PM2.5,
    and any other daily value with fewer valid hours than that is left empty."""
    out = []
    for city, df in hourly.groupby("city", sort=False):
        by_day = df.resample("D")
        d = by_day[MEAN_COLS].mean()
        d["precipitation"] = by_day["precipitation"].sum(min_count=1)
        d["wind_direction_10m"] = by_day["wind_direction_10m"].apply(circular_mean)
        cols = AIR_VARS + WEATHER_VARS
        d[cols] = d[cols].where(by_day[cols].count() >= min_hours)
        d = d[by_day["pm2_5"].count() >= min_hours].copy()
        d["season"] = d.index.month.map(SEASON_OF_MONTH)
        d["city"] = city
        out.append(d)
    return pd.concat(out)
