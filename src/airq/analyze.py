"""All the numbers. Pure functions: DataFrames in, small DataFrames out.

`hourly` is the long table from clean.load_all (DatetimeIndex, `city` column).
`daily` is clean.daily(hourly).
"""
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from airq.cities import CITIES, DIWALI
from airq.clean import SEASON_OF_MONTH, SEASONS

NAAQS_24H = 60.0  # India, PM2.5 24-hour standard, µg/m³
WHO_24H = 15.0  # WHO 2021 guideline, PM2.5 24-hour, µg/m³

WIND_BANDS = [0, 5, 10, 15, 20, np.inf]  # km/h
WIND_LABELS = ["0-5", "5-10", "10-15", "15-20", "20+"]
RAIN_THRESHOLD = 0.1  # mm in an hour
SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
SEA_SECTORS = ("SW", "W")  # the sea side, at least for Surat
LAND_SECTORS = ("N", "NE", "E")
CORR_COLS = [
    "pm2_5", "temperature_2m", "relative_humidity_2m", "precipitation",
    "wind_speed_10m", "boundary_layer_height",
]

SPLIT = date(2025, 8, 1)  # train before, test from here on
MIN_TRAIN_DAYS = 30
FEATURES = ["log_blh", "wind_speed", "log_rain", "humidity", "temperature", "wind_sin", "wind_cos"]


def _city_order(cities):
    present = set(cities)
    return [c for c in CITIES if c in present] + sorted(present - set(CITIES))


# --- Q1: when is the air bad? -----------------------------------------------

def monthly_profile(hourly):
    """Mean PM2.5 per calendar month (rows) and city (columns)."""
    out = hourly.groupby([hourly.index.month, "city"])["pm2_5"].mean().unstack("city")
    out.index.name = "month"
    return out[_city_order(out.columns)]


def hour_month(hourly, city):
    """Mean PM2.5 by hour of day (rows) and month (columns) for one city."""
    df = hourly[hourly["city"] == city]
    out = df.pivot_table(values="pm2_5", index=df.index.hour, columns=df.index.month, aggfunc="mean")
    out.index.name, out.columns.name = "hour", "month"
    return out


def exceedance(daily):
    """Days over the India and WHO 24-hour limits, per city and season."""
    d = daily.dropna(subset=["pm2_5"]).assign(
        over_naaqs=lambda x: x["pm2_5"] > NAAQS_24H,
        over_who=lambda x: x["pm2_5"] > WHO_24H,
    )
    out = d.groupby(["city", "season"]).agg(
        days=("pm2_5", "size"),
        over_naaqs=("over_naaqs", "sum"),
        over_who=("over_who", "sum"),
    ).reset_index()
    out["pct_naaqs"] = 100 * out["over_naaqs"] / out["days"]
    out["pct_who"] = 100 * out["over_who"] / out["days"]
    return _sort_city_season(out)


def _sort_city_season(df):
    order = {c: i for i, c in enumerate(_city_order(df["city"].unique()))}
    return (df.assign(_c=df["city"].map(order), _s=df["season"].map(SEASONS.index))
              .sort_values(["_c", "_s"]).drop(columns=["_c", "_s"]).reset_index(drop=True))


# --- Q2: why winter? -----------------------------------------------------------

def by_wind_speed(hourly):
    band = pd.cut(hourly["wind_speed_10m"], WIND_BANDS, labels=WIND_LABELS, right=False)
    out = (hourly.assign(band=band)
                 .groupby(["city", "band"], observed=True)["pm2_5"]
                 .agg(median_pm25="median", hours="count")
                 .reset_index())
    out["band"] = out["band"].astype(str)
    return out


def by_blh_decile(hourly):
    """Split hours into ten equal-sized groups by boundary layer height."""
    parts = []
    for city, df in hourly.groupby("city", sort=False):
        df = df.dropna(subset=["boundary_layer_height", "pm2_5"])
        decile = pd.qcut(df["boundary_layer_height"], 10, labels=list(range(1, 11)))
        g = df.groupby(decile, observed=True)
        part = pd.DataFrame({
            "blh_median": g["boundary_layer_height"].median(),
            "median_pm25": g["pm2_5"].median(),
            "hours": g.size(),
        })
        part.index = part.index.astype(int)
        part.index.name = "decile"
        parts.append(part.reset_index().assign(city=city))
    return pd.concat(parts, ignore_index=True)


def rain_vs_dry(hourly, threshold=RAIN_THRESHOLD):
    df = hourly.dropna(subset=["precipitation", "pm2_5"])
    condition = np.where(df["precipitation"] > threshold, "rain", "dry")
    return (df.assign(condition=condition)
              .groupby(["city", "condition"])["pm2_5"]
              .agg(median_pm25="median", hours="count")
              .reset_index())


def spearman(hourly, city):
    """Rank correlation between PM2.5 and the weather. Wind direction is left
    out: it is an angle, so ranking it makes no sense."""
    return hourly.loc[hourly["city"] == city, CORR_COLS].corr(method="spearman")


# --- Q3: does the coast help? ------------------------------------------------

def sector(deg):
    """Compass sector index 0-7 (N, NE, ... NW). Each sector is 45° wide and
    centred on its direction, so N covers 337.5° up to 22.5°."""
    return ((np.asarray(deg, dtype=float) + 22.5) // 45) % 8


def by_wind_sector(hourly):
    df = hourly.assign(sector=sector(hourly["wind_direction_10m"])).dropna(subset=["sector", "pm2_5"])
    df["sector"] = df["sector"].astype(int)
    parts = []
    for city in _city_order(df["city"].unique()):
        g = df[df["city"] == city].groupby("sector")["pm2_5"]
        parts.append(pd.DataFrame({
            "city": city,
            "sector": SECTORS,
            "median_pm25": g.median().reindex(range(8)).to_numpy(),
            "hours": g.count().reindex(range(8), fill_value=0).to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True)


def sea_vs_land(hourly):
    """Median PM2.5 with wind from the SW/W vs from the N/NE/E, within each
    season, so the season itself isn't mistaken for a wind effect."""
    s = sector(hourly["wind_direction_10m"])
    label = pd.Series(np.array(SECTORS + ["?"], dtype=object)[np.where(np.isnan(s), 8, s).astype(int)],
                      index=hourly.index)
    df = hourly.assign(label=label, season=hourly.index.month.map(SEASON_OF_MONTH)).dropna(subset=["pm2_5"])
    rows = []
    for (city, season), g in df.groupby(["city", "season"]):
        sea = g.loc[g["label"].isin(SEA_SECTORS), "pm2_5"]
        land = g.loc[g["label"].isin(LAND_SECTORS), "pm2_5"]
        rows.append({"city": city, "season": season, "sea": sea.median(), "land": land.median(),
                     "sea_hours": len(sea), "land_hours": len(land)})
    return _sort_city_season(pd.DataFrame(rows))


# --- Q4: regression ----------------------------------------------------------

def design_matrix(daily_city):
    """One row per day: the weather features and log(PM2.5). Days with any
    missing value are dropped."""
    d = daily_city
    rad = np.deg2rad(d["wind_direction_10m"])
    out = pd.DataFrame({
        "log_blh": np.log(d["boundary_layer_height"].where(d["boundary_layer_height"] > 0)),
        "wind_speed": d["wind_speed_10m"],
        "log_rain": np.log1p(d["precipitation"]),
        "humidity": d["relative_humidity_2m"],
        "temperature": d["temperature_2m"],
        # wind direction as a point on a circle, so 359° and 1° end up close together
        "wind_sin": np.sin(rad),
        "wind_cos": np.cos(rad),
        "log_pm25": np.log(d["pm2_5"].where(d["pm2_5"] > 0)),
        "season": d["season"],
    }, index=d.index)
    return out.dropna()


def fit_ols(X, y):
    """Ordinary least squares with an intercept. Returns [intercept, b1, b2, ...]."""
    A = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
    coef, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)
    return coef


def _predict(coef, X):
    # numpy on macOS's Accelerate BLAS raises false overflow/divide warnings
    # inside matmul even when the result is correct, so silence them and
    # check the result ourselves instead.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        out = coef[0] + X @ coef[1:]
    if not np.isfinite(out).all():
        raise FloatingPointError("regression produced non-finite predictions")
    return out


def r_squared(y, y_hat):
    y, y_hat = np.asarray(y, dtype=float), np.asarray(y_hat, dtype=float)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


@dataclass
class RegressionResult:
    city: str
    coefficients: pd.Series  # change in log(PM2.5) per 1 standard deviation of each feature
    intercept: float
    r2_train: float
    r2_test: float
    n_train: int
    n_test: int
    predictions: pd.DataFrame  # per day: actual, predicted (µg/m³), split
    winter_gap_actual: float  # typical winter minus typical monsoon PM2.5, µg/m³, all days
    winter_gap_predicted: float
    winter_gap_test_actual: float  # the same, test period only
    winter_gap_test_predicted: float

    @property
    def effect_pct(self):
        """Percent change in PM2.5 for a 1-standard-deviation rise in each feature."""
        return (np.exp(self.coefficients) - 1) * 100


def regression(daily_city, split=SPLIT, min_train=MIN_TRAIN_DAYS):
    dm = design_matrix(daily_city)
    is_train = dm.index < pd.Timestamp(split)
    train = dm[is_train]
    if len(train) < min_train:
        raise ValueError(f"need at least {min_train} training days before {split}, got {len(train)}")

    # z-scores put every coefficient in "per standard deviation" units so they
    # can be compared. With an intercept this doesn't change the predictions.
    mean, std = train[FEATURES].mean(), train[FEATURES].std(ddof=0)
    flat = list(std[std == 0].index)
    if flat:
        raise ValueError(f"no variation in training data for: {', '.join(flat)}")
    z = ((dm[FEATURES] - mean) / std).to_numpy()

    coef = fit_ols(z[is_train], train["log_pm25"])
    pred_log = _predict(coef, z)
    y = dm["log_pm25"].to_numpy()

    n_test = int((~is_train).sum())
    r2_test = r_squared(y[~is_train], pred_log[~is_train]) if n_test >= 2 else float("nan")

    city = daily_city["city"].iloc[0] if "city" in daily_city else ""
    predictions = pd.DataFrame({
        "actual": np.exp(y),
        "predicted": np.exp(pred_log),
        "split": np.where(is_train, "train", "test"),
    }, index=dm.index)

    return RegressionResult(
        city=city,
        coefficients=pd.Series(coef[1:], index=FEATURES),
        intercept=float(coef[0]),
        r2_train=r_squared(y[is_train], pred_log[is_train]),
        r2_test=r2_test,
        n_train=int(is_train.sum()),
        n_test=n_test,
        predictions=predictions,
        winter_gap_actual=winter_gap(y, dm["season"]),
        winter_gap_predicted=winter_gap(pred_log, dm["season"]),
        winter_gap_test_actual=winter_gap(y[~is_train], dm["season"][~is_train]),
        winter_gap_test_predicted=winter_gap(pred_log[~is_train], dm["season"][~is_train]),
    )


def winter_gap(log_values, seasons):
    """Typical (geometric mean) winter PM2.5 minus typical monsoon PM2.5.
    Geometric means because the model works in log(PM2.5)."""
    typical = np.exp(pd.Series(log_values, index=seasons.index).groupby(seasons).mean())
    return float(typical.get("winter", np.nan) - typical.get("monsoon", np.nan))


# --- Q5: Diwali --------------------------------------------------------------

def diwali_windows(daily, dates=DIWALI, days=7):
    """Daily PM2.5 for `days` either side of each Diwali inside the data."""
    rows = []
    for city, d in daily.groupby("city", sort=False):
        pm = d["pm2_5"]
        for year, day0 in dates.items():
            day0 = pd.Timestamp(day0)
            if not pm.index.min() <= day0 <= pm.index.max():
                continue
            for offset in range(-days, days + 1):
                day = day0 + pd.Timedelta(days=offset)
                rows.append({"city": city, "year": year, "offset": offset,
                             "date": day, "pm2_5": pm.get(day, np.nan)})
    return pd.DataFrame(rows, columns=["city", "year", "offset", "date", "pm2_5"])


def diwali_ratio(windows, baseline=(-7, -3)):
    """Diwali-day PM2.5 divided by the median of the days before it."""
    rows = []
    for (city, year), w in windows.groupby(["city", "year"], sort=False):
        w = w.set_index("offset")["pm2_5"]
        base = w.loc[baseline[0]:baseline[1]].median()
        on_day = w.get(0, np.nan)
        rows.append({"city": city, "year": year, "diwali": on_day,
                     "baseline": base, "ratio": on_day / base})
    return pd.DataFrame(rows, columns=["city", "year", "diwali", "baseline", "ratio"])
