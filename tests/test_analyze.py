from datetime import date

import numpy as np
import pandas as pd
import pytest

from airq.analyze import (
    FEATURES,
    SECTORS,
    by_blh_decile,
    by_wind_sector,
    by_wind_speed,
    diwali_ratio,
    diwali_windows,
    exceedance,
    fit_ols,
    hour_month,
    monthly_profile,
    r_squared,
    rain_vs_dry,
    regression,
    sector,
    spearman,
)
from airq.clean import SEASON_OF_MONTH


def hourly(values, start="2024-01-01", city="surat", **cols):
    idx = pd.date_range(start, periods=len(values), freq="h", name="time")
    df = pd.DataFrame({"pm2_5": values}, index=idx)
    defaults = {
        "pm10": 100.0, "nitrogen_dioxide": 10.0, "temperature_2m": 25.0,
        "relative_humidity_2m": 60.0, "precipitation": 0.0, "wind_speed_10m": 8.0,
        "wind_direction_10m": 90.0, "boundary_layer_height": 500.0,
    }
    for k, v in {**defaults, **cols}.items():
        df[k] = v
    df["city"] = city
    return df


def daily_frame(pm25, start="2024-01-01", city="surat"):
    idx = pd.date_range(start, periods=len(pm25), freq="D", name="time")
    df = pd.DataFrame({"pm2_5": pm25}, index=idx)
    df["season"] = df.index.month.map(SEASON_OF_MONTH)
    df["city"] = city
    return df


# --- Q1 ----------------------------------------------------------------------

def test_monthly_profile_is_month_by_city():
    jan = hourly([10.0, 20.0], start="2024-01-01")
    feb = hourly([40.0], start="2024-02-01")
    ahd = hourly([5.0], start="2024-01-01", city="ahmedabad")

    out = monthly_profile(pd.concat([jan, feb, ahd]))

    assert out.loc[1, "surat"] == 15.0
    assert out.loc[2, "surat"] == 40.0
    assert out.loc[1, "ahmedabad"] == 5.0
    assert list(out.columns) == ["surat", "ahmedabad"]


def test_hour_month_has_one_row_per_hour():
    df = hourly(list(range(48)), start="2024-01-01")

    out = hour_month(df, "surat")

    assert out.shape == (24, 1)
    assert out.loc[0, 1] == 12.0  # mean of hour 0 on day 1 (0) and day 2 (24)


def test_exceedance_counts_days_over_each_limit():
    d = daily_frame([70.0, 50.0, 10.0, 61.0, 60.0])  # all January

    out = exceedance(d).set_index(["city", "season"])

    row = out.loc[("surat", "winter")]
    assert row["days"] == 5
    assert row["over_naaqs"] == 2  # 70 and 61; exactly 60 is not over
    assert row["over_who"] == 4
    assert row["pct_naaqs"] == pytest.approx(40.0)


# --- Q2 ----------------------------------------------------------------------

def test_wind_speed_bands_take_median_pm25():
    df = hourly([80.0, 60.0, 20.0, 30.0], wind_speed_10m=[2.0, 4.0, 12.0, 25.0])

    out = by_wind_speed(df).set_index("band")

    assert out.loc["0-5", "median_pm25"] == 70.0
    assert out.loc["0-5", "hours"] == 2
    assert out.loc["10-15", "median_pm25"] == 20.0
    assert out.loc["20+", "median_pm25"] == 30.0


def test_blh_deciles_split_hours_into_ten_groups():
    blh = np.arange(100.0, 1100.0, 10.0)  # 100 hours
    df = hourly(list(2000.0 / blh), boundary_layer_height=blh)

    out = by_blh_decile(df)

    assert out["decile"].tolist() == list(range(1, 11))
    assert out["median_pm25"].is_monotonic_decreasing
    assert out["hours"].sum() == 100


def test_rain_hours_compared_with_dry_hours():
    df = hourly([20.0, 30.0, 60.0, 80.0], precipitation=[1.0, 0.5, 0.0, 0.05])

    out = rain_vs_dry(df).set_index("condition")

    assert out.loc["rain", "median_pm25"] == 25.0
    assert out.loc["dry", "median_pm25"] == 70.0


def test_spearman_sees_a_perfect_inverse_rank_relationship():
    blh = np.linspace(100, 1000, 50)
    df = hourly(list(1e4 / blh), boundary_layer_height=blh,
                temperature_2m=np.linspace(20, 30, 50))

    corr = spearman(df, "surat")

    assert corr.loc["pm2_5", "boundary_layer_height"] == pytest.approx(-1.0)
    assert "wind_direction_10m" not in corr.columns


# --- Q3 ----------------------------------------------------------------------

@pytest.mark.parametrize("deg,label", [
    (0.0, "N"), (359.0, "N"), (22.4, "N"), (22.5, "NE"),
    (90.0, "E"), (225.0, "SW"), (337.4, "NW"), (337.5, "N"),
])
def test_wind_sector_boundaries(deg, label):
    assert SECTORS[int(sector(deg))] == label


def test_missing_wind_direction_has_no_sector():
    assert np.isnan(sector(np.nan))


def test_wind_sectors_include_all_eight_directions():
    df = hourly([10.0, 50.0, 70.0], wind_direction_10m=[0.0, 270.0, 270.0])

    out = by_wind_sector(df)
    surat = out[out["city"] == "surat"].set_index("sector")

    assert list(surat.index) == SECTORS
    assert surat.loc["N", "median_pm25"] == 10.0
    assert surat.loc["W", "median_pm25"] == 60.0
    assert surat.loc["E", "hours"] == 0


# --- Q4 ----------------------------------------------------------------------

def test_least_squares_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    y = 2 + 3 * X[:, 0] - X[:, 1] + rng.normal(scale=0.01, size=500)

    coef = fit_ols(X, y)

    assert coef == pytest.approx([2.0, 3.0, -1.0], abs=0.01)
    assert r_squared(y, coef[0] + X @ coef[1:]) > 0.999


def weather_driven_daily(n_days=730, start="2024-01-01", noise=0.05, seed=1):
    """Daily data where log(PM2.5) really is driven by the weather."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_days, freq="D", name="time")
    blh = rng.uniform(200, 1500, n_days)
    wind = rng.uniform(2, 25, n_days)
    rain = rng.exponential(1.0, n_days) * (rng.random(n_days) < 0.3)
    rh = rng.uniform(30, 95, n_days)
    temp = rng.uniform(15, 38, n_days)
    wdir = rng.uniform(0, 360, n_days)
    log_pm = (6.0 - 0.6 * np.log(blh) - 0.03 * wind - 0.3 * np.log1p(rain)
              + 0.3 * np.cos(np.deg2rad(wdir)) + rng.normal(0, noise, n_days))
    df = pd.DataFrame({
        "pm2_5": np.exp(log_pm), "boundary_layer_height": blh, "wind_speed_10m": wind,
        "precipitation": rain, "relative_humidity_2m": rh, "temperature_2m": temp,
        "wind_direction_10m": wdir,
    }, index=idx)
    df["season"] = df.index.month.map(SEASON_OF_MONTH)
    df["city"] = "surat"
    return df


def test_regression_explains_weather_driven_data():
    res = regression(weather_driven_daily())

    assert res.r2_train > 0.9
    assert res.r2_test > 0.9
    assert list(res.coefficients.index) == FEATURES
    assert res.coefficients["log_blh"] < 0
    assert res.n_train + res.n_test == 730
    assert set(res.predictions["split"]) == {"train", "test"}


def test_regression_drops_days_with_missing_weather():
    df = weather_driven_daily()
    df.iloc[-5:, df.columns.get_loc("boundary_layer_height")] = np.nan  # archive lag

    res = regression(df)

    assert res.n_train + res.n_test == 725


def test_regression_needs_thirty_training_days():
    df = weather_driven_daily(n_days=29, start="2025-06-01")

    with pytest.raises(ValueError, match="30"):
        regression(df, split=date(2025, 8, 1))


def test_winter_gap_is_reported_in_ugm3():
    res = regression(weather_driven_daily())

    assert np.isfinite(res.winter_gap_actual)
    assert np.isfinite(res.winter_gap_predicted)


# --- Q5 ----------------------------------------------------------------------

def test_diwali_window_and_ratio():
    pm = [100.0] * 40
    d = daily_frame(pm, start="2024-10-15")
    d.loc["2024-10-24":"2024-10-28", "pm2_5"] = [10.0, 20.0, 30.0, 40.0, 50.0]  # days -7..-3
    d.loc["2024-10-31", "pm2_5"] = 90.0  # Diwali 2024

    windows = diwali_windows(d)
    ratio = diwali_ratio(windows).set_index(["city", "year"])

    assert windows["offset"].min() == -7 and windows["offset"].max() == 7
    assert set(windows["year"]) == {2024}  # other Diwalis are outside the data
    row = ratio.loc[("surat", 2024)]
    assert row["diwali"] == 90.0
    assert row["baseline"] == 30.0
    assert row["ratio"] == pytest.approx(3.0)


def test_regression_runs_without_numpy_warnings():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = regression(weather_driven_daily())

    assert np.isfinite(res.predictions["predicted"]).all()


# --- review fixes --------------------------------------------------------------

def test_winter_gap_uses_geometric_means():
    from airq.analyze import winter_gap

    seasons = pd.Series(["winter", "winter", "monsoon", "monsoon"])
    logs = np.log([40.0, 90.0, 10.0, 40.0])  # geometric means 60 and 20

    assert winter_gap(logs, seasons) == pytest.approx(40.0)


def test_regression_reports_the_gap_on_the_test_period_alone():
    from airq.analyze import winter_gap

    # test period Sep 2024 - Sep 2025 holds a winter and a monsoon
    res = regression(weather_driven_daily(n_days=1000, start="2023-01-01"), split=date(2024, 9, 1))

    p = res.predictions[res.predictions["split"] == "test"]
    assert np.isfinite(res.winter_gap_test_actual)
    seasons = p.index.month.map(SEASON_OF_MONTH).to_series(index=p.index)
    assert res.winter_gap_test_actual == pytest.approx(winter_gap(np.log(p["actual"]), seasons))
    assert res.winter_gap_test_predicted == pytest.approx(winter_gap(np.log(p["predicted"]), seasons))


def test_sea_vs_land_wind_by_season():
    from airq.analyze import sea_vs_land

    df = pd.concat([
        hourly([20.0, 30.0], start="2024-07-01", wind_direction_10m=[225.0, 270.0]),  # monsoon, sea
        hourly([60.0, 80.0], start="2024-07-02", wind_direction_10m=[0.0, 90.0]),  # monsoon, land
        hourly([50.0], start="2024-01-01", wind_direction_10m=[180.0]),  # winter, south: neither
    ])

    out = sea_vs_land(df).set_index(["city", "season"])

    row = out.loc[("surat", "monsoon")]
    assert row["sea"] == 25.0 and row["land"] == 70.0
    assert row["sea_hours"] == 2 and row["land_hours"] == 2
    assert out.loc[("surat", "winter"), "sea_hours"] == 0
