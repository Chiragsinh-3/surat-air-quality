import numpy as np
import pandas as pd
import pytest

from airq.cities import AIR_VARS, WEATHER_VARS
from airq.clean import (
    MissingDataError,
    circular_mean,
    clean_hourly,
    daily,
    fill_short_gaps,
    load_city,
    season,
)
from airq.fetch import write_csv_gz


def hourly_frame(n, start="2024-01-01", pm25=40.0, pm10=60.0, city="surat"):
    """A clean hourly table shaped like load_city output."""
    idx = pd.date_range(start, periods=n, freq="h", name="time")
    df = pd.DataFrame(index=idx)
    df["pm2_5"] = pm25
    df["pm10"] = pm10
    df["nitrogen_dioxide"] = 10.0
    df["temperature_2m"] = 25.0
    df["relative_humidity_2m"] = 60.0
    df["precipitation"] = 0.0
    df["wind_speed_10m"] = 8.0
    df["wind_direction_10m"] = 90.0
    df["boundary_layer_height"] = 500.0
    df["city"] = city
    return df


def angle_gap(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


# --- gap filling -------------------------------------------------------------

def test_gap_of_three_hours_is_filled_linearly():
    s = pd.Series([10.0, np.nan, np.nan, np.nan, 50.0])
    assert fill_short_gaps(s).tolist() == [10.0, 20.0, 30.0, 40.0, 50.0]


def test_gap_of_four_hours_is_left_alone():
    s = pd.Series([10.0, np.nan, np.nan, np.nan, np.nan, 60.0])
    out = fill_short_gaps(s)
    assert out.isna().sum() == 4


def test_leading_missing_hours_stay_missing():
    s = pd.Series([np.nan, np.nan, 30.0, 40.0])
    out = fill_short_gaps(s)
    assert out.isna().tolist() == [True, True, False, False]


# --- sanity checks -----------------------------------------------------------

def test_negative_concentrations_become_missing_and_are_counted():
    df = hourly_frame(3)
    df.loc[df.index[1], "pm2_5"] = -5.0
    df.loc[df.index[1], "pm10"] = -1.0

    out, counts = clean_hourly(df)

    assert counts["negatives"] == 2
    # a single missing hour sits between two valid ones, so it is refilled
    assert out["pm2_5"].tolist() == [40.0, 40.0, 40.0]
    assert counts["filled"] == 2


def test_pm25_above_pm10_is_flagged():
    df = hourly_frame(3)
    df.loc[df.index[2], "pm2_5"] = 80.0

    out, counts = clean_hourly(df)

    assert counts["pm25_gt_pm10"] == 1
    assert out["pm_flag"].tolist() == [False, False, True]


# --- loading -----------------------------------------------------------------

def write_raw(raw_dir, city, air, weather):
    write_csv_gz(air, raw_dir / f"{city}_air.csv.gz")
    write_csv_gz(weather, raw_dir / f"{city}_weather.csv.gz")


def test_load_city_joins_on_the_hour_and_fills_missing_hours(tmp_path):
    times = pd.date_range("2024-01-01", periods=4, freq="h").strftime("%Y-%m-%dT%H:%M")
    air = pd.DataFrame({"time": times, "pm2_5": [1.0, 2.0, 3.0, 4.0],
                        "pm10": [9.0] * 4, "nitrogen_dioxide": [5.0] * 4})
    air = air.drop(index=2)  # API skipped an hour
    weather = pd.DataFrame({"time": times, **{v: [float(i) for i in range(4)] for v in WEATHER_VARS}})
    write_raw(tmp_path, "surat", air, weather)

    df, counts = load_city("surat", tmp_path)

    assert len(df) == 4
    assert (df.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()
    assert df["pm2_5"].tolist() == [1.0, 2.0, 3.0, 4.0]  # missing hour filled
    assert df.loc["2024-01-01 03:00", "temperature_2m"] == 3.0
    assert (df["city"] == "surat").all()


def test_missing_raw_file_says_to_run_fetch(tmp_path):
    with pytest.raises(MissingDataError, match="python -m airq fetch"):
        load_city("surat", tmp_path)


def test_missing_column_is_named(tmp_path):
    times = ["2024-01-01T00:00"]
    air = pd.DataFrame({"time": times, "pm2_5": [1.0], "pm10": [2.0]})
    weather = pd.DataFrame({"time": times, **{v: [1.0] for v in WEATHER_VARS}})
    write_raw(tmp_path, "surat", air, weather)

    with pytest.raises(ValueError, match="nitrogen_dioxide"):
        load_city("surat", tmp_path)


# --- daily -------------------------------------------------------------------

def test_circular_mean_of_350_and_10_is_north():
    assert angle_gap(circular_mean([350.0, 10.0]), 0.0) < 1e-9


def test_circular_mean_ignores_missing_values():
    assert angle_gap(circular_mean([90.0, np.nan, 90.0]), 90.0) < 1e-9


def test_day_with_fifteen_hours_is_dropped_and_sixteen_kept():
    df = hourly_frame(48)
    df.iloc[15:24, df.columns.get_loc("pm2_5")] = np.nan  # day 1: 15 valid hours
    df.iloc[24 + 16:48, df.columns.get_loc("pm2_5")] = np.nan  # day 2: 16 valid hours

    d = daily(df)

    assert [str(x.date()) for x in d.index] == ["2024-01-02"]


def test_daily_sums_rain_and_averages_the_rest():
    df = hourly_frame(24)
    df["precipitation"] = 0.5
    df["wind_direction_10m"] = [350.0, 10.0] * 12

    d = daily(df)

    assert d["precipitation"].iloc[0] == pytest.approx(12.0)
    assert d["pm2_5"].iloc[0] == pytest.approx(40.0)
    assert angle_gap(d["wind_direction_10m"].iloc[0], 0.0) < 1e-9
    assert d["season"].iloc[0] == "winter"


def test_days_without_weather_keep_air_values():
    df = hourly_frame(48)
    df.iloc[24:, df.columns.get_indexer(WEATHER_VARS)] = np.nan  # archive lag

    d = daily(df)

    assert len(d) == 2
    assert d["boundary_layer_height"].isna().tolist() == [False, True]
    assert d["precipitation"].isna().tolist() == [False, True]


def test_daily_keeps_cities_apart():
    both = pd.concat([hourly_frame(24, pm25=10.0, city="surat"),
                      hourly_frame(24, pm25=30.0, city="ahmedabad")])

    d = daily(both)

    assert d.set_index("city")["pm2_5"].to_dict() == {"surat": 10.0, "ahmedabad": 30.0}


@pytest.mark.parametrize("month,expected", [
    (1, "winter"), (2, "winter"),
    (3, "pre-monsoon"), (4, "pre-monsoon"), (5, "pre-monsoon"),
    (6, "monsoon"), (7, "monsoon"), (8, "monsoon"), (9, "monsoon"),
    (10, "post-monsoon"), (11, "post-monsoon"), (12, "post-monsoon"),
])
def test_imd_seasons(month, expected):
    assert season(month) == expected


# --- review fixes --------------------------------------------------------------

def test_long_run_of_zero_no2_is_masked_as_model_outage():
    df = hourly_frame(12)
    df.iloc[3:9, df.columns.get_loc("nitrogen_dioxide")] = 0.0  # 6 hours
    df.iloc[3:9, df.columns.get_loc("pm2_5")] = 1.0

    out, counts = clean_hourly(df)

    assert counts["model_outage_hours"] == 6
    assert out["pm2_5"].iloc[3:9].isna().all()  # 6-hour gap is too long to fill
    assert out["pm2_5"].iloc[[2, 9]].tolist() == [40.0, 40.0]


def test_short_run_of_zero_no2_is_kept():
    df = hourly_frame(12)
    df.iloc[3:8, df.columns.get_loc("nitrogen_dioxide")] = 0.0  # 5 hours

    out, counts = clean_hourly(df)

    assert counts["model_outage_hours"] == 0
    assert out["pm2_5"].notna().all()


def test_daily_weather_needs_sixteen_hours_too():
    df = hourly_frame(24)
    df.iloc[5:, df.columns.get_loc("boundary_layer_height")] = np.nan  # only 5 hours left
    df.iloc[5:, df.columns.get_loc("precipitation")] = np.nan

    d = daily(df)

    assert np.isnan(d["boundary_layer_height"].iloc[0])
    assert np.isnan(d["precipitation"].iloc[0])
    assert d["temperature_2m"].iloc[0] == 25.0


def test_missing_raw_file_message_shows_the_full_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(MissingDataError) as err:
        load_city("surat", "data/raw")

    assert str((tmp_path / "data/raw/surat_air.csv.gz").resolve()) in str(err.value)
