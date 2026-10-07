"""Run the whole analysis once and bundle the results for the charts and the
printed summary."""
from dataclasses import dataclass

import pandas as pd

from airq import analyze as an
from airq.cities import WEATHER_VARS, display_name
from airq.clean import SEASONS, daily, load_all

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass
class Results:
    cities: list
    hourly: pd.DataFrame
    daily: pd.DataFrame
    counts: dict
    monthly: pd.DataFrame
    hour_month: dict
    exceedance: pd.DataFrame
    wind_speed: pd.DataFrame
    blh: pd.DataFrame
    rain: pd.DataFrame
    corr: dict
    sectors: pd.DataFrame
    sea_land: pd.DataFrame
    regressions: dict
    diwali: pd.DataFrame
    diwali_ratio: pd.DataFrame


def run(raw_dir):
    hourly, counts = load_all(raw_dir)
    return analyze_all(hourly, counts)


def analyze_all(hourly, counts):
    days = daily(hourly)
    cities = list(dict.fromkeys(hourly["city"]))
    windows = an.diwali_windows(days)
    return Results(
        cities=cities,
        hourly=hourly,
        daily=days,
        counts=counts,
        monthly=an.monthly_profile(hourly),
        hour_month={c: an.hour_month(hourly, c) for c in cities},
        exceedance=an.exceedance(days),
        wind_speed=an.by_wind_speed(hourly),
        blh=an.by_blh_decile(hourly),
        rain=an.rain_vs_dry(hourly),
        corr={c: an.spearman(hourly, c) for c in cities},
        sectors=an.by_wind_sector(hourly),
        sea_land=an.sea_vs_land(hourly),
        regressions={c: an.regression(days[days["city"] == c]) for c in cities},
        diwali=windows,
        diwali_ratio=an.diwali_ratio(windows),
    )


def summary(res: Results) -> str:
    """The numbers the README is written from."""
    out = []

    def section(title):
        out.append(f"\n== {title} ==")

    section("Data")
    for city in res.cities:
        h = res.hourly[res.hourly["city"] == city]
        c = res.counts[city]
        out.append(
            f"{display_name(city)}: {h.index.min():%Y-%m-%d} to {h.index.max():%Y-%m-%d}, "
            f"{len(h)} hours, {(res.daily['city'] == city).sum()} valid days | "
            f"model outage hours {c['model_outage_hours']}, "
            f"negatives {c['negatives']}, PM2.5>PM10 {c['pm25_gt_pm10']}, "
            f"filled {c['filled']}, still missing PM2.5 {c['missing_pm25']}"
        )

    for line in _weather_gaps(res.daily):
        out.append(line)

    section("Q1 Mean PM2.5 by month (µg/m³)")
    monthly = res.monthly.rename(index=lambda m: MONTHS[m - 1], columns=display_name).T
    out.append(monthly.round(0).to_string())

    section("Q1 Mean daily PM2.5 by season (µg/m³)")
    seasonal = (res.daily.groupby(["city", "season"])["pm2_5"].mean().unstack("season")
                .reindex(columns=SEASONS).reindex(res.cities).rename(index=display_name))
    seasonal["winter / monsoon"] = seasonal["winter"] / seasonal["monsoon"]
    out.append(seasonal.round(1).to_string())

    section("Q1 Days over the 24-hour limits (% of days)")
    exc = res.exceedance.assign(city=res.exceedance["city"].map(display_name))
    out.append(exc[["city", "season", "days", "pct_naaqs", "pct_who"]]
               .rename(columns={"pct_naaqs": "% over 60 (India)", "pct_who": "% over 15 (WHO)"})
               .round(1).to_string(index=False))

    section("Q2 Median PM2.5 by wind speed (km/h)")
    out.append(_by_city(res.wind_speed, "band", "median_pm25", an.WIND_LABELS).to_string())

    section("Q2 Median PM2.5 by boundary layer height decile (1 = lowest)")
    for city in res.cities:
        b = res.blh[res.blh["city"] == city].set_index("decile")
        out.append(f"{display_name(city)}: decile 1 ({b.loc[1, 'blh_median']:.0f} m) "
                   f"{b.loc[1, 'median_pm25']:.0f} µg/m³ vs decile 10 "
                   f"({b.loc[10, 'blh_median']:.0f} m) {b.loc[10, 'median_pm25']:.0f} µg/m³")

    section("Q2 Median PM2.5, rain vs dry hours")
    out.append(_by_city(res.rain, "condition", "median_pm25").to_string())

    section("Q2 Spearman correlation with PM2.5")
    corr = pd.DataFrame({display_name(c): m["pm2_5"].drop("pm2_5") for c, m in res.corr.items()})
    out.append(corr.round(2).to_string())

    section("Q3 Median PM2.5 by wind direction (where the wind comes from)")
    out.append(_by_city(res.sectors, "sector", "median_pm25", an.SECTORS).to_string())
    out.append("hours per sector:")
    out.append(_by_city(res.sectors, "sector", "hours", an.SECTORS).to_string())
    out.append("Median PM2.5 within each season, wind from SW/W vs N/NE/E:")
    sl = res.sea_land.assign(city=res.sea_land["city"].map(display_name))
    out.append(sl.rename(columns={"sea": "SW/W", "land": "N/NE/E", "sea_hours": "SW/W hours",
                                  "land_hours": "N/NE/E hours"}).round(0).to_string(index=False))

    section("Q4 Regression of daily log(PM2.5) on weather")
    for city, r in res.regressions.items():
        out.append(f"{display_name(city)}: R² train {r.r2_train:.2f} ({r.n_train} days), "
                   f"test {r.r2_test:.2f} ({r.n_test} days); winter minus monsoon "
                   f"actual {r.winter_gap_actual:.0f} µg/m³, weather-predicted {r.winter_gap_predicted:.0f} µg/m³ "
                   f"(all days); test period only: actual {r.winter_gap_test_actual:.0f}, "
                   f"predicted {r.winter_gap_test_predicted:.0f}")
    effects = pd.DataFrame({display_name(c): r.effect_pct for c, r in res.regressions.items()})
    out.append("% change in PM2.5 per 1 standard deviation rise:")
    out.append(effects.round(1).to_string())

    section("Q5 Diwali day vs median of days -7..-3")
    dr = res.diwali_ratio.assign(city=res.diwali_ratio["city"].map(display_name))
    out.append(dr.round(2).to_string(index=False))

    return "\n".join(out).lstrip("\n") + "\n"


def _by_city(df, row, value, order=None):
    t = df.pivot(index=row, columns="city", values=value)
    if order is not None:
        t = t.reindex([o for o in order if o in t.index])
    return t[list(df["city"].unique())].rename(columns=display_name).round(0)


def _weather_gaps(daily):
    """One line per run of consecutive days with a weather value missing."""
    lines = []
    weather = [c for c in WEATHER_VARS if c in daily.columns]
    for city, d in daily.groupby("city", sort=False):
        for col in weather:
            missing = d.index[d[col].isna()]
            if missing.empty:
                continue
            run = (missing.to_series().diff() != pd.Timedelta("1D")).cumsum()
            for _, days in missing.to_series().groupby(run):
                lines.append(f"  {display_name(city)}: {col} missing {days.iloc[0]:%Y-%m-%d} "
                             f"to {days.iloc[-1]:%Y-%m-%d} ({len(days)} days)")
    return lines

