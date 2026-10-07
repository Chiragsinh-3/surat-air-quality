"""Static charts (matplotlib + seaborn) for the README. No math here: every
function draws something report.Results already holds."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, never open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from airq.analyze import NAAQS_24H, SECTORS, WHO_24H, WIND_LABELS  # noqa: E402
from airq.cities import CITIES, display_name  # noqa: E402
from airq.clean import SEASONS  # noqa: E402
from airq.report import MONTHS  # noqa: E402

# Okabe-Ito blue and vermillion: readable for colour-blind viewers
COLORS = {"surat": "#0072B2", "ahmedabad": "#D55E00"}
HEAT_CMAP = "YlOrRd"
UNIT = "PM2.5 (µg/m³)"

LABELS = {
    "pm2_5": "PM2.5",
    "temperature_2m": "Temperature",
    "relative_humidity_2m": "Humidity",
    "precipitation": "Rain",
    "wind_speed_10m": "Wind speed",
    "boundary_layer_height": "Boundary layer",
}
FEATURE_LABELS = {
    "log_blh": "Boundary layer height (log)",
    "wind_speed": "Wind speed",
    "log_rain": "Rain (log)",
    "humidity": "Humidity",
    "temperature": "Temperature",
    "wind_sin": "Wind from the east",
    "wind_cos": "Wind from the north",
}

CHART_FILES = [
    "monthly_profile.png",
    "hour_month_heatmap.png",
    "exceedance_days.png",
    "weather_drivers.png",
    "correlation.png",
    "wind_sectors.png",
    "regression.png",
    "diwali.png",
]


def _setup():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold"})


def _save(fig, path):
    path = Path(path)
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)
    return path


def _palette(cities):
    present = set(cities)
    ordered = [c for c in CITIES if c in present] + sorted(present - set(CITIES))
    return {display_name(c): COLORS.get(c, "grey") for c in ordered}


def _with_city_names(df):
    return df.assign(City=df["city"].map(display_name))


# --- Q1 ----------------------------------------------------------------------

def monthly_profile(monthly, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axvspan(5.5, 9.5, color="#56B4E9", alpha=0.12, lw=0)
    ax.text(7.5, 0.97, "monsoon", transform=ax.get_xaxis_transform(), ha="center", va="top", color="#3a7ca5")
    for city in monthly.columns:
        ax.plot(monthly.index, monthly[city], marker="o", lw=2, color=COLORS.get(city), label=display_name(city))
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(UNIT)
    ax.set_title("Average PM2.5 by month")
    ax.legend(frameon=False)
    return _save(fig, path)


def hour_month_heatmap(pivots, path):
    n = len(pivots)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6.5), sharey=True, squeeze=False)
    vmin = min(p.min().min() for p in pivots.values())
    vmax = max(p.max().max() for p in pivots.values())
    for i, (ax, (city, p)) in enumerate(zip(axes[0], pivots.items())):
        p = p.rename(columns=lambda m: MONTHS[m - 1])
        sns.heatmap(p, ax=ax, cmap=HEAT_CMAP, vmin=vmin, vmax=vmax, cbar=i == n - 1,
                    cbar_kws={"label": UNIT})
        ax.set_title(display_name(city))
        ax.set_xlabel("")
        ax.set_ylabel("Hour of day (IST)" if i == 0 else "")
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("PM2.5 by hour and month", fontweight="bold")
    return _save(fig, path)


def exceedance_days(exc, path):
    data = _with_city_names(exc)
    seasons = [s for s in SEASONS if s in set(data["season"])]
    palette = _palette(data["city"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    panels = [("pct_naaqs", f"Over India's 24-hour limit ({NAAQS_24H:.0f} µg/m³)"),
              ("pct_who", f"Over WHO's 24-hour guideline ({WHO_24H:.0f} µg/m³)")]
    for ax, (col, title) in zip(axes, panels):
        sns.barplot(data=data, x="season", y=col, hue="City", order=seasons,
                    hue_order=list(palette), palette=palette, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("% of days")
        ax.set_ylim(0, 100)
    axes[0].legend(frameon=False, title="")
    axes[1].get_legend().remove()
    return _save(fig, path)


# --- Q2 ----------------------------------------------------------------------

def weather_drivers(wind, blh, rain, path):
    palette = _palette(wind["city"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    w = _with_city_names(wind)
    sns.barplot(data=w, x="band", y="median_pm25", hue="City", palette=palette,
                order=[b for b in WIND_LABELS if b in set(w["band"])], hue_order=list(palette), ax=axes[0])
    axes[0].set_title("Wind speed")
    axes[0].set_xlabel("km/h")

    for city, d in blh.groupby("city", sort=False):
        axes[1].plot(d["blh_median"], d["median_pm25"], marker="o", lw=2,
                     color=COLORS.get(city), label=display_name(city))
    axes[1].set_title("Boundary layer height")
    axes[1].set_xlabel("metres (median of each tenth of hours)")

    r = _with_city_names(rain)
    sns.barplot(data=r, x="condition", y="median_pm25", hue="City", palette=palette,
                order=["dry", "rain"], hue_order=list(palette), ax=axes[2])
    axes[2].set_title("Rain (more than 0.1 mm in the hour)")
    axes[2].set_xlabel("")

    axes[0].set_ylabel("Median " + UNIT)
    axes[0].legend(frameon=False, title="")
    axes[2].get_legend().remove()
    fig.suptitle("What the weather does to PM2.5", fontweight="bold", y=1.03)
    return _save(fig, path)


def correlation(corr, path):
    n = len(corr)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5), squeeze=False, layout="constrained")
    for ax, (city, m) in zip(axes[0], corr.items()):
        m = m.rename(index=LABELS, columns=LABELS)
        mask = np.triu(np.ones_like(m, dtype=bool), k=1)
        sns.heatmap(m, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1,
                    square=True, cbar=False, ax=ax)
        ax.set_title(display_name(city))
    fig.colorbar(axes[0][0].collections[0], ax=axes[0].tolist(), shrink=0.8, label="Spearman correlation")
    fig.suptitle("Rank (Spearman) correlation, hourly data", fontweight="bold")
    return _save(fig, path)


# --- Q3 ----------------------------------------------------------------------

def wind_sectors(sectors, path, min_share=0.02):
    cities = list(sectors["city"].unique())
    n = len(cities)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.8), subplot_kw={"projection": "polar"},
                             squeeze=False, layout="constrained")
    theta = np.deg2rad(np.arange(8) * 45)
    rmax = sectors["median_pm25"].max() * 1.1
    for ax, city in zip(axes[0], cities):
        d = sectors[sectors["city"] == city].set_index("sector").loc[SECTORS]
        share = d["hours"] / d["hours"].sum()
        colors = [COLORS.get(city) if s >= min_share else "lightgrey" for s in share]
        ax.bar(theta, d["median_pm25"].fillna(0), width=np.deg2rad(40), color=colors, edgecolor="white")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(theta, SECTORS)
        ax.set_ylim(0, rmax)
        ax.set_yticks(np.arange(30, rmax, 10))  # small rings would sit under the bars
        ax.set_rlabel_position(202.5)  # put the scale where the bars are short
        ax.set_title(display_name(city), pad=18)
    fig.suptitle("Median PM2.5 by the direction the wind comes from", fontweight="bold")
    fig.supxlabel(f"Grey: wind blows from there less than {min_share:.0%} of the time",
                  fontsize=9, color="grey")
    return _save(fig, path)


# --- Q4 ----------------------------------------------------------------------

def regression(regressions, path):
    cities = list(regressions)
    fig, axd = plt.subplot_mosaic([["coef"] * len(cities), cities], figsize=(13, 9),
                                  height_ratios=[1, 1], layout="constrained")
    palette = _palette(cities)

    rows = [{"Feature": FEATURE_LABELS[f], "City": display_name(c), "effect": v}
            for c, r in regressions.items() for f, v in r.effect_pct.items()]
    sns.barplot(data=pd.DataFrame(rows), y="Feature", x="effect", hue="City", orient="h",
                palette=palette, hue_order=list(palette), ax=axd["coef"])
    axd["coef"].axvline(0, color="black", lw=0.8)
    axd["coef"].set_xlabel("% change in daily PM2.5 for a typical (1 standard deviation) rise")
    axd["coef"].set_ylabel("")
    axd["coef"].set_title("Which weather matters most")
    axd["coef"].legend(frameon=False, title="")

    for city, r in regressions.items():
        ax = axd[city]
        p = r.predictions.asfreq("D")  # missing days become NaN, so the lines break there
        ax.plot(p.index, p["actual"], color="grey", lw=1, label="actual")
        ax.plot(p.index, p["predicted"], color=COLORS.get(city), lw=1.5, label="predicted from weather")
        test = p[p["split"] == "test"]
        if not test.empty:
            ax.axvspan(test.index.min(), test.index.max(), color="grey", alpha=0.1, lw=0)
        ax.set_title(f"{display_name(city)}: R² {r.r2_train:.2f} train, {r.r2_test:.2f} test (shaded)")
        ax.set_ylabel(UNIT)
        ax.legend(frameon=False, fontsize=9)
        ax.tick_params(axis="x", rotation=30)
    return _save(fig, path)


# --- Q5 ----------------------------------------------------------------------

def diwali(windows, path):
    cities = list(windows["city"].unique())
    n = len(cities)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), sharey=True, squeeze=False)
    years = sorted(windows["year"].unique())
    colors = dict(zip(years, sns.color_palette("viridis", len(years))))
    for ax, city in zip(axes[0], cities):
        for year, w in windows[windows["city"] == city].groupby("year"):
            ax.plot(w["offset"], w["pm2_5"], marker="o", ms=3, color=colors[year], label=str(year))
        ax.axvline(0, color="black", ls="--", lw=0.8)
        ax.set_title(display_name(city))
        ax.set_xlabel("Days from Diwali")
        ax.legend(frameon=False, fontsize=9)
    axes[0][0].set_ylabel("Daily " + UNIT)
    fig.suptitle("PM2.5 around Diwali", fontweight="bold")
    return _save(fig, path)


def render_all(res, out_dir):
    _setup()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = dict(zip([name.removesuffix(".png") for name in CHART_FILES], [out / n for n in CHART_FILES]))
    return [
        monthly_profile(res.monthly, f["monthly_profile"]),
        hour_month_heatmap(res.hour_month, f["hour_month_heatmap"]),
        exceedance_days(res.exceedance, f["exceedance_days"]),
        weather_drivers(res.wind_speed, res.blh, res.rain, f["weather_drivers"]),
        correlation(res.corr, f["correlation"]),
        wind_sectors(res.sectors, f["wind_sectors"]),
        regression(res.regressions, f["regression"]),
        diwali(res.diwali, f["diwali"]),
    ]
