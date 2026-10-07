from airq import report
from airq.plots import CHART_FILES, render_all


def test_report_bundle_has_both_cities(raw_dir):
    res = report.run(raw_dir)

    assert res.cities == ["surat", "ahmedabad"]
    assert set(res.regressions) == {"surat", "ahmedabad"}
    assert set(res.hour_month) == {"surat", "ahmedabad"}
    assert not res.diwali.empty


def test_render_all_writes_every_chart(raw_dir, tmp_path):
    res = report.run(raw_dir)

    paths = render_all(res, tmp_path / "charts")

    assert sorted(p.name for p in paths) == sorted(CHART_FILES)
    assert len(CHART_FILES) == 8
    for p in paths:
        assert p.stat().st_size > 10_000, p.name


def test_summary_mentions_the_key_numbers(raw_dir):
    text = report.summary(report.run(raw_dir))

    for needle in ["Surat", "Ahmedabad", "R²", "Diwali", "winter", "% of days"]:
        assert needle in text




def test_summary_lists_wind_bands_in_order(raw_dir):
    text = report.summary(report.run(raw_dir))

    section = text.split("by wind speed (km/h) ==")[1].split("==")[0]
    positions = [section.index(f"\n{band} ") for band in ["0-5", "5-10", "10-15", "15-20", "20+"]]
    assert positions == sorted(positions)


def test_summary_reports_gaps_in_weather_data(raw_dir):
    import numpy as np

    res = report.run(raw_dir)
    hourly = res.hourly.copy()
    gap = (hourly["city"] == "surat") & (hourly.index >= "2025-07-01") & (hourly.index < "2025-07-11")
    hourly.loc[gap, "boundary_layer_height"] = np.nan

    text = report.summary(report.analyze_all(hourly, res.counts))

    assert "boundary_layer_height missing 2025-07-01 to 2025-07-10 (10 days)" in text



def test_summary_prints_the_numbers_behind_the_findings(raw_dir):
    text = report.summary(report.run(raw_dir))

    for needle in ["model outage hours", "Mean daily PM2.5 by season",
                   "winter / monsoon", "SW/W vs N/NE/E", "test period only"]:
        assert needle in text, needle
