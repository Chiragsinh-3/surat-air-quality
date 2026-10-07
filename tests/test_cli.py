from datetime import date

from airq import cli
from airq.plots import CHART_FILES


def test_report_before_fetch_says_what_to_do(tmp_path, capsys):
    code = cli.main(["report", "--raw-dir", str(tmp_path / "nothing-here")])

    assert code == 1
    assert "python -m airq fetch" in capsys.readouterr().err


def test_report_writes_charts_and_summary_only(raw_dir, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)

    code = cli.main(["report", "--raw-dir", str(raw_dir)])

    assert code == 0
    assert sorted(p.name for p in (tmp_path / "charts").iterdir()) == sorted(CHART_FILES)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["charts", "raw"]  # no web page
    assert "Q4 Regression" in capsys.readouterr().out


def test_fetch_downloads_every_city_up_to_the_end_date(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "fetch_city", lambda city, start, end, out_dir: calls.append((city, start, end)) or [])

    code = cli.main(["fetch", "--end", "2026-09-30", "--raw-dir", str(tmp_path)])

    assert code == 0
    assert calls == [("surat", date(2022, 8, 1), date(2026, 9, 30)),
                     ("ahmedabad", date(2022, 8, 1), date(2026, 9, 30))]


def test_fetch_failure_exits_with_error(monkeypatch, tmp_path, capsys):
    from airq.fetch import FetchError

    def boom(*a, **k):
        raise FetchError("HTTP 400: bad date")

    monkeypatch.setattr(cli, "fetch_city", boom)

    assert cli.main(["fetch", "--raw-dir", str(tmp_path)]) == 1
    assert "bad date" in capsys.readouterr().err
