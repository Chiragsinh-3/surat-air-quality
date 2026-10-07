"""Command line: `python -m airq fetch` and `python -m airq report`."""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from airq.cities import CITIES, START
from airq.clean import MissingDataError
from airq.fetch import FetchError, fetch_city

ARCHIVE_LAG_DAYS = 7  # the weather archive runs about 5 days behind


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m airq",
                                     description="Surat vs Ahmedabad air quality and weather")
    sub = parser.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download a fresh snapshot from Open-Meteo")
    f.add_argument("--start", type=date.fromisoformat, default=START)
    f.add_argument("--end", type=date.fromisoformat, default=None,
                   help=f"last day (default: today minus {ARCHIVE_LAG_DAYS} days)")
    f.add_argument("--raw-dir", type=Path, default=Path("data/raw"))

    r = sub.add_parser("report", help="analyse the snapshot and draw the charts")
    r.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    r.add_argument("--charts-dir", type=Path, default=Path("charts"))

    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            _fetch(args)
        else:
            _report(args)
    except (FetchError, MissingDataError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def _fetch(args):
    end = args.end or date.today() - timedelta(days=ARCHIVE_LAG_DAYS)
    for city in CITIES:
        print(f"fetching {city} {args.start} to {end} ...")
        for path in fetch_city(city, args.start, end, args.raw_dir):
            print(f"  wrote {path}")


def _report(args):
    # imported here so `fetch` doesn't pay for loading matplotlib
    from airq import report
    from airq.plots import render_all

    res = report.run(args.raw_dir)
    print(report.summary(res))
    for path in render_all(res, args.charts_dir):
        print(f"wrote {path}")
