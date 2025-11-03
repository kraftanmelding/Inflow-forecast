import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from modelzoo.meps_cache import prefetch_meps_range


def _parse_int_list(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prefetch MEPS NetCDF files into the local cache.")
    parser.add_argument("--start", default=date.today().isoformat(), help="Start date (YYYY-MM-DD).")
    parser.add_argument(
        "--end",
        default=(date.today() + timedelta(days=1)).isoformat(),
        help="End date (YYYY-MM-DD, inclusive).",
    )
    parser.add_argument("--init-hours", default="0,3,6,9,12,15,18,21", help="Comma-separated init hours (UTC).")
    parser.add_argument("--leadtimes", default="0,1,2", help="Comma-separated lead times (hours).")
    parser.add_argument("--cache-dir", default="Data/meps_cache", help="Destination cache directory.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logging.")
    args = parser.parse_args(argv)

    start_date = pd.to_datetime(args.start).normalize()
    end_date = pd.to_datetime(args.end).normalize()
    init_hours = _parse_int_list(args.init_hours)
    leadtimes = _parse_int_list(args.leadtimes)
    cache_dir = Path(args.cache_dir)

    prefetch_meps_range(
        start_date=start_date,
        end_date=end_date,
        init_hours=init_hours,
        leadtimes=leadtimes,
        cache_dir=cache_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
