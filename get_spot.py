"""
get_spot.py
===========
Refresh ``data/SPY_spot.csv`` from Yahoo Finance.

By default, fetches closes spanning the full date range of the local option
file ``data/SPY.csv``. Override with ``--start`` / ``--end`` to target a
specific window.

Output: ``data/SPY_spot.csv`` with columns ``date,close`` — one row per
trading day at the unadjusted 4:00 PM ET close.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
OPTIONS_CSV = DATA_DIR / "SPY.csv"
SPOT_CSV = DATA_DIR / "SPY_spot.csv"

sys.path.insert(0, str(ROOT_DIR))

from util.spot_data import fetch_spy_spot_history, save_spot_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD. Default: earliest date in SPY.csv.")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD. Default: latest date in SPY.csv.")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--out", default=str(SPOT_CSV))
    args = parser.parse_args()

    start = args.start
    end = args.end
    if start is None or end is None:
        opts = pd.read_csv(OPTIONS_CSV, usecols=["date"], parse_dates=["date"])
        if start is None:
            start = opts["date"].min().strftime("%Y-%m-%d")
        if end is None:
            end = opts["date"].max().strftime("%Y-%m-%d")

    df = fetch_spy_spot_history(start, end, ticker=args.ticker)
    out_path = Path(args.out)
    save_spot_csv(df, out_path)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")


if __name__ == "__main__":
    main()
