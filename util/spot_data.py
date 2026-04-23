"""
spot_data.py
============
Historical underlying spot prices for backtest-safe valuation.

We use Yahoo Finance closes as the source of truth for S(t). This decouples
spot from the option chain, so put-call parity on that same chain can then
be used to solve for the dividend yield q(T) cleanly — instead of mixing
spot and dividend together (see ``deamerican.implied_spot_from_parity`` for
the chain-only fallback).

Workflow
--------
- ``fetch_spy_spot_history(start, end)``: download via yfinance.
- ``load_spot_csv(path)``: read previously-saved CSV.
- ``save_spot_csv(df, path)``: persist (date, close).
- ``spot_at(date, spot_df)``: latest close on or before ``date`` (handles
  holidays / non-trading dates by walking back to the prior trading day).
- ``ensure_spot_data(path, start, end)``: top-level convenience — load the
  cache if it covers ``[start, end]``, otherwise refresh from yfinance.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def fetch_spy_spot_history(start, end, ticker: str = "SPY") -> pd.DataFrame:
    """Download daily closes from Yahoo Finance.

    Returns a DataFrame with ``date`` (timezone-naive midnight Timestamp),
    ``close`` (unadjusted close), and ``adj_close`` when Yahoo supplies it.
    """
    import yfinance as yf

    end_excl = pd.Timestamp(end) + pd.Timedelta(days=1)
    raw = yf.download(
        ticker,
        start=pd.Timestamp(start).strftime("%Y-%m-%d"),
        end=end_excl.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError(
            f"yfinance returned no data for {ticker} between {start} and {end}"
        )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    out = pd.DataFrame({
        "date": pd.to_datetime(raw.index).tz_localize(None).normalize(),
        "close": raw["Close"].astype(float).values,
    })
    if "Adj Close" in raw.columns:
        out["adj_close"] = raw["Adj Close"].astype(float).values
    else:
        out["adj_close"] = out["close"]
    return out.reset_index(drop=True)


def load_spot_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    if "adj_close" not in df.columns:
        df["adj_close"] = pd.NA
    df["adj_close"] = df["adj_close"].fillna(df["close"])
    return df.sort_values("date").reset_index(drop=True)


def save_spot_csv(df: pd.DataFrame, path: Path) -> None:
    cols = ["date", "close"]
    if "adj_close" in df.columns:
        cols.append("adj_close")
    out = df[cols].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def _price_at(date, spot_df: pd.DataFrame, column: str) -> float:
    """Latest price column on or before ``date``.

    If the exact date is missing (e.g., trading-day misalignment), walks back
    to the most recent prior row.
    """
    if column not in spot_df.columns:
        raise ValueError(f"Spot DataFrame does not contain column {column!r}")
    target = pd.Timestamp(date).tz_localize(None).normalize()
    dates = pd.to_datetime(spot_df["date"]).values
    idx = pd.Series(dates).searchsorted(target, side="right") - 1
    if idx < 0:
        raise ValueError(
            f"No spot on or before {target.date()} (earliest cached is "
            f"{pd.Timestamp(dates[0]).date()})"
        )
    value = spot_df[column].iloc[idx]
    if pd.isna(value):
        raise ValueError(f"No {column} value on or before {target.date()}")
    return float(value)


def spot_at(date, spot_df: pd.DataFrame) -> float:
    """Latest unadjusted close on or before ``date``."""
    return _price_at(date, spot_df, "close")


def adj_spot_at(date, spot_df: pd.DataFrame) -> float:
    """Latest adjusted close on or before ``date``."""
    return _price_at(date, spot_df, "adj_close")


def fetch_ttm_dividend_yield(ticker: str = "SPY",
                             as_of=None,
                             spot: float | None = None) -> float:
    """Trailing-12-month dividend yield from Yahoo Finance.

    Sums cash dividends with ex-date in ``(as_of - 365d, as_of]`` and divides
    by spot. Backtest-safe: pass the valuation-date spot explicitly and
    ``as_of`` = the trade date. The yfinance ``.dividends`` series carries
    full history, so filtering by ``as_of`` eliminates any forward leakage.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker. Default ``'SPY'``.
    as_of : date-like, optional
        The trailing window ends at this date. Default: today.
    spot : float, optional
        Underlying price used as the denominator. If omitted, falls back to
        the last Yahoo close on or before ``as_of``.

    Returns
    -------
    float
        Continuous-compounding-equivalent? No — a simple TTM *rate*, i.e.
        (sum of cash dividends over last 365 days) / spot. For the order of
        magnitude SPY pays (~1%), the gap between simple and continuous is
        negligible, so this is used directly as the q-anchor.
    """
    import yfinance as yf

    if as_of is None:
        as_of = pd.Timestamp.today().normalize()
    else:
        as_of = pd.Timestamp(as_of).tz_localize(None).normalize()

    t = yf.Ticker(ticker)
    divs = t.dividends
    if divs is None or len(divs) == 0:
        raise RuntimeError(f"No dividend history returned by yfinance for {ticker}")
    divs = divs.copy()
    idx = divs.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    divs.index = pd.to_datetime(idx).normalize()
    window_start = as_of - pd.Timedelta(days=365)
    ttm = divs[(divs.index > window_start) & (divs.index <= as_of)]
    if len(ttm) == 0:
        raise RuntimeError(
            f"No {ticker} dividends in TTM window ending {as_of.date()}"
        )
    ttm_cash = float(ttm.sum())

    if spot is None:
        hist = t.history(
            start=(as_of - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
            end=(as_of + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
        )
        if hist.empty:
            raise RuntimeError(
                f"No price data for {ticker} around {as_of.date()}"
            )
        spot = float(hist["Close"].iloc[-1])

    if spot <= 0:
        raise ValueError(f"Non-positive spot {spot} for {ticker}")
    return ttm_cash / spot


def ensure_spot_data(path: Path, start, end, ticker: str = "SPY") -> pd.DataFrame:
    """Return a spot DataFrame that covers ``[start, end]``.

    Coverage is measured via ``spot_at`` semantics: there must be at least
    one row on or before ``start`` (so the earliest requested date resolves
    to a prior trading day), and at least one row on or after ``end``. If
    the cache at ``path`` already satisfies that, return it. Otherwise fetch
    a padded window from yfinance, merge, and rewrite the CSV.
    """
    start = pd.Timestamp(start).tz_localize(None).normalize()
    end = pd.Timestamp(end).tz_localize(None).normalize()

    cache = load_spot_csv(path) if path.exists() else None
    covers = (
        cache is not None
        and not cache.empty
        and (cache["date"] <= start).any()
        and cache["date"].max() >= end
        and "adj_close" in cache.columns
        and cache["adj_close"].notna().all()
    )
    if covers:
        return cache

    # Pad by a week on each side so non-trading-day edges (weekends,
    # holidays) resolve to the nearest prior trading day.
    fetch_start = start - pd.Timedelta(days=7)
    fetch_end = end + pd.Timedelta(days=7)
    fresh = fetch_spy_spot_history(fetch_start, fetch_end, ticker=ticker)
    if cache is None or cache.empty:
        merged = fresh
    else:
        merged = (
            pd.concat([cache, fresh], ignore_index=True)
            .drop_duplicates(subset="date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_spot_csv(merged, path)
    return merged
