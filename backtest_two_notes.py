"""
backtest_two_notes.py
=====================
Backtest two five-year SPY autocallables from 2020-01-03 onward and write
both runs into a single Excel workbook.

Notes priced
------------
1. Laddered-down: AC barriers (100, 95, 90, 85, 80)% with 30% KI.
2. Flat:          AC barriers flat at 100% with 30% KI.

Workbook tabs
-------------
- TimeSeries_Laddered: daily MTM, normalised SPY adjusted-close, etc.
- TimeSeries_Flat:     same for the flat-barrier note.
- PerformanceStats:    summary stats for both notes side-by-side. Sharpe
                       uses the daily 3-month CMT rate inferred from
                       rates.csv as the risk-free rate.

The script reuses the per-day pricing/seasoning logic from
``example_backtest.py``; only the product spec, coupon solver and
output-writing pieces are customised here.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from example import (
    OUTPUT_DIR,
    OPTIONS_CSV,
    RATES_CSV,
    build_market_snapshot,
    load_options,
    load_rates,
    load_spot,
    price_note,
    rate_curve_for,
)
from example_backtest import (
    build_obs_dates,
    intrinsic_terminal_payoff,
    make_seasoned_product,
    write_simple_xlsx,
    year_fractions_from_dates,
)
from util.autocall_prep import build_autocall_inputs
from util.spot_data import adj_spot_at


REQUESTED_ISSUE_DATE = pd.Timestamp("2020-01-03")
# 2020-01-03 (Friday) is not present in the bundled SPY option chain — the
# earliest 2020 row in data/SPY.csv is 2020-01-04. We snap forward to the
# first available trade date at runtime.

OBS_TIMES = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
KI_BARRIER = 0.70  # "30% knock-in" interpreted as a 30% buffer: KI level at 70% of inception spot.

NOTE_SPECS = [
    {
        "label": "Laddered",
        "ac_barriers": np.array([1.00, 0.95, 0.90, 0.85, 0.80], dtype=np.float64),
    },
    {
        "label": "Flat",
        "ac_barriers": np.full(len(OBS_TIMES), 1.00, dtype=np.float64),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--paths", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--target-price", type=float, default=100.0)
    parser.add_argument("--coupon-low", type=float, default=0.0)
    parser.add_argument("--coupon-high", type=float, default=0.30)
    parser.add_argument("--substeps-per-interval", type=int, default=1)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output xlsx path. Default: outputs/autocall_two_notes_<issue>.xlsx",
    )
    return parser.parse_args()


def build_product(ac_barriers: np.ndarray, coupon_rate: float) -> dict:
    n_obs = len(ac_barriers)
    coupons = np.array([coupon_rate * (i + 1) for i in range(n_obs)], dtype=np.float64)
    return {
        "notional": 100.0,
        "strike": 1.0,
        "ki_barrier": KI_BARRIER,
        "continuous_ki": True,
        "ac_barriers": ac_barriers.astype(np.float64),
        "coupons": coupons,
        "coupon_rate": float(coupon_rate),
    }


def solve_coupon(
    spot: float,
    inputs: dict,
    ac_barriers: np.ndarray,
    *,
    target_price: float,
    coupon_low: float,
    coupon_high: float,
    n_paths: int,
    seed: int,
    substeps_per_interval: int,
    tol: float = 1e-4,
    max_iter: int = 32,
) -> tuple[float, float]:
    def price_for(rate: float) -> float:
        product = build_product(ac_barriers, rate)
        result = price_note(
            spot,
            inputs,
            product,
            n_paths=n_paths,
            seed=seed,
            compute_greeks=False,
            substeps_per_interval=substeps_per_interval,
        )
        return float(result.cv_price)

    lo, hi = float(coupon_low), float(coupon_high)
    p_lo, p_hi = price_for(lo), price_for(hi)
    if not (p_lo <= target_price <= p_hi):
        raise ValueError(
            f"Target {target_price:.4f} not bracketed: "
            f"price({lo:.4%})={p_lo:.4f}, price({hi:.4%})={p_hi:.4f}"
        )
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = price_for(mid)
        if abs(p_mid - target_price) < tol:
            return mid, p_mid
        if p_mid < target_price:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return mid, price_for(mid)


def roll_obs_to_trading_days(
    obs_dates: list[pd.Timestamp],
    trade_dates: list[pd.Timestamp],
) -> list[pd.Timestamp]:
    """Roll each scheduled obs date forward to the first trading day on or
    after it. Real-world annual observations follow the same modified-following
    convention; matching here lets the existing ``obs_date == current_date``
    check fire instead of silently skipping observations on weekends/holidays.
    """
    rolled: list[pd.Timestamp] = []
    for obs in obs_dates:
        match = next((d for d in trade_dates if d >= obs), None)
        if match is None:
            raise ValueError(f"No trading day on or after observation {obs.date()}")
        rolled.append(match)
    return rolled


def resolve_issue_date(
    requested: pd.Timestamp, options_df: pd.DataFrame
) -> pd.Timestamp:
    trade_dates = sorted(pd.Timestamp(d) for d in options_df["date"].unique())
    match = next((d for d in trade_dates if d >= requested), None)
    if match is None:
        raise ValueError(f"No trade date on or after {requested.date()}")
    return match


def run_backtest(
    *,
    label: str,
    issue_date: pd.Timestamp,
    ac_barriers: np.ndarray,
    args: argparse.Namespace,
    options_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    spot_df: pd.DataFrame,
) -> dict:
    trade_dates = sorted(pd.Timestamp(d) for d in options_df["date"].unique())
    if issue_date not in trade_dates:
        raise ValueError(f"Issue date {issue_date.date()} not in options dataset")
    trade_dates = [d for d in trade_dates if d >= issue_date]

    scheduled_obs_dates = build_obs_dates(issue_date, OBS_TIMES)
    obs_dates = roll_obs_to_trading_days(scheduled_obs_dates, trade_dates)

    inception_snapshot = build_market_snapshot(options_df, rates_df, spot_df, issue_date)
    inception_spot = inception_snapshot["spot"]
    inception_spy_adj = adj_spot_at(issue_date, spot_df)

    long_end_anchor = (
        (1.0, inception_snapshot["ttm_yield"])
        if inception_snapshot.get("ttm_yield") is not None else None
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inception_inputs = build_autocall_inputs(
            obs_times=OBS_TIMES,
            vol_ts=inception_snapshot["vol_ts"],
            curve=inception_snapshot["curve"],
            q_provider=inception_snapshot["q_map"],
            long_end_q_anchor=long_end_anchor,
        )

    fair_coupon, fitted_issue_px = solve_coupon(
        inception_spot,
        inception_inputs,
        ac_barriers,
        target_price=args.target_price,
        coupon_low=args.coupon_low,
        coupon_high=args.coupon_high,
        n_paths=args.paths,
        seed=args.seed,
        substeps_per_interval=args.substeps_per_interval,
    )
    original_product = build_product(ac_barriers, fair_coupon)

    rows: list[dict] = []
    ki_hit = False
    ki_hit_date: pd.Timestamp | None = None
    status = "alive"

    # Self-financing delta replication state — initialised on the first row.
    prev_spot = np.nan
    prev_date: pd.Timestamp | None = None
    hedge_units_after = 0.0
    cash_after = np.nan
    portfolio_after_rehedge = np.nan

    for current_date in trade_dates:
        snapshot = build_market_snapshot(options_df, rates_df, spot_df, current_date)
        spot = snapshot["spot"]
        spy_adj = adj_spot_at(current_date, spot_df)

        if (not ki_hit) and spot <= original_product["ki_barrier"] * inception_spot:
            ki_hit = True
            ki_hit_date = current_date

        due_idx = [
            i for i, obs_date in enumerate(obs_dates) if obs_date == current_date
        ]

        event = ""
        mtm = np.nan
        delta = 0.0

        if due_idx:
            i = due_idx[0]
            if spot >= original_product["ac_barriers"][i] * inception_spot:
                status = "autocalled"
                event = f"autocalled_obs_{i + 1}"
                mtm = original_product["notional"] * (1.0 + original_product["coupons"][i])
                delta = 0.0

        if status == "alive":
            seasoned, remaining_obs_times = make_seasoned_product(
                issue_spot=inception_spot,
                current_spot=spot,
                current_date=current_date,
                original_product=original_product,
                original_obs_dates=obs_dates,
                ki_hit_already=ki_hit,
            )
            if seasoned is None:
                status = "matured"
                event = event or "maturity"
                mtm = intrinsic_terminal_payoff(
                    issue_spot=inception_spot,
                    current_spot=spot,
                    product=original_product,
                    ki_hit=ki_hit,
                )
                delta = 0.0
            else:
                if not due_idx or status == "alive":
                    event = event or (f"missed_obs_{due_idx[0] + 1}" if due_idx else "")
                long_end = (
                    (1.0, snapshot["ttm_yield"])
                    if snapshot.get("ttm_yield") is not None else None
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    inputs = build_autocall_inputs(
                        obs_times=remaining_obs_times,
                        vol_ts=snapshot["vol_ts"],
                        curve=snapshot["curve"],
                        q_provider=snapshot["q_map"],
                        long_end_q_anchor=long_end,
                    )
                result = price_note(
                    spot,
                    inputs,
                    seasoned,
                    n_paths=args.paths,
                    seed=args.seed,
                    compute_greeks=True,
                    substeps_per_interval=args.substeps_per_interval,
                )
                mtm = float(result.cv_price)
                delta = float(result.delta)

        # Self-financing delta-hedge bookkeeping: portfolio_before_rehedge is
        # the strategy MTM at the start of the day with yesterday's hedge,
        # marked to today's spot. That's the "delta replication" series.
        if prev_date is None:
            hedge_units_before = 0.0
            cash_before = mtm
            portfolio_before_rehedge = mtm
        else:
            hedge_units_before = hedge_units_after
            cash_before = cash_after
            portfolio_before_rehedge = hedge_units_before * spot + cash_before

        target_hedge_units = 0.0 if status in {"autocalled", "matured"} else float(delta)
        trade_units = target_hedge_units - hedge_units_before
        trade_cashflow = -trade_units * spot
        cash_after = cash_before + trade_cashflow
        hedge_units_after = target_hedge_units
        portfolio_after_rehedge = hedge_units_after * spot + cash_after

        replication_error = portfolio_before_rehedge - mtm

        rf_3m_cont = rate_curve_for(current_date, rates_df).r_continuous(0.25)

        rows.append({
            "date": current_date.date(),
            "status": status,
            "event": event,
            "spot": spot,
            "spy_adj_close": spy_adj,
            "perf_vs_issue": spot / inception_spot,
            "spy_adj_norm": spy_adj / inception_spy_adj,
            "spy_adj_rebased_100": 100.0 * spy_adj / inception_spy_adj,
            "ki_hit_so_far": ki_hit,
            "ki_hit_date": None if ki_hit_date is None else ki_hit_date.date(),
            "mtm": mtm,
            "mtm_rebased_100": 100.0 * mtm / args.target_price,
            "delta": delta,
            "hedge_units_after_rehedge": hedge_units_after,
            "cash_after_rehedge": cash_after,
            "delta_rep_portfolio": portfolio_before_rehedge,
            "delta_rep_rebased_100": 100.0 * portfolio_before_rehedge / args.target_price,
            "replication_error": replication_error,
            "rf_3m_cont": rf_3m_cont,
        })

        prev_spot = spot
        prev_date = current_date
        if status in {"autocalled", "matured"}:
            break

    print(
        f"  [{label}] coupon={fair_coupon:.4%} fitted_issue={fitted_issue_px:.4f} "
        f"rows={len(rows)} final_status={rows[-1]['status']}"
    )

    return {
        "label": label,
        "rows": pd.DataFrame(rows),
        "fair_coupon": fair_coupon,
        "fitted_issue_price": fitted_issue_px,
        "inception_spot": inception_spot,
        "inception_spy_adj": inception_spy_adj,
        "ac_barriers": ac_barriers,
        "obs_dates": obs_dates,
        "final_status": rows[-1]["status"],
    }


def _safe_cagr(start: float, end: float, years: float) -> float:
    if years <= 0 or start <= 0 or end <= 0:
        return float("nan")
    return (end / start) ** (1.0 / years) - 1.0


def _max_drawdown(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float((clean / clean.cummax() - 1.0).min())


def compute_stats(label: str, df: pd.DataFrame) -> dict:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    note_value = pd.to_numeric(df["mtm"], errors="coerce")
    spy_value = pd.to_numeric(df["spy_adj_close"], errors="coerce")
    rep_value = pd.to_numeric(df["delta_rep_portfolio"], errors="coerce")
    rep_err = pd.to_numeric(df["replication_error"], errors="coerce")
    rf_cont = pd.to_numeric(df["rf_3m_cont"], errors="coerce")

    note_ret = note_value.pct_change()
    spy_ret = spy_value.pct_change()
    rep_ret = rep_value.pct_change()

    start_date = df["date"].iloc[0]
    end_date = df["date"].iloc[-1]
    years = max((end_date - start_date).days / 365.0, 0.0)
    note_start, note_end = float(note_value.iloc[0]), float(note_value.iloc[-1])
    spy_start, spy_end = float(spy_value.iloc[0]), float(spy_value.iloc[-1])
    rep_start, rep_end = float(rep_value.iloc[0]), float(rep_value.iloc[-1])

    # Per-day simple risk-free from 3M CMT (continuous → per-trading-day simple).
    rf_daily = np.expm1(rf_cont / 252.0)

    def excess_cagr(ret: pd.Series) -> float:
        excess = (ret - rf_daily).dropna()
        if excess.empty or years <= 0.0:
            return float("nan")
        wealth = float((1.0 + excess).prod())
        if wealth <= 0.0:
            return float("nan")
        return wealth ** (1.0 / years) - 1.0

    def ann_vol(ret: pd.Series) -> float:
        clean = ret.dropna()
        if clean.shape[0] < 2:
            return float("nan")
        return float(clean.std(ddof=1) * np.sqrt(252.0))

    def sharpe(excess_cagr_val: float, vol: float) -> float:
        if not np.isfinite(excess_cagr_val) or not np.isfinite(vol) or vol == 0.0:
            return float("nan")
        return excess_cagr_val / vol

    note_cagr = _safe_cagr(note_start, note_end, years)
    spy_cagr = _safe_cagr(spy_start, spy_end, years)
    rep_cagr = _safe_cagr(rep_start, rep_end, years)
    note_ann_vol = ann_vol(note_ret)
    spy_ann_vol = ann_vol(spy_ret)
    rep_ann_vol = ann_vol(rep_ret)
    note_excess_cagr = excess_cagr(note_ret)
    spy_excess_cagr = excess_cagr(spy_ret)
    rep_excess_cagr = excess_cagr(rep_ret)

    rep_err_clean = rep_err.dropna()
    mean_abs_rep_err = float(rep_err_clean.abs().mean()) if not rep_err_clean.empty else float("nan")
    rmse_rep_err = (
        float(np.sqrt(np.mean(np.square(rep_err_clean))))
        if not rep_err_clean.empty else float("nan")
    )
    max_abs_rep_err = float(rep_err_clean.abs().max()) if not rep_err_clean.empty else float("nan")

    return {
        "metric_order": [
            "label", "start_date", "end_date", "years_elapsed", "final_status",
            "note_start", "note_end", "note_total_return",
            "note_cagr", "note_excess_cagr", "note_ann_vol", "note_sharpe", "note_max_drawdown",
            "delta_rep_start", "delta_rep_end", "delta_rep_total_return",
            "delta_rep_cagr", "delta_rep_excess_cagr", "delta_rep_ann_vol",
            "delta_rep_sharpe", "delta_rep_max_drawdown",
            "mean_abs_replication_error", "rmse_replication_error", "max_abs_replication_error",
            "spy_start", "spy_end", "spy_total_return",
            "spy_cagr", "spy_excess_cagr", "spy_ann_vol", "spy_sharpe", "spy_max_drawdown",
            "mean_rf_annual_continuous",
        ],
        "label": label,
        "start_date": start_date.date().isoformat(),
        "end_date": end_date.date().isoformat(),
        "years_elapsed": years,
        "final_status": df["status"].iloc[-1],
        "note_start": note_start,
        "note_end": note_end,
        "note_total_return": note_end / note_start - 1.0,
        "note_cagr": note_cagr,
        "note_excess_cagr": note_excess_cagr,
        "note_ann_vol": note_ann_vol,
        "note_sharpe": sharpe(note_excess_cagr, note_ann_vol),
        "note_max_drawdown": _max_drawdown(note_value),
        "delta_rep_start": rep_start,
        "delta_rep_end": rep_end,
        "delta_rep_total_return": rep_end / rep_start - 1.0,
        "delta_rep_cagr": rep_cagr,
        "delta_rep_excess_cagr": rep_excess_cagr,
        "delta_rep_ann_vol": rep_ann_vol,
        "delta_rep_sharpe": sharpe(rep_excess_cagr, rep_ann_vol),
        "delta_rep_max_drawdown": _max_drawdown(rep_value),
        "mean_abs_replication_error": mean_abs_rep_err,
        "rmse_replication_error": rmse_rep_err,
        "max_abs_replication_error": max_abs_rep_err,
        "spy_start": spy_start,
        "spy_end": spy_end,
        "spy_total_return": spy_end / spy_start - 1.0,
        "spy_cagr": spy_cagr,
        "spy_excess_cagr": spy_excess_cagr,
        "spy_ann_vol": spy_ann_vol,
        "spy_sharpe": sharpe(spy_excess_cagr, spy_ann_vol),
        "spy_max_drawdown": _max_drawdown(spy_value),
        "mean_rf_annual_continuous": float(rf_cont.mean()),
    }


def build_stats_sheet(results: list[dict]) -> pd.DataFrame:
    stats_per_run = [compute_stats(r["label"], r["rows"]) for r in results]
    metric_order = stats_per_run[0]["metric_order"]
    extra_rows = [
        ("coupon_rate_per_obs", [r["fair_coupon"] for r in results]),
        ("fitted_issue_price", [r["fitted_issue_price"] for r in results]),
        ("inception_spot", [r["inception_spot"] for r in results]),
        ("inception_spy_adj_close", [r["inception_spy_adj"] for r in results]),
        ("ki_barrier", [KI_BARRIER for _ in results]),
        ("ac_barriers", [", ".join(f"{x:.2f}" for x in r["ac_barriers"]) for r in results]),
        ("obs_dates", [", ".join(d.date().isoformat() for d in r["obs_dates"]) for r in results]),
    ]

    columns = ["metric"] + [s["label"] for s in stats_per_run]
    rows = []
    for metric in metric_order:
        rows.append([metric] + [s[metric] for s in stats_per_run])
    for metric, values in extra_rows:
        rows.append([metric] + values)
    return pd.DataFrame(rows, columns=columns)


def build_timeseries_sheet(result: dict) -> pd.DataFrame:
    df = result["rows"].copy()
    return df[
        [
            "date", "status", "event", "spot", "spy_adj_close",
            "perf_vs_issue", "spy_adj_norm", "spy_adj_rebased_100",
            "ki_hit_so_far", "ki_hit_date",
            "mtm", "mtm_rebased_100",
            "delta", "hedge_units_after_rehedge", "cash_after_rehedge",
            "delta_rep_portfolio", "delta_rep_rebased_100", "replication_error",
            "rf_3m_cont",
        ]
    ]


def build_combined_normalized_sheet(results: list[dict]) -> pd.DataFrame:
    """One tidy tab with everything rebased to 100 at inception, ready to plot."""
    frames = []
    for r in results:
        rebased = r["rows"][[
            "date", "spy_adj_rebased_100",
            "mtm_rebased_100", "delta_rep_rebased_100",
        ]].copy()
        rebased = rebased.rename(columns={
            "mtm_rebased_100": f"mtm_{r['label']}",
            "delta_rep_rebased_100": f"delta_rep_{r['label']}",
        })
        frames.append(rebased)

    # Outer-join on date so plotting is straightforward even if the two
    # backtests stop on different days. SPY column is identical across runs;
    # take it from the first frame to avoid duplicate columns.
    combined = frames[0]
    for nxt in frames[1:]:
        nxt = nxt.drop(columns=["spy_adj_rebased_100"])
        combined = combined.merge(nxt, on="date", how="outer")
    combined = combined.sort_values("date").reset_index(drop=True)

    note_cols = []
    for r in results:
        note_cols.append(f"mtm_{r['label']}")
        note_cols.append(f"delta_rep_{r['label']}")
    return combined[["date", "spy_adj_rebased_100", *note_cols]]


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

    options_df = load_options(OPTIONS_CSV)
    rates_df = load_rates(RATES_CSV)
    spot_df = load_spot(options_df)

    issue_date = resolve_issue_date(REQUESTED_ISSUE_DATE, options_df)
    if issue_date != REQUESTED_ISSUE_DATE:
        print(
            f"Requested issue date {REQUESTED_ISSUE_DATE.date()} not in option "
            f"chain; using next available trade date {issue_date.date()}."
        )

    results = []
    for spec in NOTE_SPECS:
        print(f"Running backtest: {spec['label']}")
        results.append(
            run_backtest(
                label=spec["label"],
                issue_date=issue_date,
                ac_barriers=spec["ac_barriers"],
                args=args,
                options_df=options_df,
                rates_df=rates_df,
                spot_df=spot_df,
            )
        )

    sheets = [("Normalized_TimeSeries", build_combined_normalized_sheet(results))]
    for r in results:
        sheets.append((f"TimeSeries_{r['label']}", build_timeseries_sheet(r)))
    sheets.append(("PerformanceStats", build_stats_sheet(results)))

    out_path = (
        Path(args.output) if args.output is not None
        else OUTPUT_DIR / f"autocall_two_notes_{issue_date.strftime('%Y%m%d')}.xlsx"
    )
    try:
        write_simple_xlsx(out_path, sheets)
    except PermissionError:
        stamp = pd.Timestamp.now().strftime("%H%M%S")
        out_path = out_path.with_name(f"{out_path.stem}_{stamp}{out_path.suffix}")
        write_simple_xlsx(out_path, sheets)

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Issue date:   {issue_date.date()}")
    print(f"KI barrier:   {KI_BARRIER:.0%}")
    print(f"Output:       {out_path}")


if __name__ == "__main__":
    main()
