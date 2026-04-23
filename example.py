"""
example.py
==========
Single-date SPY autocall example using the local market data files.

What it does
------------
1. Loads one SPY trade date from `data/SPY.csv`
2. Builds the matching rates curve from `data/rates.csv`
3. Recovers spot from near-dated put-call parity
4. Extracts parity-implied dividend yields by expiry
5. De-Americanizes the same-day chain
6. Builds forward vols / rates / divs from the de-Americanized surface
7. Solves for the per-observation coupon rate that matches a target note price
8. Prices one default single-asset autocallable with the compiled pricer in `prod/`

Default note terms
------------------
- 5-year maturity
- annual observations at 1y, 2y, 3y, 4y, 5y
- flat autocall barrier at 100%
- flat knock-in barrier at 69%
- coupon solved to match target price (default: par)

By default it uses the latest trade date in the CSV that has at least two
expirations. You can override that with `--date YYYY-MM-DD`.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
PROD_DIR = ROOT_DIR / "prod"
OUTPUT_DIR = ROOT_DIR / "outputs"
UTIL_DIR = ROOT_DIR / "util"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(UTIL_DIR))
sys.path.insert(0, str(PROD_DIR))

from util.deamerican import (  # noqa: E402
    YieldCurve,
    parity_implied_q_iterated,
    deamericanize_chain,
)
from util.autocall_prep import (  # noqa: E402
    ATMVolTermStructure,
    build_autocall_inputs,
    summarize_inputs,
)
from util.spot_data import (  # noqa: E402
    ensure_spot_data,
    fetch_ttm_dividend_yield,
    spot_at,
)

try:
    import autocall_pricer_lv as ap  # noqa: E402
except ModuleNotFoundError as exc:
    cp310_build = PROD_DIR / "autocall_pricer_lv.cp310-win_amd64.pyd"
    if cp310_build.exists() and sys.version_info[:2] != (3, 10):
        pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(
            f"autocall_pricer_lv is only available in this repo as a CPython 3.10 "
            f"extension ({cp310_build.name}), but you are running Python {pyver}. "
            "Use '.venv\\Scripts\\python.exe example.py' or rebuild the extension "
            "for your current interpreter with "
            "'python prod\\setup.py build_ext --inplace'."
        ) from exc
    raise


OPTIONS_CSV = DATA_DIR / "SPY.csv"
RATES_CSV = DATA_DIR / "rates.csv"
SPOT_CSV = DATA_DIR / "SPY_spot.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Price one default single-asset autocall from local SPY option data."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Trade date in YYYY-MM-DD. Defaults to the latest date with >= 2 expirations.",
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=100_000,
        help="Monte Carlo paths for the autocall pricer. Default: 100000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the autocall pricer. Default: 42.",
    )
    parser.add_argument(
        "--target-price",
        type=float,
        default=100.0,
        help="Target price used to solve the fair coupon rate. Default: 100.0.",
    )
    parser.add_argument(
        "--coupon-low",
        type=float,
        default=0.0,
        help="Lower bound for the per-observation coupon-rate solver. Default: 0.0.",
    )
    parser.add_argument(
        "--coupon-high",
        type=float,
        default=0.20,
        help="Upper bound for the per-observation coupon-rate solver. Default: 0.20.",
    )
    parser.add_argument(
        "--substeps-per-interval",
        type=int,
        default=1,
        help="Euler sub-steps per observation interval for the local-vol pricer. Default: 1.",
    )
    parser.add_argument(
        "--ac-barriers",
        type=str,
        default=None,
        help="Comma-separated autocall barriers (one per obs date). "
             "Example: '1.0,0.95,0.9,0.85,0.8'. Default: flat 1.00.",
    )
    parser.add_argument(
        "--ki-barrier",
        type=float,
        default=None,
        help="Knock-in barrier as fraction of spot (e.g. 0.70). Default: 0.69.",
    )
    return parser.parse_args()


def load_options(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date", "expiration"])
    if "act_symbol" in df.columns:
        df = df[df["act_symbol"] == "SPY"].copy()
    return df


def load_rates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date").sort_index()


def load_spot(options_df: pd.DataFrame, path: Path = SPOT_CSV) -> pd.DataFrame:
    """Return a (date, close) DataFrame covering every trade date in
    ``options_df``. Refreshes the CSV from Yahoo Finance if the cache does
    not already span the required window.
    """
    dates = pd.to_datetime(options_df["date"])
    return ensure_spot_data(path, dates.min(), dates.max())


def rate_curve_for(trade_date: pd.Timestamp, rates_df: pd.DataFrame) -> YieldCurve:
    idx = rates_df.index.searchsorted(trade_date, side="right") - 1
    if idx < 0:
        raise ValueError(f"No rates row on or before {trade_date.date()}")
    return YieldCurve.from_cmt_row(rates_df.iloc[idx])


def pick_trade_date(options_df: pd.DataFrame, requested_date: str | None) -> pd.Timestamp:
    expiry_counts = options_df.groupby("date")["expiration"].nunique().sort_index()
    eligible = expiry_counts[expiry_counts >= 2]
    if eligible.empty:
        raise ValueError("Need at least one trade date with >= 2 expirations.")

    if requested_date is None:
        return pd.Timestamp(eligible.index.max())

    dt = pd.Timestamp(requested_date)
    if dt not in expiry_counts.index:
        raise ValueError(f"{requested_date} is not present in {OPTIONS_CSV.name}")
    if expiry_counts.loc[dt] < 2:
        raise ValueError(f"{requested_date} has only {expiry_counts.loc[dt]} expiration(s); need >= 2.")
    return dt


def build_default_obs_schedule() -> np.ndarray:
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)


def build_default_product(
    obs_times: np.ndarray,
    coupon_rate: float,
    ac_barriers: np.ndarray | None = None,
    ki_barrier: float = 0.69,
) -> dict:
    n_obs = len(obs_times)
    coupons = np.array([coupon_rate * (i + 1) for i in range(n_obs)], dtype=np.float64)
    if ac_barriers is None:
        ac = np.full(n_obs, 1.00, dtype=np.float64)
    else:
        ac = np.asarray(ac_barriers, dtype=np.float64)
        if ac.shape[0] != n_obs:
            raise ValueError(
                f"ac_barriers length {ac.shape[0]} must match obs_times length {n_obs}"
            )
    return {
        "notional": 100.0,
        "strike": 1.0,
        "ki_barrier": float(ki_barrier),
        "continuous_ki": True,
        "ac_barriers": ac,
        "coupons": coupons,
        "coupon_rate": float(coupon_rate),
    }


def build_market_snapshot(
    options_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    spot_df: pd.DataFrame,
    trade_date: pd.Timestamp,
) -> dict:
    chain = options_df[options_df["date"] == trade_date].copy()
    if chain.empty:
        raise ValueError(f"No option rows found for {trade_date.date()}")

    curve = rate_curve_for(trade_date, rates_df)

    # Spot comes from Yahoo Finance closes — independent of the option chain.
    # Put-call parity on the chain then cleanly solves for q(T), instead of
    # conflating spot and dividend together.
    spot = spot_at(trade_date, spot_df)

    q_map = parity_implied_q_iterated(
        chain_df=chain,
        spot=spot,
        curve=curve,
        n_iter=2,
    )

    deam = deamericanize_chain(
        chain_df=chain,
        spot=spot,
        curve=curve,
        q_per_expiry=q_map,
        valuation_date=trade_date,
        method="baw",
    )

    vol_ts = ATMVolTermStructure.from_chain(deam, spot=spot, band=0.03)

    # Anchor the long end of the dividend curve with SPY TTM yield. The
    # quoted option chain typically only sees the next 1-2 quarterly
    # ex-dividends, so flat-extrapolating parity-implied q beyond the last
    # expiry under-dividends any multi-year product. Falling back to None on
    # a yfinance failure leaves the pricing path unanchored (legacy
    # behaviour), with a warning rather than a crash.
    try:
        ttm_yield = fetch_ttm_dividend_yield(
            ticker="SPY", as_of=trade_date, spot=float(spot),
        )
    except Exception as exc:
        warnings.warn(
            f"TTM dividend yield fetch failed ({exc}); long-end q anchor "
            "disabled. Long-dated forwards will flat-extrapolate short-end "
            "parity, which under-states SPY dividends.",
            RuntimeWarning,
        )
        ttm_yield = None

    return {
        "trade_date": trade_date,
        "chain": chain,
        "curve": curve,
        "spot": float(spot),
        "q_map": q_map,
        "ttm_yield": ttm_yield,
        "deamericanized": deam,
        "vol_ts": vol_ts,
    }


def price_note(
    spot: float,
    inputs: dict,
    product: dict,
    *,
    n_paths: int,
    seed: int,
    compute_greeks: bool,
    substeps_per_interval: int = 1,
):
    spec = ap.make_single_spec(
        spot,
        product["notional"],
        product["strike"],
        product["ki_barrier"],
        product["continuous_ki"],
        inputs["obs_times"],
        product["ac_barriers"],
        product["coupons"],
        inputs["fwd_vols"],
        inputs["fwd_rates"],
        inputs["fwd_divs"],
    )
    return ap.price_single_asset(
        spec,
        n_paths=n_paths,
        seed=seed,
        use_sobol=True,
        use_brownian_bridge=True,
        simd_width=-1,
        use_control_variate=True,
        compute_greeks=compute_greeks,
        n_threads=0,
        substeps_per_interval=substeps_per_interval,
    )


def solve_coupon_rate(
    spot: float,
    obs_times: np.ndarray,
    inputs: dict,
    *,
    target_price: float,
    coupon_low: float,
    coupon_high: float,
    n_paths: int,
    seed: int,
    substeps_per_interval: int = 1,
    tol: float = 1e-4,
    max_iter: int = 32,
    ac_barriers: np.ndarray | None = None,
    ki_barrier: float = 0.69,
) -> tuple[float, float]:
    def price_for(rate: float) -> float:
        product = build_default_product(obs_times, rate, ac_barriers, ki_barrier)
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

    lo = float(coupon_low)
    hi = float(coupon_high)
    price_lo = price_for(lo)
    price_hi = price_for(hi)

    if not (price_lo <= target_price <= price_hi):
        raise ValueError(
            f"Target price {target_price:.6f} is not bracketed by coupon bounds: "
            f"price({lo:.4%})={price_lo:.6f}, price({hi:.4%})={price_hi:.6f}."
        )

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        price_mid = price_for(mid)
        err = price_mid - target_price
        if abs(err) < tol:
            return mid, price_mid
        if err < 0.0:
            lo = mid
        else:
            hi = mid

    mid = 0.5 * (lo + hi)
    return mid, price_for(mid)


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

    options_df = load_options(OPTIONS_CSV)
    rates_df = load_rates(RATES_CSV)
    spot_df = load_spot(options_df)
    trade_date = pick_trade_date(options_df, args.date)

    snapshot = build_market_snapshot(options_df, rates_df, spot_df, trade_date)
    obs_times = build_default_obs_schedule()

    if args.ac_barriers is not None:
        ac_barriers_arg = np.array(
            [float(x) for x in args.ac_barriers.split(",")], dtype=np.float64
        )
    else:
        ac_barriers_arg = None
    ki_barrier_arg = 0.69 if args.ki_barrier is None else float(args.ki_barrier)

    long_end_anchor = (
        (1.0, snapshot["ttm_yield"]) if snapshot.get("ttm_yield") is not None
        else None
    )
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        inputs = build_autocall_inputs(
            obs_times=obs_times,
            vol_ts=snapshot["vol_ts"],
            curve=snapshot["curve"],
            q_provider=snapshot["q_map"],
            long_end_q_anchor=long_end_anchor,
        )

    fair_coupon_rate, fitted_price = solve_coupon_rate(
        snapshot["spot"],
        obs_times,
        inputs,
        target_price=args.target_price,
        coupon_low=args.coupon_low,
        coupon_high=args.coupon_high,
        n_paths=args.paths,
        seed=args.seed,
        substeps_per_interval=args.substeps_per_interval,
        ac_barriers=ac_barriers_arg,
        ki_barrier=ki_barrier_arg,
    )
    product = build_default_product(
        obs_times, fair_coupon_rate, ac_barriers_arg, ki_barrier_arg
    )
    result = price_note(
        snapshot["spot"],
        inputs,
        product,
        n_paths=args.paths,
        seed=args.seed,
        compute_greeks=True,
        substeps_per_interval=args.substeps_per_interval,
    )

    summary = summarize_inputs(inputs)
    date_tag = trade_date.strftime("%Y%m%d")
    summary_path = OUTPUT_DIR / f"autocall_inputs_{date_tag}.csv"
    deam_path = OUTPUT_DIR / f"deamericanized_slice_{date_tag}.csv"
    result_path = OUTPUT_DIR / f"autocall_result_{date_tag}.txt"
    summary.to_csv(summary_path, index=False)
    snapshot["deamericanized"].to_csv(deam_path, index=False)

    lines = [
        "Single-asset autocall example",
        f"trade_date={trade_date.date()}",
        f"spot={snapshot['spot']:.4f}",
        f"n_paths={args.paths}",
        f"seed={args.seed}",
        f"target_price={args.target_price:.6f}",
        f"coupon_rate={fair_coupon_rate:.6f}",
        f"method={result.method}",
        f"price={result.cv_price:.6f}",
        f"fitted_price={fitted_price:.6f}",
        f"stderr={result.cv_stderr:.6f}",
        f"delta={result.delta:.6f}",
        f"gamma={result.gamma:.6f}",
        f"vega_total={result.vega_total:.6f}",
        f"rho_total={result.rho_total:.6f}",
        f"autocall_prob={result.autocall_prob:.6%}",
        f"ki_prob={result.ki_prob:.6%}",
    ]
    result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 72)
    print("SINGLE-ASSET AUTOCALL EXAMPLE")
    print("=" * 72)
    if hasattr(ap, "has_avx512"):
        print(f"AVX-512 available: {ap.has_avx512()}")
    if hasattr(ap, "has_avx2"):
        print(f"AVX2 available:    {ap.has_avx2()}")
    print(f"Trade date:        {trade_date.date()}")
    print(f"Substeps/interval: {args.substeps_per_interval}")
    print(f"Spot:              {snapshot['spot']:.4f}")
    print(f"Expiries used:     {len(obs_times)}")
    print(f"Obs times (yrs):   {np.array2string(obs_times, precision=5)}")
    print(f"Autocall barriers: {np.array2string(product['ac_barriers'], precision=4)}")
    print(f"Target price:      {args.target_price:.6f}")
    print(f"Coupon rate/obs:   {fair_coupon_rate:.4%}")
    print(f"Coupons:           {np.array2string(product['coupons'], precision=4)}")
    if warning_list:
        print("Warnings:")
        for item in warning_list:
            print(f"  - {item.message}")

    print("\nForward market inputs:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nPricing result:")
    print(f"  Fitted price: {fitted_price:.6f}")
    print(f"  Price:       {result.cv_price:.6f} +/- {result.cv_stderr:.6f}")
    print(f"  Delta:       {result.delta:+.6f}")
    print(f"  Gamma:       {result.gamma:+.6f}")
    print(f"  Vega total:  {result.vega_total:+.6f}")
    print(f"  Rho total:   {result.rho_total:+.6f}")
    print(f"  AC prob:     {result.autocall_prob:.2%}")
    print(f"  KI prob:     {result.ki_prob:.2%}")

    print("\nFiles written:")
    print(f"  {summary_path}")
    print(f"  {deam_path}")
    print(f"  {result_path}")


if __name__ == "__main__":
    main()
