"""
example_backtest.py
===================
Seasoned-note backtest example for a single-asset autocallable.

Scenario
--------
- Issue the note on 2026-04-01
- Solve the inception coupon to a target price (default: par)
- Revalue the same live note on each subsequent trade date in the dataset
- Carry forward note state externally:
  - whether KI has already been hit
  - whether the note has already autocalled
  - which observation dates remain

The script writes a daily backtest table including:
- note mark-to-market and Greeks
- SPY unadjusted close and adjusted close
- self-financing delta-replication portfolio
- hedge units bought/sold each day
- no cash carry; hedge tracked in excess-return terms
- replication error versus the note MTM
"""
from __future__ import annotations

import argparse
import math
import warnings
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from example import (
    OUTPUT_DIR,
    OPTIONS_CSV,
    RATES_CSV,
    build_default_obs_schedule,
    build_default_product,
    build_market_snapshot,
    load_options,
    load_rates,
    load_spot,
    price_note,
    solve_coupon_rate,
)
from util.autocall_prep import build_autocall_inputs
from util.spot_data import adj_spot_at


def _xlsx_col_name(idx: int) -> str:
    name = ""
    idx += 1
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        name = chr(65 + rem) + name
    return name


def _xlsx_cell(value, row_idx: int, col_idx: int) -> str:
    ref = f"{_xlsx_col_name(col_idx)}{row_idx}"
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f'<c r="{ref}"/>'
    if isinstance(value, (np.floating, float, np.integer, int)) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _build_sheet_xml(df: pd.DataFrame) -> str:
    rows_xml = []
    header_cells = "".join(_xlsx_cell(col, 1, i) for i, col in enumerate(df.columns))
    rows_xml.append(f'<row r="1">{header_cells}</row>')

    for r, row in enumerate(df.itertuples(index=False, name=None), start=2):
        row_cells = "".join(_xlsx_cell(val, r, c) for c, val in enumerate(row))
        rows_xml.append(f'<row r="{r}">{row_cells}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>'
        f'{"".join(rows_xml)}'
        '</sheetData>'
        '</worksheet>'
    )
    return sheet_xml


def write_simple_xlsx(path: Path, sheets: list[tuple[str, pd.DataFrame]]) -> None:
    sanitized_names: list[str] = []
    seen_names: set[str] = set()
    for idx, (sheet_name, _) in enumerate(sheets, start=1):
        cleaned = "".join(ch for ch in sheet_name if ch not in '[]:*?/\\')[:31] or f"Sheet{idx}"
        base = cleaned
        suffix = 1
        while cleaned in seen_names:
            suffix += 1
            cleaned = f"{base[: max(0, 31 - len(str(suffix)) - 1)]}_{suffix}"
        seen_names.add(cleaned)
        sanitized_names.append(cleaned)

    sheet_entries = "".join(
        f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, name in enumerate(sanitized_names, start=1)
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets>'
        f"{sheet_entries}"
        '</sheets>'
        '</workbook>'
    )
    worksheet_rels = "".join(
        '<Relationship Id="rId{idx}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet{idx}.xml"/>'.format(idx=idx)
        for idx in range(1, len(sheets) + 1)
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{worksheet_rels}"
        f'<Relationship Id="rId{len(sheets) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        '</Relationships>'
    )
    worksheet_content_types = "".join(
        '<Override PartName="/xl/worksheets/sheet{idx}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'.format(idx=idx)
        for idx in range(1, len(sheets) + 1)
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        f"{worksheet_content_types}"
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '</Types>'
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
        '<cellXfs count="1"><xf xfId="0"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        for idx, (_, df) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _build_sheet_xml(df))
        zf.writestr("xl/styles.xml", styles_xml)


def _safe_cagr(start_value: float, end_value: float, years: float) -> float:
    if years <= 0.0 or start_value <= 0.0 or end_value <= 0.0:
        return np.nan
    return (end_value / start_value) ** (1.0 / years) - 1.0


def _safe_ann_vol(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if len(clean) < 2:
        return np.nan
    return float(clean.std(ddof=1) * np.sqrt(252.0))


def _max_drawdown(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    running_peak = clean.cummax()
    drawdown = clean / running_peak - 1.0
    return float(drawdown.min())


def build_summary_sheet(backtest_df: pd.DataFrame) -> pd.DataFrame:
    summary = backtest_df.copy()
    summary["date"] = pd.to_datetime(summary["date"])

    note_value = pd.to_numeric(summary["mtm"], errors="coerce")
    strategy_value = pd.to_numeric(summary["portfolio_before_rehedge"], errors="coerce")
    replication_gap = strategy_value - note_value
    note_return = note_value.pct_change()
    strategy_return = strategy_value.pct_change()
    replication_error = pd.to_numeric(summary["note_pnl_minus_strategy_pnl"], errors="coerce")
    spy_adj_source = (
        summary["spy_adj_close"]
        if "spy_adj_close" in summary.columns
        else pd.Series(np.nan, index=summary.index)
    )
    spy_adj_value = pd.to_numeric(spy_adj_source, errors="coerce")
    spy_adj_return = spy_adj_value.pct_change()

    start_date = summary["date"].iloc[0]
    end_date = summary["date"].iloc[-1]
    years = max((end_date - start_date).days / 365.0, 0.0)

    note_start = float(note_value.iloc[0])
    note_end = float(note_value.iloc[-1])
    strategy_start = float(strategy_value.iloc[0])
    strategy_end = float(strategy_value.iloc[-1])
    spy_adj_start = float(spy_adj_value.iloc[0]) if not spy_adj_value.empty else np.nan
    spy_adj_end = float(spy_adj_value.iloc[-1]) if not spy_adj_value.empty else np.nan

    stats_rows = [
        {"metric": "issue_date", "note": str(start_date.date()), "delta_strategy": str(start_date.date()), "spy_adj": str(start_date.date()), "replication_gap": ""},
        {"metric": "last_date", "note": str(end_date.date()), "delta_strategy": str(end_date.date()), "spy_adj": str(end_date.date()), "replication_gap": ""},
        {"metric": "years_elapsed", "note": years, "delta_strategy": years, "spy_adj": years, "replication_gap": ""},
        {"metric": "start_value", "note": note_start, "delta_strategy": strategy_start, "spy_adj": spy_adj_start, "replication_gap": replication_gap.iloc[0]},
        {"metric": "end_value", "note": note_end, "delta_strategy": strategy_end, "spy_adj": spy_adj_end, "replication_gap": replication_gap.iloc[-1]},
        {"metric": "total_return", "note": note_end / note_start - 1.0, "delta_strategy": strategy_end / strategy_start - 1.0, "spy_adj": spy_adj_end / spy_adj_start - 1.0, "replication_gap": replication_gap.iloc[-1]},
        {"metric": "cagr", "note": _safe_cagr(note_start, note_end, years), "delta_strategy": _safe_cagr(strategy_start, strategy_end, years), "spy_adj": _safe_cagr(spy_adj_start, spy_adj_end, years), "replication_gap": ""},
        {"metric": "ann_vol", "note": _safe_ann_vol(note_return), "delta_strategy": _safe_ann_vol(strategy_return), "spy_adj": _safe_ann_vol(spy_adj_return), "replication_gap": ""},
        {"metric": "max_drawdown", "note": _max_drawdown(note_value), "delta_strategy": _max_drawdown(strategy_value), "spy_adj": _max_drawdown(spy_adj_value), "replication_gap": ""},
        {"metric": "mean_daily_return", "note": float(note_return.dropna().mean()) if not note_return.dropna().empty else np.nan, "delta_strategy": float(strategy_return.dropna().mean()) if not strategy_return.dropna().empty else np.nan, "spy_adj": float(spy_adj_return.dropna().mean()) if not spy_adj_return.dropna().empty else np.nan, "replication_gap": ""},
        {"metric": "mean_abs_daily_replication_error", "note": "", "delta_strategy": "", "replication_gap": float(replication_error.abs().dropna().mean()) if not replication_error.dropna().empty else np.nan},
        {"metric": "rmse_daily_replication_error", "note": "", "delta_strategy": "", "replication_gap": float(np.sqrt(np.mean(np.square(replication_error.dropna())))) if not replication_error.dropna().empty else np.nan},
        {"metric": "max_abs_replication_gap", "note": "", "delta_strategy": "", "replication_gap": float(replication_gap.abs().dropna().max()) if not replication_gap.dropna().empty else np.nan},
        {"metric": "final_status", "note": str(summary["status"].iloc[-1]), "delta_strategy": str(summary["event"].iloc[-1]), "replication_gap": ""},
    ]
    return pd.DataFrame(stats_rows)


def build_spot_normalized_sheet(backtest_df: pd.DataFrame) -> pd.DataFrame:
    normalized = backtest_df.copy()
    normalized["date"] = pd.to_datetime(normalized["date"])

    spot = pd.to_numeric(normalized["spot"], errors="coerce")
    spy_adj_close = pd.to_numeric(normalized["spy_adj_close"], errors="coerce")
    inception_spot = pd.to_numeric(normalized["inception_spot"], errors="coerce")
    inception_spy_adj_close = pd.to_numeric(normalized["inception_spy_adj_close"], errors="coerce")
    note_value = pd.to_numeric(normalized["mtm"], errors="coerce")
    strategy_value = pd.to_numeric(normalized["portfolio_before_rehedge"], errors="coerce")

    spot_index = spot / inception_spot
    spy_adj_index = spy_adj_close / inception_spy_adj_close
    normalized["spot_index"] = spot_index
    normalized["spy_adj_index"] = spy_adj_index
    normalized["autocall_mtm_normalized"] = note_value / spot_index
    normalized["delta_replication_normalized"] = strategy_value / spot_index
    normalized["normalized_gap"] = (
        normalized["delta_replication_normalized"] - normalized["autocall_mtm_normalized"]
    )
    normalized["autocall_mtm_rebased_100"] = 100.0 * note_value / note_value.iloc[0]
    normalized["delta_replication_rebased_100"] = 100.0 * strategy_value / strategy_value.iloc[0]
    normalized["spot_rebased_100"] = 100.0 * spot / spot.iloc[0]
    normalized["spy_adj_rebased_100"] = 100.0 * spy_adj_close / spy_adj_close.iloc[0]

    return normalized[
        [
            "date",
            "status",
            "event",
            "spot",
            "spy_adj_close",
            "inception_spot",
            "inception_spy_adj_close",
            "spot_index",
            "spy_adj_index",
            "spot_rebased_100",
            "spy_adj_rebased_100",
            "mtm",
            "portfolio_before_rehedge",
            "autocall_mtm_normalized",
            "delta_replication_normalized",
            "normalized_gap",
            "autocall_mtm_rebased_100",
            "delta_replication_rebased_100",
            "replication_error",
            "note_pnl_from_prev",
            "strategy_pnl_from_prev",
        ]
    ].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest a seasoned single-asset autocallable from inception onward."
    )
    parser.add_argument(
        "--issue-date",
        type=str,
        default="2020-01-04",
        help="Issue date in YYYY-MM-DD. Default: 2026-04-01.",
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=25_000,
        help="Monte Carlo paths for each daily repricing. Default: 25000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Random seed. Default: 42.",
    )
    parser.add_argument(
        "--target-price",
        type=float,
        default=100.0,
        help="Inception target price used to solve the coupon. Default: 100.0.",
    )
    parser.add_argument(
        "--coupon-low",
        type=float,
        default=0.0,
        help="Lower bound for coupon solver. Default: 0.0.",
    )
    parser.add_argument(
        "--coupon-high",
        type=float,
        default=0.20,
        help="Upper bound for coupon solver. Default: 0.20.",
    )
    parser.add_argument(
        "--substeps-per-interval",
        type=int,
        default=1,
        help="Euler sub-steps per observation interval for the local-vol pricer. Default: 1.",
    )
    return parser.parse_args()


def year_fractions_from_dates(start_date: pd.Timestamp, end_dates: list[pd.Timestamp]) -> np.ndarray:
    return np.array(
        [
            (end_date - start_date).total_seconds() / (365.0 * 86400.0)
            for end_date in end_dates
        ],
        dtype=np.float64,
    )


def build_obs_dates(issue_date: pd.Timestamp, obs_times_years: np.ndarray) -> list[pd.Timestamp]:
    return [issue_date + pd.DateOffset(years=int(round(t))) for t in obs_times_years]


def make_seasoned_product(
    *,
    issue_spot: float,
    current_spot: float,
    current_date: pd.Timestamp,
    original_product: dict,
    original_obs_dates: list[pd.Timestamp],
    ki_hit_already: bool,
) -> tuple[dict | None, np.ndarray | None]:
    remaining_idx = [i for i, obs_date in enumerate(original_obs_dates) if obs_date > current_date]
    if not remaining_idx:
        return None, None

    perf_today = current_spot / issue_spot
    remaining_dates = [original_obs_dates[i] for i in remaining_idx]
    remaining_obs_times = year_fractions_from_dates(current_date, remaining_dates)

    seasoned = {
        "notional": original_product["notional"],
        "strike": original_product["strike"] / perf_today,
        "ki_barrier": (10.0 if ki_hit_already else original_product["ki_barrier"] / perf_today),
        "continuous_ki": original_product["continuous_ki"],
        "ac_barriers": original_product["ac_barriers"][remaining_idx] / perf_today,
        "coupons": original_product["coupons"][remaining_idx].copy(),
        "coupon_rate": original_product["coupon_rate"],
    }
    return seasoned, remaining_obs_times


def intrinsic_terminal_payoff(
    *,
    issue_spot: float,
    current_spot: float,
    product: dict,
    ki_hit: bool,
) -> float:
    if ki_hit and current_spot < product["strike"] * issue_spot:
        return product["notional"] * (current_spot / issue_spot) / product["strike"]
    return product["notional"]


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

    issue_date = pd.Timestamp(args.issue_date)
    options_df = load_options(OPTIONS_CSV)
    rates_df = load_rates(RATES_CSV)
    spot_df = load_spot(options_df)

    trade_dates = sorted(pd.Timestamp(d) for d in options_df["date"].unique())
    if issue_date not in trade_dates:
        raise ValueError(f"Issue date {issue_date.date()} is not present in {Path(OPTIONS_CSV).name}.")

    trade_dates = [d for d in trade_dates if d >= issue_date]

    original_obs_times = build_default_obs_schedule()
    original_obs_dates = build_obs_dates(issue_date, original_obs_times)

    inception_snapshot = build_market_snapshot(options_df, rates_df, spot_df, issue_date)
    inception_spot = inception_snapshot["spot"]
    inception_spy_adj_close = adj_spot_at(issue_date, spot_df)

    inception_long_end_anchor = (
        (1.0, inception_snapshot["ttm_yield"])
        if inception_snapshot.get("ttm_yield") is not None
        else None
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        inception_inputs = build_autocall_inputs(
            obs_times=original_obs_times,
            vol_ts=inception_snapshot["vol_ts"],
            curve=inception_snapshot["curve"],
            q_provider=inception_snapshot["q_map"],
            long_end_q_anchor=inception_long_end_anchor,
        )

    fair_coupon_rate, fitted_issue_price = solve_coupon_rate(
        inception_spot,
        original_obs_times,
        inception_inputs,
        target_price=args.target_price,
        coupon_low=args.coupon_low,
        coupon_high=args.coupon_high,
        n_paths=args.paths,
        seed=args.seed,
        substeps_per_interval=args.substeps_per_interval,
    )
    original_product = build_default_product(original_obs_times, fair_coupon_rate)

    rows: list[dict] = []
    mtm0 = np.nan
    prev_spot = np.nan
    prev_date: pd.Timestamp | None = None
    hedge_units_after = 0.0
    cash_after = np.nan
    portfolio_after_rehedge = np.nan
    ki_hit = False
    ki_hit_date: pd.Timestamp | None = None
    status = "alive"

    for current_date in trade_dates:
        snapshot = build_market_snapshot(options_df, rates_df, spot_df, current_date)
        spot = snapshot["spot"]
        spy_adj_close = adj_spot_at(current_date, spot_df)
        if (not ki_hit) and spot <= original_product["ki_barrier"] * inception_spot:
            ki_hit = True
            ki_hit_date = current_date

        due_obs_idx = [
            i for i, obs_date in enumerate(original_obs_dates)
            if obs_date == current_date
        ]

        event = ""
        mtm = np.nan
        stderr = np.nan
        delta = np.nan
        gamma = np.nan
        vega_total = np.nan
        rho_total = np.nan
        ac_prob = np.nan
        ki_prob = np.nan
        remaining_obs = 0

        if due_obs_idx:
            i = due_obs_idx[0]
            if spot >= original_product["ac_barriers"][i] * inception_spot:
                status = "autocalled"
                event = f"autocalled_obs_{i + 1}"
                mtm = original_product["notional"] * (1.0 + original_product["coupons"][i])
                stderr = 0.0
                delta = 0.0
                gamma = 0.0
                vega_total = 0.0
                rho_total = 0.0
                ac_prob = 1.0
                ki_prob = 1.0 if ki_hit else 0.0
            else:
                event = f"missed_obs_{i + 1}"

        if status == "alive":
            seasoned_product, remaining_obs_times = make_seasoned_product(
                issue_spot=inception_spot,
                current_spot=spot,
                current_date=current_date,
                original_product=original_product,
                original_obs_dates=original_obs_dates,
                ki_hit_already=ki_hit,
            )

            if seasoned_product is None:
                status = "matured"
                event = event or "maturity"
                mtm = intrinsic_terminal_payoff(
                    issue_spot=inception_spot,
                    current_spot=spot,
                    product=original_product,
                    ki_hit=ki_hit,
                )
                stderr = 0.0
                delta = 0.0
                gamma = 0.0
                vega_total = 0.0
                rho_total = 0.0
                ac_prob = 0.0
                ki_prob = 1.0 if ki_hit else 0.0
            else:
                remaining_obs = len(remaining_obs_times)
                long_end_anchor = (
                    (1.0, snapshot["ttm_yield"])
                    if snapshot.get("ttm_yield") is not None
                    else None
                )
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("ignore")
                    inputs = build_autocall_inputs(
                        obs_times=remaining_obs_times,
                        vol_ts=snapshot["vol_ts"],
                        curve=snapshot["curve"],
                        q_provider=snapshot["q_map"],
                        long_end_q_anchor=long_end_anchor,
                    )
                result = price_note(
                    spot,
                    inputs,
                    seasoned_product,
                    n_paths=args.paths,
                    seed=args.seed,
                    compute_greeks=True,
                    substeps_per_interval=args.substeps_per_interval,
                )
                mtm = float(result.cv_price)
                stderr = float(result.cv_stderr)
                delta = float(result.delta)
                gamma = float(result.gamma)
                vega_total = float(result.vega_total)
                rho_total = float(result.rho_total)
                ac_prob = float(result.autocall_prob)
                ki_prob = float(result.ki_prob)

        if prev_date is None:
            dt_years = 0.0
            hedge_units_before = 0.0
            cash_before = mtm
            spot_pnl = np.nan
            cash_carry = np.nan
            portfolio_before_rehedge = mtm
            strategy_pnl = np.nan
            note_pnl = np.nan
        else:
            dt_years = (current_date - prev_date).total_seconds() / (365.0 * 86400.0)
            hedge_units_before = hedge_units_after
            cash_before = cash_after
            spot_pnl = hedge_units_before * (spot - prev_spot)
            cash_carry = 0.0
            portfolio_before_rehedge = hedge_units_before * spot + cash_before
            strategy_pnl = portfolio_before_rehedge - portfolio_after_rehedge
            note_pnl = mtm - rows[-1]["mtm"]

        target_hedge_units = 0.0 if status in {"autocalled", "matured"} else float(delta)
        trade_units = target_hedge_units - hedge_units_before
        trade_cashflow = -trade_units * spot
        cash_after = cash_before + trade_cashflow
        hedge_units_after = target_hedge_units
        portfolio_after_rehedge = hedge_units_after * spot + cash_after

        replication_error = portfolio_before_rehedge - mtm
        note_pnl_minus_strategy = np.nan if np.isnan(note_pnl) else note_pnl - strategy_pnl
        cumulative_note_pnl = np.nan if np.isnan(mtm0) else mtm - mtm0
        cumulative_strategy_pnl = np.nan if np.isnan(mtm0) else portfolio_before_rehedge - mtm0
        cumulative_replication_error = np.nan if np.isnan(mtm0) else cumulative_strategy_pnl - cumulative_note_pnl

        rows.append(
            {
                "date": current_date.date(),
                "status": status,
                "event": event,
                "spot": spot,
                "spy_adj_close": spy_adj_close,
                "inception_spot": inception_spot,
                "inception_spy_adj_close": inception_spy_adj_close,
                "perf_vs_issue": spot / inception_spot,
                "spy_adj_perf_vs_issue": spy_adj_close / inception_spy_adj_close,
                "coupon_rate_per_obs": fair_coupon_rate,
                "ki_hit_so_far": ki_hit,
                "ki_hit_date": None if ki_hit_date is None else ki_hit_date.date(),
                "remaining_obs": remaining_obs,
                "mtm": mtm,
                "stderr": stderr,
                "delta": delta,
                "gamma": gamma,
                "vega_total": vega_total,
                "rho_total": rho_total,
                "autocall_prob": ac_prob,
                "ki_prob": ki_prob,
                "dt_years_from_prev": dt_years,
                "hedge_units_before": hedge_units_before,
                "target_hedge_units": target_hedge_units,
                "trade_units": trade_units,
                "trade_cashflow": trade_cashflow,
                "cash_before_rehedge": cash_before,
                "cash_after_rehedge": cash_after,
                "spot_pnl_from_prev": spot_pnl,
                "cash_carry_from_prev": cash_carry,
                "strategy_pnl_from_prev": strategy_pnl,
                "portfolio_before_rehedge": portfolio_before_rehedge,
                "portfolio_after_rehedge": portfolio_after_rehedge,
                "replication_error": replication_error,
                "note_pnl_from_prev": note_pnl,
                "note_pnl_minus_strategy_pnl": note_pnl_minus_strategy,
                "cumulative_note_pnl": cumulative_note_pnl,
                "cumulative_strategy_pnl": cumulative_strategy_pnl,
                "cumulative_replication_error": cumulative_replication_error,
            }
        )

        if np.isnan(mtm0):
            mtm0 = mtm
        prev_spot = spot
        prev_date = current_date

        if status in {"autocalled", "matured"}:
            break

    out_df = pd.DataFrame(rows)
    summary_df = build_summary_sheet(out_df)
    normalized_df = build_spot_normalized_sheet(out_df)
    date_tag = issue_date.strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"autocall_backtest_{date_tag}.xlsx"
    sheets = [
        ("Backtest", out_df),
        ("Summary", summary_df),
        ("SpotNormalized", normalized_df),
    ]
    try:
        write_simple_xlsx(out_path, sheets)
    except PermissionError:
        stamp = pd.Timestamp.now().strftime("%H%M%S")
        out_path = OUTPUT_DIR / f"autocall_backtest_{date_tag}_{stamp}.xlsx"
        write_simple_xlsx(out_path, sheets)

    print("=" * 72)
    print("AUTOCALL BACKTEST EXAMPLE")
    print("=" * 72)
    print(f"Issue date:           {issue_date.date()}")
    print(f"Substeps/interval:    {args.substeps_per_interval}")
    print(f"Inception spot:       {inception_spot:.4f}")
    print(f"Inception SPY adj:    {inception_spy_adj_close:.4f}")
    print(f"Target inception px:  {args.target_price:.6f}")
    print(f"Fitted inception px:  {fitted_issue_price:.6f}")
    print(f"Coupon rate / obs:    {fair_coupon_rate:.4%}")
    print(f"Rows written:         {len(out_df)}")
    print(f"Output:               {out_path}")
    print()
    print(out_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
