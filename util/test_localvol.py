"""
test_localvol.py
================
Unit tests for util/localvol.py.

The big three checks:

    [A]  Flat implied-vol surface -> flat local-vol surface.
         This is the most important correctness test: if sigma_imp(K,T) = sigma0
         everywhere, Dupire must return sigma_loc(K,T) == sigma0 at every node
         (modulo the wing regularisation).

    [B]  Term-structure only (no skew) -> sigma_loc(T) reproduces forward vol.
         sigma_imp(K,T) = sigma(T) with sigma(T) piecewise-constant gives
         sigma_loc that matches the forward vol on each interval.

    [C]  Synthetic smile with mild skew -> finite, bounded surface;
         no NaNs, no sub-floor or super-cap values.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import math

import numpy as np
import pandas as pd

from deamerican import YieldCurve
from localvol import (
    IVSmile,
    ImpliedVolSurface,
    LocalVolSurface,
    build_local_vol_grid_from_chain,
)


PASSED, FAILED = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}   {detail}")


# --------------------- flat curve for all tests ------------------------

def flat_curve(r_dec: float) -> YieldCurve:
    bey_pct = 2.0 * (math.exp(r_dec / 2.0) - 1.0) * 100.0  # BEY such that r_cts ~ r
    row = pd.Series(
        {lbl: bey_pct for lbl in
         ["1_month", "3_month", "6_month", "1_year", "2_year",
          "5_year", "10_year", "30_year"]},
        name="t",
    )
    return YieldCurve.from_cmt_row(row)


def flat_q(q_dec: float):
    def f(T):
        return q_dec
    return f


# =====================================================================
# A. Flat IV surface -> flat local vol
# =====================================================================
print("\n[A] Flat implied vol -> flat local vol")

sigma0 = 0.20
r0 = 0.03
q0 = 0.015
expiries = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
# Build a deam-style DataFrame with iv == sigma0 at many strikes / expiries.
rows = []
spot = 100.0
strikes = np.linspace(60, 160, 21)
trade_date = pd.Timestamp("2024-01-02")
for T in expiries:
    exp = trade_date + pd.Timedelta(days=int(round(T * 365)))
    for K in strikes:
        for cp in ("Call", "Put"):
            rows.append(dict(
                date=trade_date, expiration=exp,
                T_years=float(T), strike=float(K), call_put=cp,
                american_iv=sigma0,
            ))
flat_df = pd.DataFrame(rows)
curve_A = flat_curve(r0)
q_fn_A = flat_q(q0)

iv_A = ImpliedVolSurface.from_chain(flat_df, spot=spot, curve=curve_A, q_provider=q_fn_A)
lv_A = LocalVolSurface.from_iv_surface(
    iv_A, curve=curve_A, q_provider=q_fn_A,
    log_m_bounds=(-0.5, 0.5), n_m=101, t_max=5.0, n_t=81,
)
# Inspect central region (avoid the very edge where finite-diff is noisier)
i_lo = 20
i_hi = lv_A.sigma_loc.shape[0] - 20
j_lo = 5
j_hi = lv_A.sigma_loc.shape[1] - 5
center = lv_A.sigma_loc[i_lo:i_hi, j_lo:j_hi]
max_err = float(np.abs(center - sigma0).max())
print(f"  central-block max |sigma_loc - sigma0| = {max_err:.5f} (sigma0={sigma0})")
check("flat IV -> flat LV (central block err < 0.01)", max_err < 0.01,
      detail=f"max err {max_err:.5f}")


# =====================================================================
# B. Term-structure only: sigma_imp(T) piecewise-const -> sigma_loc matches forward vol
# =====================================================================
print("\n[B] Term-structure only -> fwd vol recovered")

# Construct sigma_term(T) that is: 0.15 for T in [0, 1], 0.30 for T in [1, 5].
# Total variance: w(T) = 0.15^2 * T for T <= 1
#                        = 0.15^2 * 1 + 0.30^2 * (T - 1) for T > 1
# Local vol with no skew is the forward vol: 0.15 for T < 1, 0.30 for T > 1.

r0 = 0.03
q0 = 0.015
curve_B = flat_curve(r0)
q_fn_B = flat_q(q0)

def sigma_term(T):
    return 0.15 if T <= 1.0 else 0.30

# Build a flat-across-strike smile per expiry at the prescribed sigma_term.
rows = []
exp_times = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 3.0, 4.0, 5.0]
for T in exp_times:
    exp = trade_date + pd.Timedelta(days=int(round(T * 365)))
    s = sigma_term(T)
    for K in strikes:
        rows.append(dict(date=trade_date, expiration=exp,
                         T_years=float(T), strike=float(K), call_put="Call",
                         american_iv=s))
        rows.append(dict(date=trade_date, expiration=exp,
                         T_years=float(T), strike=float(K), call_put="Put",
                         american_iv=s))
df_B = pd.DataFrame(rows)
iv_B = ImpliedVolSurface.from_chain(df_B, spot=spot, curve=curve_B, q_provider=q_fn_B)
lv_B = LocalVolSurface.from_iv_surface(
    iv_B, curve=curve_B, q_provider=q_fn_B,
    log_m_bounds=(-0.4, 0.4), n_m=81, t_max=5.0, n_t=101,
)
# Probe ATM at T=0.5 and T=3.0
atm_idx = int(np.argmin(np.abs(lv_B.log_m_grid)))
j_short = int(np.argmin(np.abs(lv_B.t_grid - 0.5)))
j_long = int(np.argmin(np.abs(lv_B.t_grid - 3.0)))
lv_short = lv_B.sigma_loc[atm_idx, j_short]
lv_long = lv_B.sigma_loc[atm_idx, j_long]
print(f"  ATM sigma_loc(T=0.5) = {lv_short:.4f}  (expect 0.15)")
print(f"  ATM sigma_loc(T=3.0) = {lv_long:.4f}  (expect 0.30)")
check("short-maturity LV ~ 0.15", abs(lv_short - 0.15) < 0.02,
      detail=f"got {lv_short:.4f}")
check("long-maturity LV ~ 0.30", abs(lv_long - 0.30) < 0.03,
      detail=f"got {lv_long:.4f}")


# =====================================================================
# C. Synthetic smile with skew -> surface is finite, bounded, non-crazy
# =====================================================================
print("\n[C] Synthetic skew -> sane surface")

# IV linear in log-moneyness: iv(K, T) = 0.20 - 0.3 * log(K / F(T))
# Gives ~2% IV change for 6.6% strike change. Realistic for SPY.
r0 = 0.03
q0 = 0.015
curve_C = flat_curve(r0)
q_fn_C = flat_q(q0)

rows = []
strikes_C = np.linspace(60, 160, 31)
exp_times_C = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
for T in exp_times_C:
    exp = trade_date + pd.Timedelta(days=int(round(T * 365)))
    F = spot * math.exp((r0 - q0) * T)
    for K in strikes_C:
        k = math.log(K / F)
        iv = max(0.05, 0.20 - 0.30 * k)
        rows.append(dict(date=trade_date, expiration=exp,
                         T_years=float(T), strike=float(K), call_put="Call",
                         american_iv=iv))
        rows.append(dict(date=trade_date, expiration=exp,
                         T_years=float(T), strike=float(K), call_put="Put",
                         american_iv=iv))
df_C = pd.DataFrame(rows)
iv_C = ImpliedVolSurface.from_chain(df_C, spot=spot, curve=curve_C, q_provider=q_fn_C)
lv_C = LocalVolSurface.from_iv_surface(
    iv_C, curve=curve_C, q_provider=q_fn_C,
    log_m_bounds=(-0.6, 0.6), n_m=121, t_max=5.0, n_t=101,
)
s_all = lv_C.sigma_loc
print(f"  shape = {s_all.shape}")
print(f"  sigma_loc: min={s_all.min():.4f}, median={np.median(s_all):.4f}, max={s_all.max():.4f}")
check("no NaNs", np.isfinite(s_all).all(),
      detail=f"{int((~np.isfinite(s_all)).sum())} NaNs")
check("no sub-0.5% vols", s_all.min() > 0.005,
      detail=f"min={s_all.min():.4f}")
check("no super-200% vols", s_all.max() < 2.01,
      detail=f"max={s_all.max():.4f}")
# Downward skew: ATM < 30% at all T, and 10%-OTP puts (k=-0.1) > ATM
i_atm = int(np.argmin(np.abs(lv_C.log_m_grid)))
i_otm_p = int(np.argmin(np.abs(lv_C.log_m_grid + 0.10)))
atm_avg = float(np.mean(s_all[i_atm, 5:]))
otm_p_avg = float(np.mean(s_all[i_otm_p, 5:]))
print(f"  ATM LV (avg over T) = {atm_avg:.4f}, OTM-put LV (log_m=-0.1) = {otm_p_avg:.4f}")
check("OTM-put LV > ATM LV (skew preserved)", otm_p_avg > atm_avg,
      detail=f"OTP={otm_p_avg:.4f} vs ATM={atm_avg:.4f}")


# =====================================================================
# D. Calendar-arb guard fires when slices have decreasing total var
# =====================================================================
print("\n[D] Calendar-arb guard (cum-max projection)")

# Construct slices where T=1 has sigma=0.30 but T=2 has sigma=0.15 at ATM
# -> raw w(ATM, 1) = 0.09, w(ATM, 2) = 0.045. Violates calendar arb.
rows = []
for T, s in [(0.25, 0.20), (0.5, 0.20), (1.0, 0.30), (2.0, 0.15), (3.0, 0.20)]:
    exp = trade_date + pd.Timedelta(days=int(round(T * 365)))
    for K in strikes:
        rows.append(dict(date=trade_date, expiration=exp,
                         T_years=float(T), strike=float(K), call_put="Call",
                         american_iv=s))
        rows.append(dict(date=trade_date, expiration=exp,
                         T_years=float(T), strike=float(K), call_put="Put",
                         american_iv=s))
df_D = pd.DataFrame(rows)
curve_D = flat_curve(0.03)
q_fn_D = flat_q(0.01)
iv_D = ImpliedVolSurface.from_chain(df_D, spot=spot, curve=curve_D, q_provider=q_fn_D)
lv_D = LocalVolSurface.from_iv_surface(
    iv_D, curve=curve_D, q_provider=q_fn_D,
    log_m_bounds=(-0.4, 0.4), n_m=81, t_max=3.0, n_t=81,
)
# Result should have finite LV everywhere (guard prevented NaNs).
check("calendar-arb-guard keeps LV finite", np.isfinite(lv_D.sigma_loc).all(),
      detail=f"{int((~np.isfinite(lv_D.sigma_loc)).sum())} non-finite")
# LV should be floored, not negative/imaginary.
check("calendar-arb-guard keeps LV >= floor", (lv_D.sigma_loc > 0.005).all(),
      detail=f"min={lv_D.sigma_loc.min():.4f}")


# =====================================================================
# E. One-shot builder works on an actual de-Am chain
# =====================================================================
print("\n[E] End-to-end one-shot builder from flat synthetic de-Am chain")
lv_E = build_local_vol_grid_from_chain(
    flat_df, spot=spot, curve=curve_A, q_provider=q_fn_A,
    log_m_bounds=(-0.5, 0.5), n_m=81, t_max=5.0, n_t=81,
)
check("E: shape matches request", lv_E.sigma_loc.shape == (81, 81),
      detail=f"got {lv_E.sigma_loc.shape}")
check("E: no NaNs", np.isfinite(lv_E.sigma_loc).all())
# Central region close to sigma0 (same check as A)
center = lv_E.sigma_loc[15:-15, 5:-5]
check("E: flat-IV central block err < 0.02", np.abs(center - 0.20).max() < 0.02,
      detail=f"max err {np.abs(center - 0.20).max():.4f}")


# =====================================================================
# FINALE
# =====================================================================
print(f"\n{'=' * 60}")
print(f"  LOCALVOL RESULTS: {PASSED} passed, {FAILED} failed")
print(f"{'=' * 60}")
sys.exit(0 if FAILED == 0 else 1)
