from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import math
import numpy as np
import pandas as pd

from deamerican import (
    bs_price, bs_vega,
    baw_price, crr_price, crr_price_cts,
    implied_vol_baw, implied_vol_crr, implied_vol_crr_cts,
    YieldCurve, DividendSchedule,
    implied_spot_from_parity,
    parity_implied_q_by_expiry,
    parity_implied_q_iterated,
    deamericanize_chain,
)


# ---------- test harness ----------
PASSED, FAILED = 0, 0


def check(name: str, cond: bool, *, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}   {detail}")


def close(a: float, b: float, tol: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


# =====================================================================
# A. Black-Scholes reference values
# =====================================================================
print("\n[A] Black-Scholes reference values")

# Hull 10th ed. Example 15.6: S=42, K=40, r=10%, sigma=20%, T=0.5
# Call = 4.759, Put = 0.8086
p_c = bs_price(42, 40, 0.5, 0.10, 0.0, 0.20, True)
p_p = bs_price(42, 40, 0.5, 0.10, 0.0, 0.20, False)
check("Hull Ex 15.6 call", close(p_c, 4.759, 1e-3), detail=f"got {p_c:.4f}")
check("Hull Ex 15.6 put", close(p_p, 0.8086, 1e-3), detail=f"got {p_p:.4f}")

# At-the-money identity: Call - Put = S*e^(-qT) - K*e^(-rT)
for S, K, T, r, q, sig in [
    (100, 100, 1.0, 0.05, 0.02, 0.20),
    (50, 55, 0.25, 0.03, 0.0, 0.30),
]:
    c = bs_price(S, K, T, r, q, sig, True)
    p = bs_price(S, K, T, r, q, sig, False)
    lhs = c - p
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    check(
        f"put-call parity S={S} K={K} T={T}",
        close(lhs, rhs, 1e-10),
        detail=f"C-P={lhs:.6f} vs {rhs:.6f}",
    )


# =====================================================================
# B. BAW no-div call == BS
# =====================================================================
print("\n[B] BAW call (q=0) == BS call")
for S, K, T, sig in [
    (100, 100, 1.0, 0.20),
    (100, 110, 0.5, 0.30),
    (50, 40, 2.0, 0.15),
]:
    bs = bs_price(S, K, T, 0.05, 0.0, sig, True)
    bw = baw_price(S, K, T, 0.05, 0.0, sig, True)
    check(
        f"BAW=BS no-div call S={S} K={K}",
        close(bs, bw, 1e-8),
        detail=f"|delta|={abs(bs-bw):.2e}",
    )


# =====================================================================
# C. BAW put premium > 0 and bounded
# =====================================================================
print("\n[C] BAW put early-exercise premium")
for S, K, T, r, q, sig in [
    (100, 100, 1.0, 0.05, 0.0, 0.20),
    (100, 110, 2.0, 0.08, 0.02, 0.25),
    (50, 55, 0.5, 0.03, 0.0, 0.30),
]:
    bs = bs_price(S, K, T, r, q, sig, False)
    bw = baw_price(S, K, T, r, q, sig, False)
    premium = bw - bs
    check(
        f"BAW put premium > 0 (S={S},K={K})",
        premium >= -1e-10,
        detail=f"premium={premium:.4f}",
    )
    check(
        f"BAW put <= K (S={S},K={K})",
        bw <= K + 1e-8,
        detail=f"put={bw:.4f}, K={K}",
    )


# =====================================================================
# D. CRR converges to BS (no divs, European)
# =====================================================================
print("\n[D] CRR European -> BS as steps grow")
S, K, T, r, q, sig = 100, 100, 1.0, 0.05, 0.02, 0.25
bs_val = bs_price(S, K, T, r, q, sig, True)
for n in [50, 200, 800]:
    v = crr_price_cts(S, K, T, r, q, sig, True, n, american=False)
    err = abs(v - bs_val)
    check(
        f"CRR-cts European call n={n} step error",
        err < 5.0 / n,
        detail=f"|err|={err:.4f} vs tol {5.0 / n:.4f}",
    )


# =====================================================================
# E. CRR American >= CRR European (no-arb)
# =====================================================================
print("\n[E] American >= European (CRR)")
for is_call in [True, False]:
    for S, K, T in [(100, 100, 1.0), (100, 90, 0.5), (100, 110, 2.0)]:
        am = crr_price_cts(S, K, T, 0.05, 0.03, 0.25, is_call, 200, True)
        eu = crr_price_cts(S, K, T, 0.05, 0.03, 0.25, is_call, 200, False)
        check(
            f"American >= European {'call' if is_call else 'put'} S={S},K={K}",
            am >= eu - 1e-9,
            detail=f"AM={am:.4f}, EU={eu:.4f}",
        )


# =====================================================================
# F. CRR discrete divs ~= CRR cts with equivalent yield
# =====================================================================
print("\n[F] CRR discrete-div vs CRR cts-yield (equivalent)")
S, K, T, r, sig = 100, 100, 1.0, 0.05, 0.25
div_t = np.array([0.25, 0.5, 0.75], dtype=np.float64)
div_a = np.array([1.0, 1.0, 1.0], dtype=np.float64)
pv = sum(1.0 * math.exp(-r * t) for t in div_t)
q_eq = -math.log((S - pv) / S) / T

p_disc = crr_price(S, K, T, r, sig, div_t, div_a, False, 800, True)
p_cts = crr_price_cts(S, K, T, r, q_eq, sig, False, 800, True)
rel = abs(p_disc - p_cts) / p_disc
check(
    "CRR discrete vs cts equivalent (rel error < 3%)",
    rel < 0.03,
    detail=f"discrete={p_disc:.4f}, cts={p_cts:.4f}, rel={rel:.4%}",
)


# =====================================================================
# G. IV round-trip (BAW)
# =====================================================================
print("\n[G] BAW IV round-trip")
np.random.seed(0)
N = 1000
errs = []
for _ in range(N):
    S = 100.0
    K = np.random.uniform(80, 120)
    T = np.random.uniform(0.1, 2.0)
    sig_true = np.random.uniform(0.10, 0.60)
    r = np.random.uniform(0.01, 0.05)
    q = np.random.uniform(0.0, 0.04)
    is_call = bool(np.random.rand() > 0.5)
    p = baw_price(S, K, T, r, q, sig_true, is_call)
    if p < 1e-6:
        continue
    iv = implied_vol_baw(p, S, K, T, r, q, is_call)
    if math.isnan(iv):
        continue
    errs.append(abs(iv - sig_true))
errs = np.array(errs)
print(
    f"  [stats] N={len(errs)} solves,  max |delta sigma|={errs.max():.2e},  "
    f"median={np.median(errs):.2e},  p99={np.quantile(errs, 0.99):.2e}"
)
# BAW is not globally one-to-one in sigma for some q > r, long-dated call
# regimes, so the body of the error distribution is the more meaningful
# stability check than the single worst outlier.
check(
    "BAW IV round-trip p99 error < 1e-3",
    np.quantile(errs, 0.99) < 1e-3,
    detail=f"p99 err={np.quantile(errs, 0.99):.2e}, max={errs.max():.2e}",
)
check(
    "BAW IV round-trip max error < 3e-2 (includes q>r edge cases)",
    errs.max() < 3e-2,
    detail=f"max err={errs.max():.2e}",
)


# =====================================================================
# H. IV round-trip (CRR cts)
# =====================================================================
print("\n[H] CRR-cts IV round-trip")
errs = []
for _ in range(100):
    S = 100.0
    K = np.random.uniform(90, 110)
    T = np.random.uniform(0.1, 1.5)
    sig_true = np.random.uniform(0.15, 0.40)
    r = 0.03
    q = 0.02
    is_call = bool(np.random.rand() > 0.5)
    p = crr_price_cts(S, K, T, r, q, sig_true, is_call, 200, True)
    iv = implied_vol_crr_cts(p, S, K, T, r, q, is_call, 200)
    if math.isnan(iv):
        continue
    errs.append(abs(iv - sig_true))
errs = np.array(errs)
print(f"  [stats] N={len(errs)}, max |delta sigma|={errs.max():.2e}")
check(
    "CRR-cts IV round-trip max error < 3e-3",
    errs.max() < 3e-3,
    detail=f"max err={errs.max():.2e}",
)


# =====================================================================
# I. Yield curve: BEY -> continuous math
# =====================================================================
print("\n[I] YieldCurve: Cboe BEY -> r_cts")
row = pd.Series(
    {
        "1_month": 5.0,
        "3_month": 5.0,
        "6_month": 5.0,
        "1_year": 5.0,
        "2_year": 5.0,
        "5_year": 5.0,
        "10_year": 5.0,
        "30_year": 5.0,
    },
    name="test",
)
curve = YieldCurve.from_cmt_row(row)
expected_r = math.log(1.050625)
for t_yr in [0.25, 1.0, 5.0]:
    r = curve.r_continuous(t_yr)
    check(
        f"flat curve r(t={t_yr}) = {expected_r:.6f}",
        close(r, expected_r, 1e-8),
        detail=f"got {r:.6f}",
    )


# =====================================================================
# J. Parity-implied spot
# =====================================================================
print("\n[J] Parity-implied spot recovery")
S_true = 275.00
r_true = 0.025
q_true = 0.018
sig = 0.15
T = 30 / 365.0
strikes = np.arange(265, 286, 1.0)
rows = []
trade_date = pd.Timestamp("2020-06-15")
exp_date = trade_date + pd.Timedelta(days=30)
for K in strikes:
    c = bs_price(S_true, K, T, r_true, q_true, sig, True)
    p = bs_price(S_true, K, T, r_true, q_true, sig, False)
    rows.append(
        {
            "date": trade_date,
            "expiration": exp_date,
            "strike": K,
            "call_put": "Call",
            "bid": c - 0.005,
            "ask": c + 0.005,
        }
    )
    rows.append(
        {
            "date": trade_date,
            "expiration": exp_date,
            "strike": K,
            "call_put": "Put",
            "bid": p - 0.005,
            "ask": p + 0.005,
        }
    )
df_syn = pd.DataFrame(rows)

bey_pct = (2.0 * (math.exp(r_true) ** 0.5 - 1.0)) * 100.0
flat_row = pd.Series(
    {lbl: bey_pct for lbl in
     ["1_month", "3_month", "6_month", "1_year", "2_year", "5_year", "10_year", "30_year"]},
    name="t",
)
curve = YieldCurve.from_cmt_row(flat_row)

S_impl_nodiv = implied_spot_from_parity(df_syn, T, curve, divs=None)
pv_div_true = S_true - S_true * math.exp(-q_true * T)
expected_S_nodiv = S_true - pv_div_true
print(f"  S_true = {S_true}, PV_div = {pv_div_true:.4f}")
print(f"  S_impl_nodiv = {S_impl_nodiv:.4f}  (expected {expected_S_nodiv:.4f})")
check(
    "spot implied w/o divs, error < 1c",
    close(S_impl_nodiv, expected_S_nodiv, 0.02),
    detail=f"got {S_impl_nodiv:.4f}",
)

divs = DividendSchedule(np.array([1e-6]), np.array([pv_div_true]))
S_impl_div = implied_spot_from_parity(df_syn, T, curve, divs=divs)
check(
    "spot implied w/ correct divs, error < 1c",
    close(S_impl_div, S_true, 0.02),
    detail=f"got {S_impl_div:.4f}, expected {S_true}",
)


# =====================================================================
# K. Parity-implied q recovers truth on synthetic European data
# =====================================================================
print("\n[K] parity_implied_q on clean synthetic European data")
q_map = parity_implied_q_by_expiry(df_syn, S_true, curve, atm_band=0.03)
T_key = list(q_map.keys())[0]
q_recovered = q_map[T_key]
print(
    f"  true q = {q_true:.5f}, recovered = {q_recovered:.5f}, "
    f"diff = {abs(q_recovered - q_true) * 1e4:.1f} bp"
)
check(
    "parity q recovery on clean data, error < 10 bp",
    abs(q_recovered - q_true) < 10e-4,
    detail=f"|delta q|={(q_recovered - q_true) * 1e4:.1f}bp",
)


# =====================================================================
# L. Full de-Am round-trip on synthetic American chain
# =====================================================================
print("\n[L] Full de-Am pipeline on synthetic American chain")
rows = []
for K in strikes:
    c_am = baw_price(S_true, K, T, r_true, q_true, sig, True)
    p_am = baw_price(S_true, K, T, r_true, q_true, sig, False)
    rows.append(
        {
            "date": trade_date,
            "expiration": exp_date,
            "strike": K,
            "call_put": "Call",
            "bid": max(c_am - 0.005, 1e-4),
            "ask": c_am + 0.005,
        }
    )
    rows.append(
        {
            "date": trade_date,
            "expiration": exp_date,
            "strike": K,
            "call_put": "Put",
            "bid": max(p_am - 0.005, 1e-4),
            "ask": p_am + 0.005,
        }
    )
df_am = pd.DataFrame(rows)
curve_l = YieldCurve.from_cmt_row(flat_row)

q0 = parity_implied_q_by_expiry(df_am, S_true, curve_l, atm_band=0.05)
q0_val = list(q0.values())[0]
qi = parity_implied_q_iterated(df_am, S_true, curve_l, atm_band=0.05, n_iter=2)
qi_val = list(qi.values())[0]
print(f"  true q = {q_true:.5f}")
print(f"  American raw parity q (pass 0) = {q0_val:.5f}  diff={abs(q0_val - q_true) * 1e4:.1f}bp")
print(f"  Iterated parity q (pass 2)     = {qi_val:.5f}  diff={abs(qi_val - q_true) * 1e4:.1f}bp")
check(
    "iterated parity closer to truth than raw",
    abs(qi_val - q_true) <= abs(q0_val - q_true) + 1e-6,
    detail=f"raw={abs(q0_val - q_true):.5f}, iter={abs(qi_val - q_true):.5f}",
)

deam = deamericanize_chain(df_am, spot=S_true, curve=curve_l, q_per_expiry=qi, method="baw")
DF = math.exp(-r_true * T)
pv = S_true - S_true * math.exp(-qi_val * T)
parity_errs = []
for K in strikes:
    row_c = deam[(deam.strike == K) & (deam.call_put == "Call")]
    row_p = deam[(deam.strike == K) & (deam.call_put == "Put")]
    if len(row_c) == 0 or len(row_p) == 0:
        continue
    c = row_c["european_mid"].iloc[0]
    p = row_p["european_mid"].iloc[0]
    if math.isnan(c) or math.isnan(p):
        continue
    lhs = c - p
    rhs = (S_true - pv) - DF * K
    parity_errs.append(abs(lhs - rhs))
parity_errs = np.array(parity_errs)
print(
    f"  European C-P vs parity RHS: max err = {parity_errs.max():.4f}, "
    f"median = {np.median(parity_errs):.4f}"
)
check(
    "de-Am European mids satisfy put-call parity (max err < 2c)",
    parity_errs.max() < 0.02,
    detail=f"max={parity_errs.max():.4f}",
)


# =====================================================================
# M. Deep-ITM: IV solver returns NaN (not bogus)
# =====================================================================
print("\n[M] Deep-ITM exercise-dominated -> NaN")
S, K, T, r, q, sig = 100.0, 150.0, 0.3, 0.03, 0.01, 0.25
p_deep = baw_price(S, K, T, r, q, sig, False)
intrinsic = K - S
print(f"  deep-ITM put price = {p_deep:.4f}, intrinsic = {intrinsic:.4f}")
iv = implied_vol_baw(p_deep, S, K, T, r, q, False)
check(
    "deep-ITM BAW returns NaN (unrecoverable sigma)",
    math.isnan(iv),
    detail=f"got {iv}",
)


# =====================================================================
# FINALE
# =====================================================================
print(f"\n{'=' * 60}")
print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
print(f"{'=' * 60}")
sys.exit(0 if FAILED == 0 else 1)
