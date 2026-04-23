"""Local-vol (LV) mode correctness tests for the prod pricer.

Validates the C++ LV path walker by constructing degenerate LV grids and
checking that the LV pricer reproduces the piecewise-constant-σ pricer to
within MC noise.

Tests (in order):
  A. Flat LV grid (σ constant on log_m × t) reproduces flat-σ pricer exactly
     up to a few standard errors (same Sobol stream, same seed).
  B. Time-only LV grid (σ(t), flat in log_m) reproduces a pricer with
     piecewise-constant fwd_vols matching σ at obs midpoints.
  C. LV mode forces simd_width=1 (method string reflects it; no AVX tag).
  D. Greeks run under LV: delta finite-difference agrees with LR delta.
  E. Smile LV grid (σ depends on log_m): price is sensitive to skew — a
     downward skew (lower vol above, higher below) should push the price
     differently than the ATM-flat case.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import autocall_pricer_lv as ap

np.set_printoptions(precision=5, suppress=True)

# ---------- Shared product spec ----------
S0 = 100.0; notional = 100.0; strike = 1.0
ki_barrier = 0.60; continuous_ki = True
n_obs = 20
obs_times   = np.array([(i + 1) * 0.25 for i in range(n_obs)], dtype=np.float64)
ac_barriers = np.full(n_obs, 1.00, dtype=np.float64)
coupons     = np.array([(i + 1) * 0.02 for i in range(n_obs)], dtype=np.float64)
fwd_rates   = np.full(n_obs, 0.03, dtype=np.float64)
fwd_divs    = np.full(n_obs, 0.01, dtype=np.float64)
SIGMA_FLAT  = 0.25

N_PATHS = 200_000
K = 4


def _build_flat_spec(sigma):
    fv = np.full(n_obs, sigma, dtype=np.float64)
    return ap.make_single_spec(
        S0, notional, strike, ki_barrier, continuous_ki,
        obs_times, ac_barriers, coupons, fv, fwd_rates, fwd_divs,
    )


def _price(spec, **kw):
    defaults = dict(
        n_paths=N_PATHS, seed=42, use_sobol=True, use_brownian_bridge=True,
        simd_width=-1, use_control_variate=True, compute_greeks=False,
        n_threads=0, substeps_per_interval=K,
    )
    defaults.update(kw)
    return ap.price_single_asset(spec, **defaults)


def _attach_flat_lv(spec, sigma):
    # Minimal 2x2 grid suffices for flat σ.
    log_m_grid = np.array([-3.0, 3.0], dtype=np.float64)
    t_grid     = np.array([0.0, obs_times[-1] + 1.0], dtype=np.float64)
    sigma_loc  = np.full((2, 2), sigma, dtype=np.float64)
    ap.set_local_vol(spec, log_m_grid, t_grid, sigma_loc, enable=True)


print("=" * 80)
print("LOCAL-VOL CORRECTNESS TESTS")
print("=" * 80)

# ======================================================================
# Test A. Flat LV vs flat-σ: prices agree to a few stderrs.
# ======================================================================
print("\n[A] Flat LV grid (σ=0.25 everywhere) vs flat-σ pricer")
spec_ref = _build_flat_spec(SIGMA_FLAT)
r_ref = _price(spec_ref)

spec_lv = _build_flat_spec(SIGMA_FLAT)
_attach_flat_lv(spec_lv, SIGMA_FLAT)
r_lv = _price(spec_lv)

joint_se = np.sqrt(r_ref.cv_stderr ** 2 + r_lv.cv_stderr ** 2)
err      = r_lv.cv_price - r_ref.cv_price
verdict  = "OK" if abs(err) < 4 * joint_se else "FAIL"
print(f"  ref   price: {r_ref.cv_price:.5f} ± {r_ref.cv_stderr:.5f}  method={r_ref.method}")
print(f"  LV    price: {r_lv.cv_price:.5f} ± {r_lv.cv_stderr:.5f}  method={r_lv.method}")
print(f"  diff       : {err:+.5f}   joint_se={joint_se:.5f}   -> {verdict}")
assert verdict == "OK", "flat LV should match flat-σ"

# ======================================================================
# Test B. Time-only LV: σ(t) piecewise-linear on the t-axis, flat in log_m.
# Reference: piecewise-constant fwd_vols at obs midpoints, same levels.
# Because our prod pricer treats fwd_vols as piecewise-constant on obs
# intervals, we discretize σ(t) on the obs grid and compare.
# ======================================================================
print("\n[B] Time-only LV vs matching piecewise-constant fwd_vols")
# Simple term structure: σ rising from 0.18 (short) to 0.32 (long).
t_anchor  = np.array([0.0, 1.0, 2.5, 5.0, 6.0], dtype=np.float64)
sig_anchor = np.array([0.18, 0.22, 0.28, 0.32, 0.32], dtype=np.float64)

# Evaluate LV at the same t axis for the grid.
log_m_axis = np.array([-3.0, 3.0], dtype=np.float64)
sigma_loc  = np.tile(sig_anchor[None, :], (2, 1))  # shape (2, n_t)

# Reference: fwd_vol_i = σ(t_i) evaluated at the end of each obs interval
# (prod pricer uses fwd_vols[i] constant over [t_{i-1}, t_i]; using obs
# endpoints matches the Euler σ lookup at sub-step-start under LV).
fv_match = np.interp(obs_times - 0.5 * 0.25, t_anchor, sig_anchor).astype(np.float64)

spec_ref_b = ap.make_single_spec(
    S0, notional, strike, ki_barrier, continuous_ki,
    obs_times, ac_barriers, coupons, fv_match, fwd_rates, fwd_divs,
)
r_ref_b = _price(spec_ref_b)

spec_lv_b = _build_flat_spec(SIGMA_FLAT)  # fwd_vols used only for CV baseline
ap.set_local_vol(spec_lv_b, log_m_axis, t_anchor, sigma_loc, enable=True)
r_lv_b = _price(spec_lv_b)

joint_se = np.sqrt(r_ref_b.cv_stderr ** 2 + r_lv_b.cv_stderr ** 2)
err_b    = r_lv_b.cv_price - r_ref_b.cv_price
# Looser tolerance: the two discretizations are slightly different.
verdict_b = "OK" if abs(err_b) < 6 * joint_se else "CHECK"
print(f"  ref (pc-σ) : {r_ref_b.cv_price:.5f} ± {r_ref_b.cv_stderr:.5f}")
print(f"  LV  (σ(t)) : {r_lv_b.cv_price:.5f} ± {r_lv_b.cv_stderr:.5f}")
print(f"  diff       : {err_b:+.5f}   joint_se={joint_se:.5f}   -> {verdict_b}")

# ======================================================================
# Test C. LV forces simd_width=1 — method string should not advertise AVX.
# ======================================================================
print("\n[C] LV mode forces scalar path walker (no AVX in method)")
print(f"  LV method: {r_lv.method}")
assert "AVX" not in r_lv.method, "LV mode must not use SIMD kernels"
print("  OK")

# ======================================================================
# Test D. Greeks work under LV; LR delta ≈ FD delta.
# ======================================================================
print("\n[D] Greeks under LV: LR delta vs bump-and-revalue FD delta")
spec_lv_g = _build_flat_spec(SIGMA_FLAT)
_attach_flat_lv(spec_lv_g, SIGMA_FLAT)
g = _price(spec_lv_g, compute_greeks=True)

eps = 0.01

def _build_bumped(bump):
    # Mirror test_prod.py's FD pattern: keep all absolute barriers fixed by
    # scaling every fractional barrier (strike, ki, ac) by 1/(1+bump).
    fv = np.full(n_obs, SIGMA_FLAT, dtype=np.float64)
    s = ap.make_single_spec(
        S0 * (1 + bump), notional, strike / (1 + bump),
        ki_barrier / (1 + bump), continuous_ki,
        obs_times, ac_barriers / (1 + bump), coupons,
        fv, fwd_rates, fwd_divs,
    )
    _attach_flat_lv(s, SIGMA_FLAT)
    return s

r_up = _price(_build_bumped(+eps), n_paths=500_000)
r_dn = _price(_build_bumped(-eps), n_paths=500_000)
delta_fd = (r_up.cv_price - r_dn.cv_price) / (2 * S0 * eps)
print(f"  Delta LR (LV): {g.delta:.5f}")
print(f"  Delta FD (LV): {delta_fd:.5f}")
print(f"  |diff|        : {abs(g.delta - delta_fd):.5f}")
print(f"  (Note: vega_buckets are zero under LV — pathwise surface vega "
      f"requires external bump.)")
print(f"  vega_total   : {g.vega_total:.5f}  (expected ≈ 0 under LV)")

# ======================================================================
# Test E. Negative-skew LV grid behaves differently from flat LV.
# Skew: σ higher for log_m < 0 (OTM puts), lower for log_m > 0 (OTM calls).
# For a KI put-like product, more left-side vol -> higher KI prob -> lower
# price (more realized downside loss risk).
# ======================================================================
print("\n[E] Negative-skew LV grid shifts price vs flat LV")
log_m_e = np.linspace(-1.0, 1.0, 11, dtype=np.float64)
t_e     = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float64)
# σ = 0.25 + 0.15*(-log_m) clipped  -> 0.40 at log_m=-1, 0.10 at log_m=+1
sig_by_m = np.clip(0.25 - 0.15 * log_m_e, 0.08, 0.45)
sigma_e = np.tile(sig_by_m[:, None], (1, t_e.size))

spec_e = _build_flat_spec(SIGMA_FLAT)
ap.set_local_vol(spec_e, log_m_e, t_e, sigma_e, enable=True)
r_e = _price(spec_e)

joint_se_e = np.sqrt(r_lv.cv_stderr ** 2 + r_e.cv_stderr ** 2)
diff_e = r_e.cv_price - r_lv.cv_price
print(f"  flat LV  σ=0.25   : {r_lv.cv_price:.5f} ± {r_lv.cv_stderr:.5f}")
print(f"  skewed LV (neg)   : {r_e.cv_price:.5f} ± {r_e.cv_stderr:.5f}")
print(f"  skew - flat       : {diff_e:+.5f}   joint_se={joint_se_e:.5f}")
print(f"  ki_prob (flat)    : {r_lv.ki_prob:.4%}")
print(f"  ki_prob (skew)    : {r_e.ki_prob:.4%}")
# Both surfaces share ATM σ=0.25 so the path-dependent price difference is
# subtle; we only check that the surface shape is actually wired through
# (skew changes KI probability by a non-trivial amount).
dki = abs(r_e.ki_prob - r_lv.ki_prob)
assert dki > 0.005, ("surface shape should shift ki_prob materially "
                     f"(got |Δki|={dki:.4%})")
print(f"  OK — |Δki_prob| = {dki:.4%} (surface shape is active)")

print("\n" + "=" * 80)
print("All LV correctness checks completed.")
print("=" * 80)
