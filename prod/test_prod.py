"""Sub-stepped autocall pricer: identity, convergence and Greeks tests.

Sweeps `substeps_per_interval`. Under piecewise-constant σ per observation
interval, sub-stepping must NOT change the price -- the only visible effect
is finer continuous-KI monitoring (and slightly different Sobol convergence
because the stream dimension grows).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import autocall_pricer_lv as ap

np.set_printoptions(precision=5, suppress=True)

print("=" * 80)
print("PROD (LOCAL-VOL SUB-STEPPED) PRICER BENCHMARK")
print("=" * 80)
print(f"AVX-512 available: {ap.has_avx512()}")
print(f"AVX2    available: {ap.has_avx2()}")
print()

# ---------- Product spec (same 5Y quarterly autocallable) ----------
S0 = 100.0; notional = 100.0; strike = 1.0
ki_barrier = 0.60; continuous_ki = True
n_obs = 20
obs_times   = np.array([(i + 1) * 0.25 for i in range(n_obs)], dtype=np.float64)
ac_barriers = np.full(n_obs, 1.00, dtype=np.float64)
coupons     = np.array([(i + 1) * 0.02 for i in range(n_obs)], dtype=np.float64)
fwd_vols    = np.full(n_obs, 0.25, dtype=np.float64)
fwd_rates   = np.full(n_obs, 0.03, dtype=np.float64)
fwd_divs    = np.full(n_obs, 0.01, dtype=np.float64)

spec = ap.make_single_spec(
    S0, notional, strike, ki_barrier, continuous_ki,
    obs_times, ac_barriers, coupons, fwd_vols, fwd_rates, fwd_divs,
)

# ---------- Sub-step sweep ----------
print("Sub-step sweep @ 200k paths (Sobol+BB+AVX-512+CV):")
print(f"{'K':>3} {'sobol_dim':>9} {'Price':>10} {'StdErr':>9} {'Time(ms)':>9}  {'Method':<40}")
print("-" * 90)

results_by_K = {}
for K in [1, 2, 4, 8]:
    t0 = time.perf_counter()
    # Sobol max dim is 200; with K=8 and n_obs=20, continuous_ki -> 2*20*8 = 320 (falls back to Philox).
    r = ap.price_single_asset(
        spec, 200_000, seed=42,
        use_sobol=True, use_brownian_bridge=True, simd_width=-1,
        use_control_variate=True, compute_greeks=False,
        n_threads=0, substeps_per_interval=K,
    )
    wall = (time.perf_counter() - t0) * 1000
    results_by_K[K] = r
    sobol_dim = 2 * n_obs * K if continuous_ki else n_obs * K
    print(f"{K:>3} {sobol_dim:>9} {r.cv_price:>10.5f} {r.cv_stderr:>9.5f} "
          f"{r.elapsed_ms:>9.1f}  {r.method:<40}")

# Expect prices across K to match within MC noise (piecewise-const σ, same
# product). Check:
ref = results_by_K[1].cv_price
ref_se = results_by_K[1].cv_stderr
print(f"\nReference (K=1): {ref:.5f} ± {ref_se:.5f}")
max_err = 0.0
for K, r in results_by_K.items():
    if K == 1:
        continue
    err = r.cv_price - ref
    joint_se = np.sqrt(ref_se ** 2 + r.cv_stderr ** 2)
    verdict = 'OK' if abs(err) < 3 * joint_se else 'FAIL'
    print(f"  K={K}: err={err:+.5f}  (joint stderr = {joint_se:.5f})  {verdict}")
    max_err = max(max_err, abs(err))

# ---------- Greeks (LR + pathwise) with K=4 ----------
print("\n" + "=" * 80)
print("GREEKS WITH K=4 (LR Delta/Gamma uses first sub-step)")
print("=" * 80)
t0 = time.perf_counter()
g = ap.price_single_asset(
    spec, 200_000, seed=42,
    use_sobol=True, use_brownian_bridge=True, simd_width=-1,
    use_control_variate=True, compute_greeks=True,
    n_threads=0, substeps_per_interval=4,
)
dt = (time.perf_counter() - t0) * 1000
print(f"Price:       {g.cv_price:.5f} (±{g.cv_stderr:.5f})")
print(f"Time:        {g.elapsed_ms:.1f} ms for price + all Greeks")
print(f"Delta (LR):  {g.delta:>+10.5f}")
print(f"Gamma (LR):  {g.gamma:>+10.5f}")
print(f"Vega  tot:   {g.vega_total:>+10.5f}  (per 1%: {g.vega_total / 100:+.5f})")
print(f"Rho   tot:   {g.rho_total:>+10.5f}   (per 1bp: {g.rho_total / 10000:+.5f})")

# Delta finite-difference cross-check
eps = 0.01
spec_up = ap.make_single_spec(
    S0 * (1 + eps), notional, strike / (1 + eps), ki_barrier / (1 + eps), continuous_ki,
    obs_times, ac_barriers / (1 + eps), coupons, fwd_vols, fwd_rates, fwd_divs,
)
spec_dn = ap.make_single_spec(
    S0 * (1 - eps), notional, strike / (1 - eps), ki_barrier / (1 - eps), continuous_ki,
    obs_times, ac_barriers / (1 - eps), coupons, fwd_vols, fwd_rates, fwd_divs,
)
r_up = ap.price_single_asset(spec_up, 500_000, seed=42, use_sobol=True,
    use_brownian_bridge=True, use_control_variate=True, substeps_per_interval=4)
r_dn = ap.price_single_asset(spec_dn, 500_000, seed=42, use_sobol=True,
    use_brownian_bridge=True, use_control_variate=True, substeps_per_interval=4)
delta_fd = (r_up.cv_price - r_dn.cv_price) / (2 * S0 * eps)
print(f"\nDelta LR : {g.delta:.5f}")
print(f"Delta FD : {delta_fd:.5f}")
print(f"|diff|   : {abs(g.delta - delta_fd):.5f}")

# ---------- Continuous-KI sensitivity to K ----------
# Increasing K gives finer barrier monitoring. For a deep-KI product this
# raises the KI hit probability slightly.
print("\n" + "=" * 80)
print("CONTINUOUS-KI probability vs K")
print("=" * 80)
print(f"{'K':>3} {'ki_prob':>10} {'price':>10}")
for K in [1, 2, 4, 8]:
    r = results_by_K[K]
    print(f"{K:>3} {r.ki_prob:>10.4%} {r.cv_price:>10.5f}")

print("\nDone.")
