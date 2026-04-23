# Autocallable Pricer — Local-Vol Sub-Stepping

Compiled Monte-Carlo autocallable pricer with **Euler sub-stepping within each
observation interval**. The pybind11 module is named `autocall_pricer_lv`.

## Design highlights

| Item | Value |
|---|---|
| Module name | `autocall_pricer_lv` |
| Path steps per path | `T = N · K` (N = observations, K = sub-steps) |
| Drift / diffusion | piecewise-constant per sub-step |
| Brownian bridge | over the full sub-step grid |
| Continuous-KI check | once per sub-step + BB probability |
| LR Delta/Gamma | uses first sub-step (`ν₁ = σ₁√(dt₁/K)`) |
| Pathwise Vega per bucket | sum of K sub-step sensitivities |
| `substeps_per_interval` | tunable (default 1) |

`substeps_per_interval=1` reduces to a single Euler step per observation.

## Inputs

The forward-vol / forward-rate / forward-div arrays come from
`util/autocall_prep.build_autocall_inputs`, with σ piecewise-constant on each
observation interval.

To plug in a **true local-vol surface σ(S, t)** for smile calibration, replace
the sub-step σ lookup in
[`autocall_pricer_prod.cpp`](autocall_pricer_prod.cpp) (search for `spec.fwd_vols[ii]`
inside `make_one_path`) with a grid lookup. The sub-step grid (`ss[k].mu`,
`ss[k].nu2`) and the SIMD kernel already take arbitrary per-sub-step
parameters — only the Python-side `W_inc` builder needs to change.

## Build

```bash
python prod/setup.py build_ext --inplace
```

Produces `autocall_pricer_lv.<abi>.pyd`. Requires AVX2+FMA; AVX-512F+DQ is
autodetected at runtime.

## Usage

```python
import autocall_pricer_lv as ap

spec = ap.make_single_spec(S0, notional, strike, ki_barrier, True,
    obs_times, ac_barriers, coupons, fwd_vols, fwd_rates, fwd_divs)

# Baseline (K=1):
r1 = ap.price_single_asset(spec, 200_000, substeps_per_interval=1)

# 4 Euler sub-steps per observation:
r4 = ap.price_single_asset(spec, 200_000, substeps_per_interval=4,
                           compute_greeks=True)
print(r4.method, r4.cv_price, r4.substeps_per_interval)
```

## Caveats / scope

- **Worst-of is unchanged** — multi-asset sub-stepping requires per-asset
  Cholesky at every sub-step and is deferred.
- **Sobol dim scales as `K·N`** (or `2·K·N` with continuous-KI). At
  `K·N > 200` the module falls back to Philox automatically.
- **Price should be invariant in K** under piecewise-constant σ. Observable
  variation (~MC noise) comes from the higher-dimensional Sobol stream and
  from the finer continuous-KI bridge. A large persistent drift across K
  indicates a bug.
- **Greeks**: LR weights use the first sub-step's Z. Pathwise vega per
  bucket aggregates K sub-step contributions to the terminal log-return.

## Files

| File | Purpose |
|---|---|
| `sobol_extended.hpp`, `brownian_bridge.hpp`, `lr_greeks.hpp`, `multi_asset.hpp` | Shared primitives |
| `simd_kernel_prod.hpp` | `SubstepData` / `ObsMeta` layout |
| `simd_kernel_prod_avx2.cpp`, `simd_kernel_prod_avx512.cpp` | Sub-stepped SIMD kernels |
| `autocall_pricer_prod.cpp` | Driver: builds sub-step grid, BB, LR/PW Greeks; pybind11 module `autocall_pricer_lv` |
| `setup.py` | Builds `autocall_pricer_lv` |
| `test_prod.py` | K-sweep invariance check + Greeks validation |
| `test_prod_lv.py` | Local-vol-mode correctness tests |
