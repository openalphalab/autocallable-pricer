#pragma once

#include <cstdint>

// Sub-stepped SIMD kernel: each observation interval is split into K sub-steps
// of equal length dt_sub = dt_i / K.  Autocall/coupon checks fire only at
// sub-step indices that terminate an observation interval; continuous-KI
// bridge checks run every sub-step.
//
// Layout for K = 1 is identical to the original per-observation-step kernel,
// so a substeps_per_interval=1 call reproduces the prior price exactly.
//
// Arrays indexed by sub-step carry T = sum_i K_i entries (currently uniform K).

namespace simd_prod {

struct SubstepData {
    double mu;        // (r - q - 0.5*sig^2) * dt_sub
    double nu2;       // sig^2 * dt_sub  (BB variance for the sub-step)
};

struct ObsMeta {
    int    substep_end_idx;  // index of the sub-step closing this observation
    double log_ac;           // log autocall barrier
    double coupon_payoff;    // notional * (1 + coupon_i) * df_i
    double df;               // discount factor at this observation
};

struct BatchResult {
    double sum_pv;
    double sum_pv2;
    double sum_cv;
    double sum_cv2;
    double sum_pvcv;
    double sum_ac_count;
    double sum_ki_count;
};

BatchResult process_batch_avx2(
    int T,                         // total sub-steps
    int N,                         // number of observations
    const SubstepData* ss,         // size T
    const ObsMeta*     obs,        // size N (sorted by substep_end_idx)
    const double*      W_inc,      // size T*4   (per sub-step, per lane)
    const double*      BB_u,       // size T*4 or nullptr
    double S0,
    double notional,
    double log_ki,
    double log_strike,
    double df_T,
    double strike,
    bool   continuous_ki,
    int64_t* ac_counts_per_obs);   // size N

BatchResult process_batch_avx512(
    int T,
    int N,
    const SubstepData* ss,
    const ObsMeta*     obs,
    const double*      W_inc,      // size T*8
    const double*      BB_u,       // size T*8 or nullptr
    double S0,
    double notional,
    double log_ki,
    double log_strike,
    double df_T,
    double strike,
    bool   continuous_ki,
    int64_t* ac_counts_per_obs);

constexpr int SIMD_WIDTH_AVX2   = 4;
constexpr int SIMD_WIDTH_AVX512 = 8;

} // namespace simd_prod
