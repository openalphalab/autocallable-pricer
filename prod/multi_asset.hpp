// multi_asset.hpp
// Multi-asset (worst-of) autocallable pricer.
//
// Payoff modification from single-asset:
//   - Let worst_i = min_m (S_m(t_i) / S0_m)  (the worst performer's return)
//   - Autocall trigger: worst_i >= ac_barriers[i]
//   - KI trigger: worst_{at any time} <= ki_barrier
//   - Terminal put: if KI and worst_N < strike, payoff scales as worst_N / strike
//
// Correlation: upper-triangular Cholesky L of the correlation matrix, applied
// to i.i.d. normals Z[m] to produce correlated normals per asset.

#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace multi_asset {

// Compute Cholesky L such that L * L^T = rho.  rho is row-major, size M*M.
// Returns row-major L (lower triangular).
static inline std::vector<double> cholesky(const std::vector<double>& rho, int M) {
    std::vector<double> L(M * M, 0.0);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = rho[i * M + j];
            for (int k = 0; k < j; ++k) s -= L[i * M + k] * L[j * M + k];
            if (i == j) {
                if (s <= 0.0) throw std::runtime_error("correlation matrix not positive definite");
                L[i * M + i] = std::sqrt(s);
            } else {
                L[i * M + j] = s / L[j * M + j];
            }
        }
    }
    return L;
}

// Multi-asset spec: all per-asset arrays are size M, time-dependent arrays are size N.
// For simplicity we use the SAME obs schedule / ac_barriers / ki_barrier for all assets
// (standard worst-of convention); only the vol/div/spot differ per asset.
struct WorstOfSpec {
    int M;                         // number of underlyings
    int N;                         // number of observations
    std::vector<double> S0;        // [M] initial spots
    double notional;
    double strike;                 // fraction of worst-of initial (applied to worst_N)
    double ki_barrier;             // fraction (worst-of)
    bool   continuous_ki;
    std::vector<double> obs_times; // [N]
    std::vector<double> ac_barriers; // [N]
    std::vector<double> coupons;     // [N]
    std::vector<double> fwd_vols;    // [N*M], row-major: interval i asset m -> fwd_vols[i*M+m]
    std::vector<double> fwd_rates;   // [N] (same rate for all assets)
    std::vector<double> fwd_divs;    // [N*M] row-major
    std::vector<double> correlation;// [M*M] correlation matrix
};

// Scalar worst-of path kernel. Returns PV and CV payoff (put on worst_T).
// Z: [N * M] normals, row-major: N_idx * M + m
// BB_u: [N] uniforms for BB KI (shared across assets for simplicity; hits if any asset crosses)
// For continuous KI, we use the Brownian bridge exit probability on the WORST path at each step.
// If any asset crosses (or the worst bridge probabilty exceeds the uniform), we mark ki.
struct PathResult {
    double pv;
    double cv_payoff;
    int    call_idx;   // -1 if never called
    bool   ki_hit;
    double worst_T;    // worst performer ratio at T, for CV
};

static inline PathResult run_worstof_path(
    const WorstOfSpec& s,
    const std::vector<double>& L,       // Cholesky (M*M)
    const double* Z,                    // [N*M] normals (correlated NOT required; we apply L)
    const double* BB_u,                 // [N] uniforms for BB KI
    const std::vector<double>& dts,
    const std::vector<double>& cum_rT)  // [N] cumulative rT (since rate shared)
{
    int M = s.M, N = s.N;
    std::vector<double> log_s(M, 0.0), prev_log_s(M, 0.0);
    std::vector<double> correlated_Z(M);
    bool ki_hit = false;
    int call_idx = -1;
    double pv = 0.0;

    double log_ki = std::log(s.ki_barrier);
    double log_ac_at = 0.0;

    for (int i = 0; i < N; ++i) {
        // Apply Cholesky: Z_correlated[m] = sum_{k<=m} L[m*M+k] * Z[i*M + k]
        for (int m = 0; m < M; ++m) {
            double acc = 0.0;
            for (int k = 0; k <= m; ++k) acc += L[m * M + k] * Z[i * M + k];
            correlated_Z[m] = acc;
        }

        double dt = dts[i];
        double r = s.fwd_rates[i];

        // Update each asset's log-return
        double worst = std::numeric_limits<double>::infinity();
        double prev_worst = std::numeric_limits<double>::infinity();
        for (int m = 0; m < M; ++m) {
            prev_log_s[m] = log_s[m];
            double sig = s.fwd_vols[i * M + m];
            double q   = s.fwd_divs[i * M + m];
            double mu_m = (r - q - 0.5 * sig * sig) * dt;
            log_s[m] += mu_m + sig * std::sqrt(dt) * correlated_Z[m];
            if (log_s[m] < worst) worst = log_s[m];
            if (prev_log_s[m] < prev_worst) prev_worst = prev_log_s[m];
        }

        // KI check on worst performer
        if (!ki_hit) {
            if (s.continuous_ki) {
                if (worst <= log_ki) ki_hit = true;
                else if (prev_worst > log_ki) {
                    // BB probability for worst-of is hard to compute exactly; we approximate
                    // using the tightest single-asset bridge probability (conservative).
                    double max_prob = 0.0;
                    for (int m = 0; m < M; ++m) {
                        double a = prev_log_s[m] - log_ki;
                        double b = log_s[m] - log_ki;
                        if (a > 0 && b > 0) {
                            double sig = s.fwd_vols[i * M + m];
                            double nu2 = sig * sig * dt;
                            double prob = std::exp(-2.0 * a * b / nu2);
                            if (prob > max_prob) max_prob = prob;
                        }
                    }
                    if (BB_u && BB_u[i] < max_prob) ki_hit = true;
                }
            } else {
                if (worst <= log_ki) ki_hit = true;
            }
        }

        // Autocall check on worst performer
        if (call_idx < 0 && worst >= std::log(s.ac_barriers[i])) {
            call_idx = i;
            double df_i = std::exp(-cum_rT[i]);
            pv = s.notional * (1.0 + s.coupons[i]) * df_i;
        }
    }

    // Final worst_T ratio (using all paths' exp to get S_m/S0_m, take min)
    double worst_T = std::exp(*std::min_element(log_s.begin(), log_s.end()));
    double df_T = std::exp(-cum_rT.back());

    PathResult out;
    out.ki_hit = ki_hit;
    out.call_idx = call_idx;
    out.worst_T = worst_T;

    if (call_idx >= 0) {
        out.pv = pv;
    } else {
        if (ki_hit && worst_T < s.strike) {
            out.pv = s.notional * (worst_T / s.strike) * df_T;
        } else {
            out.pv = s.notional * df_T;
        }
    }
    // CV: put on worst_T struck at strike
    out.cv_payoff = std::max(s.strike - worst_T, 0.0) * s.notional * df_T;
    return out;
}

} // namespace multi_asset
