// autocall_pricer_prod.cpp  (prod/ edition)
// Production autocallable pricer with local-vol sub-stepping.
//
// Adds a `substeps_per_interval` (K) parameter on top of the `final/` build.
// Each observation interval dt_i is split into K equal Euler sub-steps of
// dt_sub = dt_i / K.  Barrier/autocall checks at observation endpoints are
// unchanged; continuous-KI monitoring now fires every sub-step.
//
//   - Drift and diffusion are built on the T = N*K sub-step grid.
//   - σ remains piecewise-constant over the observation interval containing
//     the sub-step, which matches the forward-vol term-structure pipeline in
//     util/autocall_prep.py.  This is the scaffolding for a future local-vol
//     surface σ(S, t): plug the sub-step σ lookup into `fwd_vol_sub` below.
//   - Brownian bridge, when enabled, is constructed over the full sub-step
//     grid (dim = T, or 2T with continuous-KI BB uniforms).
//   - LR Delta/Gamma weights use the first sub-step (ν_1 = σ_1·√(dt_1/K)).
//   - Pathwise Vega per observation bucket aggregates all K sub-step
//     sensitivities inside the bucket.
//
// K = 1 reproduces the prior price exactly (identical sub-step/obs grid).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <chrono>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sobol_extended.hpp"
#include "brownian_bridge.hpp"
#include "simd_kernel_prod.hpp"
#include "lr_greeks.hpp"
#include "multi_asset.hpp"

namespace py = pybind11;

// ============================================================
// Runtime SIMD detection
// ============================================================
static bool has_avx512_runtime() {
#if defined(__GNUC__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq");
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int cpu_info[4] = {};
    __cpuid(cpu_info, 1);
    const bool osxsave = (cpu_info[2] & (1 << 27)) != 0;
    const bool avx = (cpu_info[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) return false;

    const unsigned long long xcr0 = _xgetbv(0);
    const unsigned long long avx512_state_mask = 0xE6;
    if ((xcr0 & avx512_state_mask) != avx512_state_mask) return false;

    __cpuidex(cpu_info, 7, 0);
    return (cpu_info[1] & (1 << 16)) != 0 && (cpu_info[1] & (1 << 17)) != 0;
#else
    return false;
#endif
}

static bool has_avx2_runtime() {
#if defined(__GNUC__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int cpu_info[4] = {};
    __cpuid(cpu_info, 1);
    const bool osxsave = (cpu_info[2] & (1 << 27)) != 0;
    const bool avx = (cpu_info[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) return false;

    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6) != 0x6) return false;

    __cpuidex(cpu_info, 7, 0);
    return (cpu_info[1] & (1 << 5)) != 0;
#else
    return false;
#endif
}

// ============================================================
// Philox + Acklam (copied verbatim for self-contained module)
// ============================================================
struct Philox4x32 {
    static constexpr uint32_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
    static constexpr uint32_t W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
    static inline std::array<uint32_t,4> gen(uint64_t a, uint64_t b, uint32_t sh, uint32_t sl) {
        std::array<uint32_t,4> c = {
            (uint32_t)a, (uint32_t)(a>>32),
            (uint32_t)b, (uint32_t)(b>>32)};
        std::array<uint32_t,2> k = { sl, sh };
        for (int i = 0; i < 10; ++i) {
            uint64_t p0 = (uint64_t)M0 * c[0];
            uint64_t p1 = (uint64_t)M1 * c[2];
            std::array<uint32_t,4> o = {
                (uint32_t)(p1>>32) ^ c[1] ^ k[0], (uint32_t)p1,
                (uint32_t)(p0>>32) ^ c[3] ^ k[1], (uint32_t)p0};
            c = o;
            k[0] += W0; k[1] += W1;
        }
        return c;
    }
};
static inline double u32_to_unit(uint32_t x) {
    return (static_cast<double>(x) + 0.5) * (1.0 / 4294967296.0);
}
static inline double inv_normal_cdf(double p) {
    static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                                -2.759285104469687e+02, 1.383577518672690e+02,
                                -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                                -1.556989798598866e+02, 6.680131188771972e+01,
                                -1.328068155288572e+01};
    static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                -2.400758277161838e+00, -2.549732539343734e+00,
                                 4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[4] = { 7.784695709041462e-03, 3.224671290700398e-01,
                                 2.445134137142996e+00, 3.754408661907416e+00};
    const double pl = 0.02425, ph = 1.0 - pl;
    double q, r;
    if (p < pl) {
        q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= ph) {
        q = p - 0.5; r = q*q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
}
static inline double norm_cdf(double x) { return 0.5 * std::erfc(-x * 0.7071067811865475); }

// ============================================================
// Local-volatility grid (optional)
// ============================================================
//
// Stores sigma_loc on a 2-D grid:
//     log_m_grid[n_m]   — log(K/S0) axis, uniformly spaced (we enforce this
//                          so lookup is O(1) in the spatial dimension).
//     t_grid[n_t]        — time-in-years axis. t_grid[0] must be 0.
//                          May be non-uniform (log-spaced is common) — we
//                          bilinearly interpolate using precomputed per-
//                          sub-step indices, so runtime cost is constant.
//     sigma_loc[n_m*n_t] — row-major: sigma_loc[i*n_t + j] = σ(log_m_i, t_j).
//
// Flat extrapolation outside the grid.
struct LocalVolGrid {
    std::vector<double> log_m_grid;
    std::vector<double> t_grid;
    std::vector<double> sigma_loc;
    int n_m = 0;
    int n_t = 0;
    double log_m_lo = 0.0;
    double log_m_hi = 0.0;
    double dlogm = 0.0;            // log_m_grid[1] - log_m_grid[0] (uniform)

    bool empty() const { return n_m == 0 || n_t == 0; }

    // Locate the t-grid cell [j, j+1] that contains t; also return beta.
    // Linear scan could be O(n_t) per sub-step; we precompute per-sub-step
    // indices elsewhere, so this is only used for ad-hoc lookup.
    inline int t_cell(double t, double& beta) const {
        if (t <= t_grid.front()) { beta = 0.0; return 0; }
        if (t >= t_grid.back())  { beta = 1.0; return n_t - 2; }
        int lo = 0, hi = n_t - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) >> 1;
            if (t_grid[mid] <= t) lo = mid; else hi = mid;
        }
        double dt = t_grid[hi] - t_grid[lo];
        beta = dt > 0.0 ? (t - t_grid[lo]) / dt : 0.0;
        return lo;
    }

    // Bilinear lookup with flat extrapolation on the spatial axis.
    // j and beta for the t axis are assumed precomputed (passed in).
    inline double sigma_at(double log_m, int j, double beta) const {
        double xi = (log_m - log_m_lo) / dlogm;
        int i;
        double alpha;
        if (xi <= 0.0) { i = 0; alpha = 0.0; }
        else if (xi >= (double)(n_m - 1)) { i = n_m - 2; alpha = 1.0; }
        else {
            i = (int)xi;
            alpha = xi - (double)i;
        }
        const double* row0 = &sigma_loc[(size_t)i * n_t + j];
        const double* row1 = &sigma_loc[(size_t)(i + 1) * n_t + j];
        double a = row0[0], b = row1[0];
        double c = row0[1], d = row1[1];
        double top = (1.0 - alpha) * a + alpha * b;
        double bot = (1.0 - alpha) * c + alpha * d;
        return (1.0 - beta) * top + beta * bot;
    }
};

// ============================================================
// Single-asset spec (identical to final; only the pricer signature adds K)
// ============================================================
struct SingleAssetSpec {
    double S0, notional, strike, ki_barrier;
    bool continuous_ki;
    std::vector<double> obs_times, ac_barriers, coupons;
    std::vector<double> fwd_vols, fwd_rates, fwd_divs;
    // Optional Dupire local-vol surface. When `use_local_vol` is true, the
    // pricer ignores spec.fwd_vols in the state-dependent σ lookup and uses
    // `lv` per sub-step. fwd_vols is still used for the analytic CV baseline
    // (biased but correlated under LV) and as fallback.
    bool use_local_vol = false;
    LocalVolGrid lv;
};

struct ProdResult {
    double price;
    double stderr_;
    double cv_price;
    double cv_stderr;
    double autocall_prob;
    double ki_prob;
    std::vector<double> ac_probs_by_date;
    int64_t n_paths;
    double elapsed_ms;
    std::string method;
    int substeps_per_interval;
    // Greeks (optional)
    bool has_greeks;
    double delta;        // LR (first sub-step)
    double gamma;        // LR (first sub-step)
    double vega_total;   // pathwise sum
    double rho_total;    // pathwise sum
    std::vector<double> vega_buckets;  // per-observation pathwise vega
    std::vector<double> rho_buckets;   // per-observation pathwise rho
};

// ============================================================
// Single-asset production pricer  (sub-stepped)
// ============================================================
ProdResult price_single_asset(
    const SingleAssetSpec& spec,
    int64_t n_paths,
    uint64_t seed,
    bool use_sobol,
    bool use_brownian_bridge,
    int  simd_width,              // 0 = scalar, 4 = AVX2, 8 = AVX-512, -1 = auto
    bool use_control_variate,
    bool compute_greeks,
    int  n_threads,
    int  substeps_per_interval)   // K >= 1
{
    int N = (int)spec.obs_times.size();
    if (N == 0) throw std::runtime_error("need >=1 observation");
    int K = std::max(1, substeps_per_interval);
    int T = N * K;                        // total sub-steps per path
    const bool LV = spec.use_local_vol && !spec.lv.empty();

    // ------------------------------------------------------------------
    // Build per-observation (dts, discounts, barriers, coupons) and per
    // sub-step (mu, nu2) arrays. Sub-step grid is uniform *within* each
    // obs interval: dt_sub = dt_i / K.
    //
    // Under local vol (LV == true), `ss[k].mu` / `ss[k].nu2` are unused
    // (σ depends on state), but we still populate them with the obs-interval
    // σ value so SIMD fallback would behave sanely.
    // ------------------------------------------------------------------
    std::vector<double> dts(N), dts_sub(N);
    std::vector<double> sub_times(T);
    std::vector<double> sub_times_start(T);    // t at start of each sub-step (for LV σ lookup)
    std::vector<simd_prod::SubstepData> ss(T);
    std::vector<simd_prod::ObsMeta>     obs(N);
    double t_prev = 0.0, cum_rate_time = 0.0;
    int sub_idx = 0;
    for (int i = 0; i < N; ++i) {
        double dt = spec.obs_times[i] - t_prev;
        if (dt <= 0.0) throw std::runtime_error("obs_times must be strictly increasing");
        dts[i]     = dt;
        dts_sub[i] = dt / K;
        double sig = spec.fwd_vols[i];
        double mu_sub  = (spec.fwd_rates[i] - spec.fwd_divs[i] - 0.5 * sig * sig) * dts_sub[i];
        double nu2_sub = sig * sig * dts_sub[i];
        for (int j = 0; j < K; ++j) {
            ss[sub_idx].mu  = mu_sub;
            ss[sub_idx].nu2 = nu2_sub;
            sub_times_start[sub_idx] = t_prev + j       * dts_sub[i];
            sub_times[sub_idx]       = t_prev + (j + 1) * dts_sub[i];
            ++sub_idx;
        }
        cum_rate_time += spec.fwd_rates[i] * dt;
        obs[i].substep_end_idx = i * K + (K - 1);
        obs[i].log_ac          = std::log(spec.ac_barriers[i]);
        obs[i].df              = std::exp(-cum_rate_time);
        obs[i].coupon_payoff   = spec.notional * (1.0 + spec.coupons[i]) * obs[i].df;
        t_prev = spec.obs_times[i];
    }
    double log_ki     = std::log(spec.ki_barrier);
    double log_strike = std::log(spec.strike);
    double df_T       = obs[N-1].df;

    // Precompute LV t-grid cell (j, beta) for each sub-step start-time.
    // Under LV we look up σ at the STATE of the path at the beginning of the
    // sub-step (Euler-Maruyama convention: σ frozen over [t_k, t_{k+1}]).
    // The spatial axis still needs the realized log_s, so lookup happens
    // inside the walker, but we can precompute the t-axis part here.
    std::vector<int>    lv_t_idx(T, 0);
    std::vector<double> lv_t_beta(T, 0.0);
    std::vector<double> sub_nu(T, 0.0);        // √(dt_sub) per sub-step (LV: pre-σ factor)
    std::vector<double> sub_dt(T, 0.0);
    std::vector<int>    sub_ii(T, 0);          // obs-interval index (i = k/K when uniform K)
    for (int k = 0; k < T; ++k) {
        int ii = k / K;
        sub_ii[k] = ii;
        sub_dt[k] = dts_sub[ii];
        sub_nu[k] = std::sqrt(dts_sub[ii]);
        if (LV) {
            double beta = 0.0;
            int j = spec.lv.t_cell(sub_times_start[k], beta);
            lv_t_idx[k]  = j;
            lv_t_beta[k] = beta;
        }
    }
    // σ at (log_m = 0, t = 0) — used for LR Greek weights in LV mode.
    double sigma0_at_S0 = 0.0;
    if (LV) {
        double beta0 = 0.0;
        int j0 = spec.lv.t_cell(0.0, beta0);
        sigma0_at_S0 = spec.lv.sigma_at(0.0, j0, beta0);
    }

    // BB over the full sub-step grid
    BrownianBridge bb(sub_times);

    // ------------------------------------------------------------------
    // Analytic CV (vanilla put at T), unchanged — depends only on total
    // variance / rate / div integrated to maturity, not on sub-stepping.
    // ------------------------------------------------------------------
    double cv_analytic = 0.0;
    {
        double Tmat = spec.obs_times[N-1];
        double total_var = 0, total_rT = 0, total_qT = 0;
        for (int i = 0; i < N; ++i) {
            total_var += spec.fwd_vols[i] * spec.fwd_vols[i] * dts[i];
            total_rT  += spec.fwd_rates[i] * dts[i];
            total_qT  += spec.fwd_divs[i] * dts[i];
        }
        double r_avg = total_rT / Tmat, q_avg = total_qT / Tmat;
        double sig_eff = std::sqrt(total_var / Tmat);
        double K_abs = spec.strike * spec.S0;
        double F = spec.S0 * std::exp((r_avg - q_avg) * Tmat);
        double d1 = (std::log(F/K_abs) + 0.5 * sig_eff*sig_eff * Tmat) / (sig_eff * std::sqrt(Tmat));
        double d2 = d1 - sig_eff * std::sqrt(Tmat);
        cv_analytic = std::exp(-total_rT) * (K_abs * norm_cdf(-d2) - F * norm_cdf(-d1));
    }

    // Select SIMD width.
    // Local-vol mode currently runs scalar: σ depends on the realized state
    // at each sub-step, so mu/nu2 cannot be precomputed into the SIMD path
    // kernel. (AVX gathers on sigma_loc are feasible but deferred.)
    if (simd_width == -1) {
        if (has_avx512_runtime()) simd_width = 8;
        else if (has_avx2_runtime()) simd_width = 4;
        else simd_width = 1;
    }
    if (simd_width == 8 && !has_avx512_runtime()) {
        simd_width = has_avx2_runtime() ? 4 : 1;
    } else if (simd_width == 4 && !has_avx2_runtime()) {
        simd_width = 1;
    }
    if (LV) simd_width = 1;

#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
    int Tthr = omp_get_max_threads();
#else
    int Tthr = 1;
#endif

    // Sobol dim: T normals + (continuous_ki ? T uniforms : 0)
    int sobol_dim = spec.continuous_ki ? 2 * T : T;
    if (sobol_dim > sobol::MAX_SOBOL_DIM) use_sobol = false;

    // Per-thread accumulators
    std::vector<double> S_pv(Tthr,0), S_pv2(Tthr,0), S_cv(Tthr,0), S_cv2(Tthr,0), S_pvcv(Tthr,0);
    std::vector<double> S_ac(Tthr,0), S_ki(Tthr,0);
    std::vector<std::vector<int64_t>> S_ac_date(Tthr, std::vector<int64_t>(N, 0));
    std::vector<double> S_delta(Tthr, 0.0), S_gamma(Tthr, 0.0);
    std::vector<std::vector<double>> S_vega(Tthr, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> S_rho(Tthr, std::vector<double>(N, 0.0));

    uint32_t seed_lo = (uint32_t)(seed & 0xFFFFFFFFu);
    uint32_t seed_hi = (uint32_t)(seed >> 32);

    int64_t paths_per_thread = (n_paths + Tthr - 1) / Tthr;
    auto t_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        int64_t p_start = (int64_t)tid * paths_per_thread;
        int64_t p_end   = std::min(p_start + paths_per_thread, n_paths);

        // Only build the Sobol generator when it will actually be used; the
        // constructor throws for dim > MAX_SOBOL_DIM, so we must guard.
        int sobol_dim_safe = use_sobol ? sobol_dim : 1;
        sobol::ScrambledSobol sobol_gen(sobol_dim_safe,
            seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(tid + 1)));

        std::vector<double> u_dim(sobol_dim);
        std::vector<double> Z(T), BB_u(T), BM(T);
        // SIMD buffers sized for max width
        alignas(64) std::vector<double> W_buf(T * 8, 0.0);
        alignas(64) std::vector<double> U_buf(T * 8, 0.0);
        // Per-lane Z cache: full sub-step path for Greeks
        std::vector<double> Z_path_batch(T * 8, 0.0);

        double& acc_pv    = S_pv[tid];
        double& acc_pv2   = S_pv2[tid];
        double& acc_cv    = S_cv[tid];
        double& acc_cv2   = S_cv2[tid];
        double& acc_pvcv  = S_pvcv[tid];
        double& acc_ac    = S_ac[tid];
        double& acc_ki    = S_ki[tid];
        auto&   acc_ad    = S_ac_date[tid];
        double& acc_delta = S_delta[tid];
        double& acc_gamma = S_gamma[tid];
        auto&   acc_vega  = S_vega[tid];
        auto&   acc_rho   = S_rho[tid];

        auto make_one_path = [&](int64_t pid, int lane_offset, int batch_width) {
            // 1. Draw primitive variates.
            if (use_sobol) {
                sobol_gen.next(u_dim.data());
                for (int k = 0; k < T; ++k) Z[k] = inv_normal_cdf(u_dim[k]);
                if (spec.continuous_ki) for (int k = 0; k < T; ++k) BB_u[k] = u_dim[T + k];
            } else {
                int blks = (sobol_dim + 3) / 4, idx = 0;
                for (int b = 0; b < blks; ++b) {
                    auto r4 = Philox4x32::gen((uint64_t)pid, (uint64_t)b, seed_hi, seed_lo);
                    for (int k = 0; k < 4 && idx < sobol_dim; ++k, ++idx) {
                        if (idx < T) Z[idx] = inv_normal_cdf(u32_to_unit(r4[k]));
                        else BB_u[idx - T] = u32_to_unit(r4[k]);
                    }
                }
            }

            // 2. Build Brownian motion on sub-step grid.
            if (use_brownian_bridge && bb.enabled()) {
                bb.build_path(Z.data(), BM.data());
                // Sequential-equivalent Z for Greeks (use sub-step dt).
                double prev = 0.0;
                for (int k = 0; k < T; ++k) {
                    // Find which observation interval this sub-step lives in
                    // (integer div by K since K uniform here).
                    int ii = k / K;
                    double z_seq = (BM[k] - prev) / std::sqrt(dts_sub[ii]);
                    Z[k] = z_seq;
                    prev = BM[k];
                }
            }

            // 3. Build σ·√dt·Z per sub-step (piecewise-const σ on obs interval).
            //    Under LV we cannot precompute σ (state-dependent), so we only
            //    stash Z / BB_u here and let the walker compute σ·√dt·Z_k
            //    in-line using sigma_loc(log_s, t).
            for (int k = 0; k < T; ++k) {
                int ii = k / K;
                if (!LV) {
                    double sig = spec.fwd_vols[ii];
                    W_buf[k * batch_width + lane_offset] =
                        sig * std::sqrt(dts_sub[ii]) * Z[k];
                }
                if (spec.continuous_ki) U_buf[k * batch_width + lane_offset] = BB_u[k];
                Z_path_batch[k * batch_width + lane_offset] = Z[k];
            }
        };

        // ------------------------------------------------------------
        // Scalar sub-step walker -- used in scalar mode and for Greeks
        // re-walk after SIMD.  Returns (pv, ki_hit, ci, log_s_final).
        // ------------------------------------------------------------
        struct ScalarOut {
            double pv;
            double log_s;
            bool   ki;
            int    ci;       // first obs index that autocalled, or -1
        };
        auto walk_scalar = [&](int lane_offset, int batch_width) -> ScalarOut {
            ScalarOut r{0.0, 0.0, false, -1};
            double log_s = 0.0, prev = 0.0;
            int obs_cursor = 0;
            for (int k = 0; k < T; ++k) {
                prev = log_s;
                // Euler-Maruyama step.
                //   Non-LV: mu and W = σ·√dt·Z were precomputed.
                //   LV    : σ is σ_loc(log_s_prev, t_start_of_step); mu and
                //           nu = σ·√dt are derived per-step.
                double nu2_step;
                if (!LV) {
                    log_s += ss[k].mu + W_buf[k * batch_width + lane_offset];
                    nu2_step = ss[k].nu2;
                } else {
                    double sig = spec.lv.sigma_at(prev, lv_t_idx[k], lv_t_beta[k]);
                    int ii = sub_ii[k];
                    double dt = sub_dt[k];
                    double mu = (spec.fwd_rates[ii] - spec.fwd_divs[ii]
                                 - 0.5 * sig * sig) * dt;
                    double Z_k = Z_path_batch[k * batch_width + lane_offset];
                    log_s += mu + sig * sub_nu[k] * Z_k;
                    nu2_step = sig * sig * dt;
                }

                if (spec.continuous_ki && !r.ki) {
                    if (log_s <= log_ki) r.ki = true;
                    else if (prev > log_ki) {
                        double a = prev - log_ki, b = log_s - log_ki;
                        double prob = std::exp(-2.0 * a * b / nu2_step);
                        if (U_buf[k * batch_width + lane_offset] < prob) r.ki = true;
                    }
                }

                if (obs_cursor < N && k == obs[obs_cursor].substep_end_idx) {
                    if (!spec.continuous_ki && !r.ki && log_s <= log_ki) r.ki = true;
                    if (r.ci < 0 && log_s >= obs[obs_cursor].log_ac) {
                        r.ci = obs_cursor;
                        r.pv = obs[obs_cursor].coupon_payoff;
                    }
                    ++obs_cursor;
                    if (r.ci >= 0) {
                        // run remaining sub-steps only if we still need log_s for CV;
                        // we do, because the analytic CV uses S_T.
                    }
                }
            }
            r.log_s = log_s;
            if (r.ci < 0) {
                if (r.ki && log_s < log_strike) {
                    r.pv = spec.notional * std::exp(log_s) / spec.strike * df_T;
                } else {
                    r.pv = spec.notional * df_T;
                }
            }
            return r;
        };

        auto accumulate_greeks_lane = [&](int lane_offset, int batch_width,
                                          const ScalarOut& r) {
            // LR Delta/Gamma via the FIRST sub-step's standardized normal.
            // Under LV, ν₁ uses σ_loc(S₀, 0) rather than fwd_vols[0].
            double Z1   = Z_path_batch[0 * batch_width + lane_offset];
            double sig0 = LV ? sigma0_at_S0 : spec.fwd_vols[0];
            double nu1  = sig0 * std::sqrt(dts_sub[0]);
            auto w = lr_greeks::compute_lr_weights(Z1, spec.S0, nu1);
            acc_delta += r.pv * w.delta;
            acc_gamma += r.pv * w.gamma;

            // Pathwise Vega/Rho per observation bucket.
            // d(log S_T)/d(σ_j) = Σ_{sub in bucket j} (-σ_j · dt_sub + √dt_sub · Z_sub)
            //                   = -σ_j·dt_j + √dt_sub · Σ_{sub in j} Z_sub.
            // d(log S_T)/d(r_j) = dt_j  (and d(df_T)/d(r_j) = -dt_j · df_T).
            // For each terminal leg:
            //   - autocalled at obs i:  pv = N(1+c_i) df_i, vol-independent.
            //                           rho[j] = -dt_j · pv for j <= i.
            //   - terminal par:         vol-independent,
            //                           rho[j] = -dt_j · pv for j < N.
            //   - terminal loss:        pv = N·exp(log_s)/strike · df_T,
            //                           vega[j] = pv · dlogs_dsig[j],
            //                           rho[j]  = 0 (drift +dt cancels df -dt).
            if (r.ci >= 0) {
                for (int j = 0; j <= r.ci; ++j) acc_rho[j] += -dts[j] * r.pv;
                return;
            }
            if (r.ki && r.log_s < log_strike) {
                // Bucket vega is ill-defined under a full local-vol surface
                // (σ is a function of state, not a per-bucket scalar). Skip
                // accumulation; users should compute surface vega by bumping
                // the LV grid externally and re-running.
                if (!LV) {
                    for (int j = 0; j < N; ++j) {
                        double sig_j = spec.fwd_vols[j];
                        double sum_Z = 0.0;
                        for (int jj = 0; jj < K; ++jj) {
                            sum_Z += Z_path_batch[(j * K + jj) * batch_width + lane_offset];
                        }
                        double dls_dsig_j = -sig_j * dts[j] + std::sqrt(dts_sub[j]) * sum_Z;
                        acc_vega[j] += r.pv * dls_dsig_j;
                    }
                }
                // rho contributions are 0 under this derivation; see note above.
            } else {
                for (int j = 0; j < N; ++j) acc_rho[j] += -dts[j] * r.pv;
            }
        };

        int BW = simd_width;
        if (BW < 1) BW = 1;

        int64_t p = p_start;

        // Scalar-width path
        if (BW == 1) {
            while (p < p_end) {
                make_one_path(p, 0, 1);
                ScalarOut r = walk_scalar(0, 1);
                if (r.ci >= 0) { acc_ac += 1.0; acc_ad[r.ci] += 1; }
                else if (r.ki) acc_ki += 1.0;
                acc_pv  += r.pv;
                acc_pv2 += r.pv * r.pv;
                double sT = spec.S0 * std::exp(r.log_s);
                double put_pay = std::max(spec.strike * spec.S0 - sT, 0.0) * df_T;
                acc_cv    += put_pay;
                acc_cv2   += put_pay * put_pay;
                acc_pvcv  += r.pv * put_pay;
                if (compute_greeks) accumulate_greeks_lane(0, 1, r);
                ++p;
            }
        }

        while (p + BW <= p_end && BW > 1) {
            for (int lane = 0; lane < BW; ++lane) make_one_path(p + lane, lane, BW);
            if (BW == 4) {
                simd_prod::BatchResult br = simd_prod::process_batch_avx2(
                    T, N, ss.data(), obs.data(), W_buf.data(),
                    spec.continuous_ki ? U_buf.data() : nullptr,
                    spec.S0, spec.notional, log_ki, log_strike, df_T,
                    spec.strike, spec.continuous_ki, acc_ad.data());
                acc_pv    += br.sum_pv;
                acc_pv2   += br.sum_pv2;
                acc_cv    += br.sum_cv;
                acc_cv2   += br.sum_cv2;
                acc_pvcv  += br.sum_pvcv;
                acc_ac    += br.sum_ac_count;
                acc_ki    += br.sum_ki_count;
            } else if (BW == 8) {
                simd_prod::BatchResult br = simd_prod::process_batch_avx512(
                    T, N, ss.data(), obs.data(), W_buf.data(),
                    spec.continuous_ki ? U_buf.data() : nullptr,
                    spec.S0, spec.notional, log_ki, log_strike, df_T,
                    spec.strike, spec.continuous_ki, acc_ad.data());
                acc_pv    += br.sum_pv;
                acc_pv2   += br.sum_pv2;
                acc_cv    += br.sum_cv;
                acc_cv2   += br.sum_cv2;
                acc_pvcv  += br.sum_pvcv;
                acc_ac    += br.sum_ac_count;
                acc_ki    += br.sum_ki_count;
            }

            if (compute_greeks) {
                for (int lane = 0; lane < BW; ++lane) {
                    ScalarOut r = walk_scalar(lane, BW);
                    accumulate_greeks_lane(lane, BW, r);
                }
            }
            p += BW;
        }

        // Scalar tail
        while (p < p_end) {
            make_one_path(p, 0, 1);
            ScalarOut r = walk_scalar(0, 1);
            if (r.ci >= 0) { acc_ac += 1.0; acc_ad[r.ci] += 1; }
            else if (r.ki) acc_ki += 1.0;
            acc_pv  += r.pv;
            acc_pv2 += r.pv * r.pv;
            double sT = spec.S0 * std::exp(r.log_s);
            double put_pay = std::max(spec.strike * spec.S0 - sT, 0.0) * df_T;
            acc_cv    += put_pay;
            acc_cv2   += put_pay * put_pay;
            acc_pvcv  += r.pv * put_pay;
            if (compute_greeks) accumulate_greeks_lane(0, 1, r);
            ++p;
        }
    }

    // Reduce
    double sum_pv = 0, sum_pv2 = 0, sum_cv = 0, sum_cv2 = 0, sum_pvcv = 0, sum_ac = 0, sum_ki = 0;
    std::vector<int64_t> ad(N, 0);
    double sum_delta = 0, sum_gamma = 0;
    std::vector<double> sum_vega(N, 0), sum_rho(N, 0);
    for (int t = 0; t < Tthr; ++t) {
        sum_pv  += S_pv[t];   sum_pv2 += S_pv2[t];
        sum_cv  += S_cv[t];   sum_cv2 += S_cv2[t];
        sum_pvcv += S_pvcv[t];
        sum_ac  += S_ac[t];   sum_ki  += S_ki[t];
        for (int i = 0; i < N; ++i) ad[i] += S_ac_date[t][i];
        sum_delta += S_delta[t]; sum_gamma += S_gamma[t];
        for (int i = 0; i < N; ++i) { sum_vega[i] += S_vega[t][i]; sum_rho[i] += S_rho[t][i]; }
    }
    double Nf = (double)n_paths;
    double mean_pv = sum_pv / Nf;
    double var_pv  = std::max(0.0, sum_pv2/Nf - mean_pv*mean_pv);
    double se_pv   = std::sqrt(var_pv / Nf);
    double mean_cv = sum_cv / Nf;
    double var_cv  = std::max(1e-30, sum_cv2/Nf - mean_cv*mean_cv);
    double cov = sum_pvcv/Nf - mean_pv*mean_cv;
    double beta = cov / var_cv;
    double cv_price = mean_pv - beta * (mean_cv - cv_analytic);
    double var_adj  = std::max(0.0, var_pv - beta*beta*var_cv);
    double se_cv    = std::sqrt(var_adj / Nf);

    auto t_end = std::chrono::high_resolution_clock::now();

    ProdResult res;
    res.price      = mean_pv;
    res.stderr_    = se_pv;
    res.cv_price   = use_control_variate ? cv_price : mean_pv;
    res.cv_stderr  = use_control_variate ? se_cv    : se_pv;
    res.autocall_prob = sum_ac / Nf;
    res.ki_prob       = sum_ki / Nf;
    res.ac_probs_by_date.resize(N);
    for (int i = 0; i < N; ++i) res.ac_probs_by_date[i] = (double)ad[i] / Nf;
    res.n_paths    = n_paths;
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    res.substeps_per_interval = K;
    std::string m = use_sobol ? "Sobol" : "Philox";
    if (use_brownian_bridge) m += "+BB";
    if (simd_width == 4) m += "+AVX2";
    else if (simd_width == 8) m += "+AVX512";
    if (use_control_variate) m += "+CV";
    if (compute_greeks) m += "+Greeks(LR+PW)";
    if (K > 1) m += "+Sub" + std::to_string(K);
    res.method = m;
    res.has_greeks = compute_greeks;
    if (compute_greeks) {
        res.delta = sum_delta / Nf;
        res.gamma = sum_gamma / Nf;
        res.vega_buckets.resize(N);
        res.rho_buckets.resize(N);
        double vtot = 0, rtot = 0;
        for (int i = 0; i < N; ++i) {
            res.vega_buckets[i] = sum_vega[i] / Nf;
            res.rho_buckets[i]  = sum_rho[i] / Nf;
            vtot += res.vega_buckets[i];
            rtot += res.rho_buckets[i];
        }
        res.vega_total = vtot;
        res.rho_total  = rtot;
    } else {
        res.delta = res.gamma = res.vega_total = res.rho_total = 0;
    }
    return res;
}

// ============================================================
// Multi-asset (worst-of) pricer — kept from final build; no sub-stepping
// (local-vol sub-stepping for worst-of adds further dimensions and is
// deferred). Worst-of still prices on the observation grid only.
// ============================================================
ProdResult price_worstof(
    const multi_asset::WorstOfSpec& spec,
    int64_t n_paths,
    uint64_t seed,
    bool use_sobol,
    int n_threads)
{
    int N = spec.N, M = spec.M;
    auto L = multi_asset::cholesky(spec.correlation, M);

    std::vector<double> dts(N), cum_rT(N);
    double t_prev = 0.0, cr = 0.0;
    for (int i = 0; i < N; ++i) {
        dts[i] = spec.obs_times[i] - t_prev;
        cr += spec.fwd_rates[i] * dts[i];
        cum_rT[i] = cr;
        t_prev = spec.obs_times[i];
    }

    int sobol_dim = N * M + (spec.continuous_ki ? N : 0);
    if (sobol_dim > sobol::MAX_SOBOL_DIM) use_sobol = false;

#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
    int Tthr = omp_get_max_threads();
#else
    int Tthr = 1;
#endif
    std::vector<double> S_pv(Tthr,0), S_pv2(Tthr,0), S_ac(Tthr,0), S_ki(Tthr,0);
    std::vector<std::vector<int64_t>> S_ad(Tthr, std::vector<int64_t>(N, 0));

    uint32_t sl = (uint32_t)(seed & 0xFFFFFFFFu), sh = (uint32_t)(seed >> 32);
    int64_t ppt = (n_paths + Tthr - 1) / Tthr;
    auto t_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        int64_t ps = (int64_t)tid * ppt, pe = std::min(ps + ppt, n_paths);
        sobol::ScrambledSobol sg(sobol_dim, seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(tid+1)));
        std::vector<double> u(sobol_dim), Z(N*M), BB_u(N);

        double& ap = S_pv[tid]; double& ap2 = S_pv2[tid];
        double& aa = S_ac[tid]; double& ak = S_ki[tid];
        auto& adate = S_ad[tid];

        for (int64_t p = ps; p < pe; ++p) {
            if (use_sobol) {
                sg.next(u.data());
                for (int k = 0; k < N*M; ++k) Z[k] = inv_normal_cdf(u[k]);
                if (spec.continuous_ki) for (int i = 0; i < N; ++i) BB_u[i] = u[N*M + i];
            } else {
                int blks = (sobol_dim + 3) / 4, idx = 0;
                for (int b = 0; b < blks; ++b) {
                    auto r4 = Philox4x32::gen((uint64_t)p, (uint64_t)b, sh, sl);
                    for (int k = 0; k < 4 && idx < sobol_dim; ++k, ++idx) {
                        if (idx < N*M) Z[idx] = inv_normal_cdf(u32_to_unit(r4[k]));
                        else BB_u[idx - N*M] = u32_to_unit(r4[k]);
                    }
                }
            }
            auto r = multi_asset::run_worstof_path(spec, L, Z.data(),
                spec.continuous_ki ? BB_u.data() : nullptr, dts, cum_rT);
            ap += r.pv; ap2 += r.pv * r.pv;
            if (r.call_idx >= 0) { aa += 1; adate[r.call_idx] += 1; }
            else if (r.ki_hit) ak += 1;
        }
    }

    double sum_pv = 0, sum_pv2 = 0, sum_ac = 0, sum_ki = 0;
    std::vector<int64_t> ad(N, 0);
    for (int t = 0; t < Tthr; ++t) {
        sum_pv += S_pv[t]; sum_pv2 += S_pv2[t]; sum_ac += S_ac[t]; sum_ki += S_ki[t];
        for (int i = 0; i < N; ++i) ad[i] += S_ad[t][i];
    }
    double Nf = (double)n_paths;
    double mean = sum_pv / Nf;
    double var  = std::max(0.0, sum_pv2/Nf - mean*mean);
    auto t_end = std::chrono::high_resolution_clock::now();

    ProdResult res;
    res.price      = mean;
    res.stderr_    = std::sqrt(var / Nf);
    res.cv_price   = mean;
    res.cv_stderr  = res.stderr_;
    res.autocall_prob = sum_ac / Nf;
    res.ki_prob       = sum_ki / Nf;
    res.ac_probs_by_date.resize(N);
    for (int i = 0; i < N; ++i) res.ac_probs_by_date[i] = (double)ad[i] / Nf;
    res.n_paths    = n_paths;
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    res.substeps_per_interval = 1;
    res.method     = "WorstOf-" + std::to_string(M) + "asset-" + (use_sobol ? "Sobol" : "Philox");
    res.has_greeks = false;
    res.delta = res.gamma = res.vega_total = res.rho_total = 0;
    return res;
}

// ============================================================
// pybind11 module
// ============================================================
SingleAssetSpec make_single_spec(
    double S0, double notional, double strike, double ki_barrier, bool ck,
    py::array_t<double> ot, py::array_t<double> ac, py::array_t<double> cp,
    py::array_t<double> fv, py::array_t<double> fr, py::array_t<double> fd)
{
    SingleAssetSpec s;
    s.S0=S0; s.notional=notional; s.strike=strike;
    s.ki_barrier=ki_barrier; s.continuous_ki=ck;
    auto ld = [](py::array_t<double> a, std::vector<double>& v){
        auto r = a.unchecked<1>(); v.resize(r.shape(0));
        for (py::ssize_t i = 0; i < r.shape(0); ++i) v[i] = r(i);
    };
    ld(ot, s.obs_times); ld(ac, s.ac_barriers); ld(cp, s.coupons);
    ld(fv, s.fwd_vols); ld(fr, s.fwd_rates); ld(fd, s.fwd_divs);
    return s;
}

// Attach (or clear) a local-vol grid on an existing SingleAssetSpec.
//   log_m_grid: 1-D array, size n_m (uniformly spaced log(K/S0) axis)
//   t_grid    : 1-D array, size n_t (non-decreasing; t_grid[0] must be 0)
//   sigma_loc : 2-D array, shape (n_m, n_t) — row-major σ(log_m_i, t_j)
// After this call, price_single_asset will use LV when spec.use_local_vol.
void set_local_vol(SingleAssetSpec& s,
                   py::array_t<double, py::array::c_style | py::array::forcecast> log_m_grid,
                   py::array_t<double, py::array::c_style | py::array::forcecast> t_grid,
                   py::array_t<double, py::array::c_style | py::array::forcecast> sigma_loc,
                   bool enable)
{
    auto lm = log_m_grid.unchecked<1>();
    auto tg = t_grid.unchecked<1>();
    auto sl = sigma_loc.unchecked<2>();
    int n_m = (int)lm.shape(0);
    int n_t = (int)tg.shape(0);
    if ((int)sl.shape(0) != n_m || (int)sl.shape(1) != n_t)
        throw std::runtime_error("sigma_loc shape must be (n_m, n_t)");
    if (n_m < 2 || n_t < 2)
        throw std::runtime_error("log_m_grid and t_grid must each have >=2 points");

    s.lv.log_m_grid.resize(n_m);
    for (int i = 0; i < n_m; ++i) s.lv.log_m_grid[i] = lm(i);
    s.lv.t_grid.resize(n_t);
    for (int j = 0; j < n_t; ++j) s.lv.t_grid[j] = tg(j);
    s.lv.sigma_loc.resize((size_t)n_m * n_t);
    for (int i = 0; i < n_m; ++i)
        for (int j = 0; j < n_t; ++j)
            s.lv.sigma_loc[(size_t)i * n_t + j] = sl(i, j);
    s.lv.n_m = n_m;
    s.lv.n_t = n_t;
    s.lv.log_m_lo = s.lv.log_m_grid.front();
    s.lv.log_m_hi = s.lv.log_m_grid.back();
    s.lv.dlogm    = (s.lv.log_m_hi - s.lv.log_m_lo) / (double)(n_m - 1);
    s.use_local_vol = enable;
}

multi_asset::WorstOfSpec make_worstof_spec(
    int M,
    py::array_t<double> S0, double notional, double strike, double ki_barrier, bool ck,
    py::array_t<double> ot, py::array_t<double> ac, py::array_t<double> cp,
    py::array_t<double> fv, py::array_t<double> fr, py::array_t<double> fd,
    py::array_t<double> corr)
{
    multi_asset::WorstOfSpec s;
    s.M = M;
    auto s0 = S0.unchecked<1>();
    s.S0.resize(s0.shape(0));
    for (py::ssize_t i = 0; i < s0.shape(0); ++i) s.S0[i] = s0(i);
    s.notional = notional; s.strike = strike; s.ki_barrier = ki_barrier; s.continuous_ki = ck;
    auto ld = [](py::array_t<double> a, std::vector<double>& v){
        auto r = a.unchecked<1>(); v.resize(r.shape(0));
        for (py::ssize_t i = 0; i < r.shape(0); ++i) v[i] = r(i);
    };
    ld(ot, s.obs_times);
    s.N = (int)s.obs_times.size();
    ld(ac, s.ac_barriers); ld(cp, s.coupons);
    ld(fv, s.fwd_vols); ld(fr, s.fwd_rates); ld(fd, s.fwd_divs);
    ld(corr, s.correlation);
    return s;
}

PYBIND11_MODULE(autocall_pricer_lv, m) {
    m.doc() = "Production autocallable pricer with local-vol sub-stepping (prod/)";

    py::class_<SingleAssetSpec>(m, "SingleAssetSpec", py::module_local())
        .def(py::init<>())
        .def_readwrite("S0", &SingleAssetSpec::S0)
        .def_readwrite("notional", &SingleAssetSpec::notional)
        .def_readwrite("strike", &SingleAssetSpec::strike)
        .def_readwrite("ki_barrier", &SingleAssetSpec::ki_barrier)
        .def_readwrite("continuous_ki", &SingleAssetSpec::continuous_ki)
        .def_readwrite("obs_times", &SingleAssetSpec::obs_times)
        .def_readwrite("ac_barriers", &SingleAssetSpec::ac_barriers)
        .def_readwrite("coupons", &SingleAssetSpec::coupons)
        .def_readwrite("fwd_vols", &SingleAssetSpec::fwd_vols)
        .def_readwrite("fwd_rates", &SingleAssetSpec::fwd_rates)
        .def_readwrite("fwd_divs", &SingleAssetSpec::fwd_divs)
        .def_readwrite("use_local_vol", &SingleAssetSpec::use_local_vol);

    py::class_<multi_asset::WorstOfSpec>(m, "WorstOfSpec", py::module_local())
        .def(py::init<>());

    py::class_<ProdResult>(m, "ProdResult", py::module_local())
        .def_readonly("price", &ProdResult::price)
        .def_readonly("stderr", &ProdResult::stderr_)
        .def_readonly("cv_price", &ProdResult::cv_price)
        .def_readonly("cv_stderr", &ProdResult::cv_stderr)
        .def_readonly("autocall_prob", &ProdResult::autocall_prob)
        .def_readonly("ki_prob", &ProdResult::ki_prob)
        .def_readonly("ac_probs_by_date", &ProdResult::ac_probs_by_date)
        .def_readonly("n_paths", &ProdResult::n_paths)
        .def_readonly("elapsed_ms", &ProdResult::elapsed_ms)
        .def_readonly("method", &ProdResult::method)
        .def_readonly("substeps_per_interval", &ProdResult::substeps_per_interval)
        .def_readonly("has_greeks", &ProdResult::has_greeks)
        .def_readonly("delta", &ProdResult::delta)
        .def_readonly("gamma", &ProdResult::gamma)
        .def_readonly("vega_total", &ProdResult::vega_total)
        .def_readonly("rho_total", &ProdResult::rho_total)
        .def_readonly("vega_buckets", &ProdResult::vega_buckets)
        .def_readonly("rho_buckets", &ProdResult::rho_buckets);

    m.def("make_single_spec", &make_single_spec);
    m.def("make_worstof_spec", &make_worstof_spec);
    m.def("set_local_vol", &set_local_vol,
          py::arg("spec"),
          py::arg("log_m_grid"),
          py::arg("t_grid"),
          py::arg("sigma_loc"),
          py::arg("enable") = true,
          "Attach a local-vol grid σ(log(K/S0), t) to an existing "
          "SingleAssetSpec. sigma_loc has shape (n_m, n_t).");

    m.def("price_single_asset", &price_single_asset,
          py::arg("spec"), py::arg("n_paths"), py::arg("seed") = 42ULL,
          py::arg("use_sobol") = true,
          py::arg("use_brownian_bridge") = true,
          py::arg("simd_width") = -1,
          py::arg("use_control_variate") = true,
          py::arg("compute_greeks") = false,
          py::arg("n_threads") = 0,
          py::arg("substeps_per_interval") = 1);

    m.def("price_worstof", &price_worstof,
          py::arg("spec"), py::arg("n_paths"), py::arg("seed") = 42ULL,
          py::arg("use_sobol") = true,
          py::arg("n_threads") = 0);

    m.def("has_avx512", &has_avx512_runtime);
    m.def("has_avx2",   &has_avx2_runtime);
}
