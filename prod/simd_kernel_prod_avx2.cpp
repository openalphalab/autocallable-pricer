#include "simd_kernel_prod.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace simd_prod {
namespace {

static inline int popcount32(unsigned int value) {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt(value));
#else
    return __builtin_popcount(value);
#endif
}

static inline __m256d fmadd_pd(__m256d a, __m256d b, __m256d c) {
#if defined(__FMA__)
    return _mm256_fmadd_pd(a, b, c);
#else
    return _mm256_add_pd(_mm256_mul_pd(a, b), c);
#endif
}

static inline __m256d cmp_ge(__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_GE_OQ); }
static inline __m256d cmp_lt(__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_LT_OQ); }
static inline __m256d cmp_le(__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_LE_OQ); }
static inline __m256d cmp_gt(__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_GT_OQ); }

static inline double hsum(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d s = _mm_add_pd(lo, hi);
    __m128d sh = _mm_unpackhi_pd(s, s);
    return _mm_cvtsd_f64(_mm_add_sd(s, sh));
}

static inline __m256d exp_pd(__m256d x) {
    const __m256d hi = _mm256_set1_pd(700.0);
    const __m256d lo = _mm256_set1_pd(-700.0);
    x = _mm256_min_pd(x, hi);
    x = _mm256_max_pd(x, lo);

    const __m256d LOG2E = _mm256_set1_pd(1.4426950408889634);
    const __m256d LN2 = _mm256_set1_pd(0.6931471805599453);
    __m256d fk = _mm256_round_pd(
        _mm256_mul_pd(x, LOG2E),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256d r = _mm256_sub_pd(x, _mm256_mul_pd(fk, LN2));

    const __m256d c[] = {
        _mm256_set1_pd(1.0 / 40320.0),
        _mm256_set1_pd(1.0 / 5040.0),
        _mm256_set1_pd(1.0 / 720.0),
        _mm256_set1_pd(1.0 / 120.0),
        _mm256_set1_pd(1.0 / 24.0),
        _mm256_set1_pd(1.0 / 6.0),
        _mm256_set1_pd(0.5),
        _mm256_set1_pd(1.0),
        _mm256_set1_pd(1.0),
    };
    __m256d p = c[0];
    for (int i = 1; i < 9; ++i) {
        p = fmadd_pd(p, r, c[i]);
    }

    __m128i ki32 = _mm256_cvtpd_epi32(fk);
    __m256i ki64 = _mm256_cvtepi32_epi64(ki32);
    ki64 = _mm256_add_epi64(ki64, _mm256_set1_epi64x(1023LL));
    ki64 = _mm256_slli_epi64(ki64, 52);
    return _mm256_mul_pd(p, _mm256_castsi256_pd(ki64));
}

} // namespace

BatchResult process_batch_avx2(
    int T,
    int N,
    const SubstepData* ss,
    const ObsMeta*     obs,
    const double*      W_inc,
    const double*      BB_u,
    double S0,
    double notional,
    double log_ki,
    double log_strike,
    double df_T,
    double strike,
    bool   continuous_ki,
    int64_t* ac_counts_per_obs)
{
    __m256d pv_v  = _mm256_setzero_pd();
    __m256d alive = _mm256_castsi256_pd(_mm256_set1_epi64x(-1LL));
    __m256d ki_hit = _mm256_setzero_pd();
    __m256d log_s = _mm256_setzero_pd();
    __m256d prev  = _mm256_setzero_pd();

    int obs_cursor = 0;

    for (int k = 0; k < T; ++k) {
        __m256d mu_v = _mm256_set1_pd(ss[k].mu);
        __m256d dw   = _mm256_loadu_pd(&W_inc[k * 4]);
        prev  = log_s;
        log_s = _mm256_add_pd(log_s, _mm256_add_pd(mu_v, dw));

        __m256d logki_v = _mm256_set1_pd(log_ki);
        if (continuous_ki) {
            __m256d hit_now   = cmp_le(log_s, logki_v);
            __m256d both_above = _mm256_and_pd(cmp_gt(prev, logki_v), cmp_gt(log_s, logki_v));
            __m256d a_bb = _mm256_sub_pd(prev, logki_v);
            __m256d b_bb = _mm256_sub_pd(log_s, logki_v);
            __m256d nu2v = _mm256_set1_pd(ss[k].nu2);
            __m256d arg  = _mm256_div_pd(
                _mm256_mul_pd(_mm256_mul_pd(a_bb, b_bb), _mm256_set1_pd(-2.0)),
                nu2v);
            __m256d bb_prob = exp_pd(arg);
            __m256d u = BB_u ? _mm256_loadu_pd(&BB_u[k * 4]) : _mm256_set1_pd(1.0);
            __m256d bridge_hit = _mm256_and_pd(both_above, cmp_lt(u, bb_prob));
            __m256d new_hits   = _mm256_or_pd(hit_now, bridge_hit);
            ki_hit = _mm256_or_pd(ki_hit, _mm256_and_pd(alive, new_hits));
        } else {
            // Discrete KI check only at observation sub-steps (matches pre-substep semantics)
            if (obs_cursor < N && k == obs[obs_cursor].substep_end_idx) {
                __m256d hit_now = cmp_le(log_s, logki_v);
                ki_hit = _mm256_or_pd(ki_hit, _mm256_and_pd(alive, hit_now));
            }
        }

        if (obs_cursor < N && k == obs[obs_cursor].substep_end_idx) {
            __m256d logac_v     = _mm256_set1_pd(obs[obs_cursor].log_ac);
            __m256d call_mask   = _mm256_and_pd(alive, cmp_ge(log_s, logac_v));
            __m256d coupon_pv_v = _mm256_set1_pd(obs[obs_cursor].coupon_payoff);
            pv_v = _mm256_add_pd(pv_v, _mm256_and_pd(call_mask, coupon_pv_v));

            int mask_bits = _mm256_movemask_pd(call_mask);
            ac_counts_per_obs[obs_cursor] += popcount32(static_cast<unsigned int>(mask_bits));
            alive = _mm256_andnot_pd(call_mask, alive);
            ++obs_cursor;
        }
    }

    __m256d sT_ratio = exp_pd(log_s);
    __m256d below_k  = cmp_lt(log_s, _mm256_set1_pd(log_strike));
    __m256d loss_v = _mm256_mul_pd(
        _mm256_mul_pd(
            _mm256_set1_pd(notional),
            _mm256_div_pd(sT_ratio, _mm256_set1_pd(strike))),
        _mm256_set1_pd(df_T));
    __m256d par_v   = _mm256_set1_pd(notional * df_T);
    __m256d is_loss = _mm256_and_pd(_mm256_and_pd(alive, ki_hit), below_k);
    __m256d alive_not_loss = _mm256_andnot_pd(is_loss, alive);
    pv_v = _mm256_add_pd(pv_v, _mm256_and_pd(is_loss, loss_v));
    pv_v = _mm256_add_pd(pv_v, _mm256_and_pd(alive_not_loss, par_v));

    __m256d sT     = _mm256_mul_pd(_mm256_set1_pd(S0), sT_ratio);
    __m256d kS     = _mm256_set1_pd(strike * S0);
    __m256d put_raw = _mm256_sub_pd(kS, sT);
    __m256d pos_mask = cmp_gt(put_raw, _mm256_setzero_pd());
    __m256d put_pay  = _mm256_mul_pd(_mm256_and_pd(pos_mask, put_raw), _mm256_set1_pd(df_T));
    __m256d ki_alive = _mm256_and_pd(alive, ki_hit);

    BatchResult out{};
    out.sum_pv    = hsum(pv_v);
    out.sum_pv2   = hsum(_mm256_mul_pd(pv_v, pv_v));
    out.sum_cv    = hsum(put_pay);
    out.sum_cv2   = hsum(_mm256_mul_pd(put_pay, put_pay));
    out.sum_pvcv  = hsum(_mm256_mul_pd(pv_v, put_pay));
    int alive_bits    = _mm256_movemask_pd(alive);
    int called        = 4 - popcount32(static_cast<unsigned int>(alive_bits));
    int ki_alive_bits = _mm256_movemask_pd(ki_alive);
    out.sum_ac_count = static_cast<double>(called);
    out.sum_ki_count = static_cast<double>(popcount32(static_cast<unsigned int>(ki_alive_bits)));
    return out;
}

} // namespace simd_prod
