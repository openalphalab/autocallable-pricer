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

static inline __m512d fmadd_pd(__m512d a, __m512d b, __m512d c) {
#if defined(__FMA__)
    return _mm512_fmadd_pd(a, b, c);
#else
    return _mm512_add_pd(_mm512_mul_pd(a, b), c);
#endif
}

static inline double hsum(__m512d v) {
    return _mm512_reduce_add_pd(v);
}

static inline __m512d exp_pd(__m512d x) {
    const __m512d hi = _mm512_set1_pd(700.0);
    const __m512d lo = _mm512_set1_pd(-700.0);
    x = _mm512_min_pd(x, hi);
    x = _mm512_max_pd(x, lo);

    const __m512d LOG2E = _mm512_set1_pd(1.4426950408889634);
    const __m512d LN2 = _mm512_set1_pd(0.6931471805599453);
    __m512d fk = _mm512_roundscale_pd(_mm512_mul_pd(x, LOG2E), _MM_FROUND_TO_NEAREST_INT);
    __m512d r = _mm512_sub_pd(x, _mm512_mul_pd(fk, LN2));

    const __m512d c[] = {
        _mm512_set1_pd(1.0 / 40320.0),
        _mm512_set1_pd(1.0 / 5040.0),
        _mm512_set1_pd(1.0 / 720.0),
        _mm512_set1_pd(1.0 / 120.0),
        _mm512_set1_pd(1.0 / 24.0),
        _mm512_set1_pd(1.0 / 6.0),
        _mm512_set1_pd(0.5),
        _mm512_set1_pd(1.0),
        _mm512_set1_pd(1.0),
    };
    __m512d p = c[0];
    for (int i = 1; i < 9; ++i) {
        p = fmadd_pd(p, r, c[i]);
    }

    __m256i ki32 = _mm512_cvtpd_epi32(fk);
    __m512i ki64 = _mm512_cvtepi32_epi64(ki32);
    ki64 = _mm512_add_epi64(ki64, _mm512_set1_epi64(1023LL));
    ki64 = _mm512_slli_epi64(ki64, 52);
    return _mm512_mul_pd(p, _mm512_castsi512_pd(ki64));
}

} // namespace

BatchResult process_batch_avx512(
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
    __m512d  pv_v     = _mm512_setzero_pd();
    __mmask8 alive    = 0xFF;
    __mmask8 ki_hit_m = 0;
    __m512d  log_s    = _mm512_setzero_pd();
    __m512d  prev     = _mm512_setzero_pd();

    int obs_cursor = 0;

    for (int k = 0; k < T; ++k) {
        __m512d mu_v = _mm512_set1_pd(ss[k].mu);
        __m512d dw   = _mm512_loadu_pd(&W_inc[k * 8]);
        prev  = log_s;
        log_s = _mm512_add_pd(log_s, _mm512_add_pd(mu_v, dw));

        __m512d logki_v = _mm512_set1_pd(log_ki);
        if (continuous_ki) {
            __mmask8 hit_now    = _mm512_cmp_pd_mask(log_s, logki_v, _CMP_LE_OQ);
            __mmask8 both_above = _mm512_cmp_pd_mask(prev, logki_v, _CMP_GT_OQ) &
                                  _mm512_cmp_pd_mask(log_s, logki_v, _CMP_GT_OQ);
            __m512d a_bb  = _mm512_sub_pd(prev, logki_v);
            __m512d b_bb  = _mm512_sub_pd(log_s, logki_v);
            __m512d nu2v  = _mm512_set1_pd(ss[k].nu2);
            __m512d arg   = _mm512_div_pd(
                _mm512_mul_pd(_mm512_mul_pd(a_bb, b_bb), _mm512_set1_pd(-2.0)),
                nu2v);
            __m512d bb_prob    = exp_pd(arg);
            __m512d u          = BB_u ? _mm512_loadu_pd(&BB_u[k * 8]) : _mm512_set1_pd(1.0);
            __mmask8 bridge_hit = _mm512_cmp_pd_mask(u, bb_prob, _CMP_LT_OQ) & both_above;
            __mmask8 new_hits   = (hit_now | bridge_hit) & alive;
            ki_hit_m |= new_hits;
        } else {
            if (obs_cursor < N && k == obs[obs_cursor].substep_end_idx) {
                __mmask8 hit_now = _mm512_cmp_pd_mask(log_s, logki_v, _CMP_LE_OQ) & alive;
                ki_hit_m |= hit_now;
            }
        }

        if (obs_cursor < N && k == obs[obs_cursor].substep_end_idx) {
            __m512d logac_v  = _mm512_set1_pd(obs[obs_cursor].log_ac);
            __mmask8 call_mask = _mm512_cmp_pd_mask(log_s, logac_v, _CMP_GE_OQ) & alive;
            ac_counts_per_obs[obs_cursor] += popcount32(static_cast<unsigned int>(call_mask));
            __m512d coupon_v = _mm512_set1_pd(obs[obs_cursor].coupon_payoff);
            pv_v  = _mm512_mask_add_pd(pv_v, call_mask, pv_v, coupon_v);
            alive &= static_cast<__mmask8>(~call_mask);
            ++obs_cursor;
        }
    }

    __m512d sT_ratio = exp_pd(log_s);
    __mmask8 below_k = _mm512_cmp_pd_mask(log_s, _mm512_set1_pd(log_strike), _CMP_LT_OQ);
    __m512d loss_v = _mm512_mul_pd(
        _mm512_mul_pd(
            _mm512_set1_pd(notional),
            _mm512_div_pd(sT_ratio, _mm512_set1_pd(strike))),
        _mm512_set1_pd(df_T));
    __m512d par_v = _mm512_set1_pd(notional * df_T);
    __mmask8 is_loss = alive & ki_hit_m & below_k;
    __mmask8 alive_not_loss = alive & static_cast<__mmask8>(~is_loss);
    pv_v = _mm512_mask_add_pd(pv_v, is_loss, pv_v, loss_v);
    pv_v = _mm512_mask_add_pd(pv_v, alive_not_loss, pv_v, par_v);

    __m512d sT     = _mm512_mul_pd(_mm512_set1_pd(S0), sT_ratio);
    __m512d kS     = _mm512_set1_pd(strike * S0);
    __m512d put_raw = _mm512_sub_pd(kS, sT);
    __mmask8 pos   = _mm512_cmp_pd_mask(put_raw, _mm512_setzero_pd(), _CMP_GT_OQ);
    __m512d put_pay = _mm512_mul_pd(_mm512_maskz_mov_pd(pos, put_raw), _mm512_set1_pd(df_T));

    BatchResult out{};
    out.sum_pv    = hsum(pv_v);
    out.sum_pv2   = hsum(_mm512_mul_pd(pv_v, pv_v));
    out.sum_cv    = hsum(put_pay);
    out.sum_cv2   = hsum(_mm512_mul_pd(put_pay, put_pay));
    out.sum_pvcv  = hsum(_mm512_mul_pd(pv_v, put_pay));
    out.sum_ac_count = static_cast<double>(8 - popcount32(static_cast<unsigned int>(alive)));
    out.sum_ki_count = static_cast<double>(
        popcount32(static_cast<unsigned int>(alive & ki_hit_m)));
    return out;
}

} // namespace simd_prod
