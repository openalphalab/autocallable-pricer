"""
deamerican.py
=============
Production-grade de-Americanization of equity / ETF option prices.

PIPELINE (per option quote)
---------------------------
    mid = (bid + ask) / 2
    sigma_american = solve [ American_price(sigma; S, K, T, r, divs) == mid ]
    european_mid   = BlackScholes(sigma_american; S, K, T, r, q_eff)

Two American pricers are provided:

    * BAW   — Barone-Adesi-Whaley closed-form approximation.
              Vectorizable, ~100 ns per call under numba. Use for production
              throughput and risk grids.
    * CRR   — Cox-Ross-Rubinstein binomial tree with Haug's escrowed-dividend
              construction. Exact to O(1/N). Use as reference to validate BAW.

Discrete dividends are handled correctly in both: cash dividends with known
ex-dates are PV'd to today and subtracted from the forward (equivalent
continuous yield is used for BAW; the tree shifts at each ex-date for CRR).

Risk-free rates are built from the U.S. Treasury CMT curve using the Cboe
methodology: bounded natural cubic spline in BEY space, then
    APY = (1 + BEY/2)^2 - 1
    r   = ln(1 + APY)   (continuously compounded)

All hot loops are numba-JIT compiled (cache=True). Top-level chain processing
uses prange for trivial parallel scan across options.

Public API
----------
    YieldCurve.from_cmt_row(row)
        Build a bounded cubic-spline curve from one row of the CMT CSV.
    DividendSchedule(ex_times, amounts)
        Discrete cash dividends, times in years from valuation date.
    implied_spot_from_parity(chain_df, T, curve, divs)
        Recover S from ATM put/call parity (no Yahoo fetch required).
    deamericanize_chain(chain_df, S, curve, divs, method='baw', ...)
        Vectorized pipeline. Adds 'american_iv' and 'european_mid' columns.

Author: production-ready reference implementation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.interpolate import CubicSpline


# ======================================================================
#                        CONSTANTS AND MATH HELPERS
# ======================================================================

# CMT tenors in days, per Cboe VIX Mathematics Methodology, section 2.1.
_CMT_TENORS_DAYS = np.array(
    [30, 60, 91, 182, 365, 730, 1095, 1825, 2555, 3650, 7300, 10950],
    dtype=np.float64,
)
_CMT_LABELS = [
    "1_month", "2_month", "3_month", "6_month", "1_year", "2_year",
    "3_year", "5_year", "7_year", "10_year", "20_year", "30_year",
]

MINUTES_PER_YEAR = 365.0 * 1440.0   # Cboe convention
SQRT_2PI = math.sqrt(2.0 * math.pi)


@njit(cache=True, inline="always")
def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via erf; numba-safe."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@njit(cache=True, inline="always")
def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


# ======================================================================
#                      BLACK-SCHOLES (GENERALIZED)
# ======================================================================
# We use the Generalized BS formulation with cost-of-carry b = r - q.
# That's the right vehicle because BAW is written in those terms and discrete
# dividends collapse to a single equivalent-yield q for the life of the option.

@njit(cache=True, inline="always")
def bs_price(S: float, K: float, T: float, r: float, q: float,
             sigma: float, is_call: bool) -> float:
    """Black-Scholes-Merton price with continuous dividend yield q."""
    if T <= 0.0 or sigma <= 0.0:
        if is_call:
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    b = r - q
    vsqrt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (b + 0.5 * sigma * sigma) * T) / vsqrt
    d2 = d1 - vsqrt
    if is_call:
        return S * math.exp((b - r) * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp((b - r) * T) * _norm_cdf(-d1)


@njit(cache=True, inline="always")
def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    b = r - q
    vsqrt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (b + 0.5 * sigma * sigma) * T) / vsqrt
    return S * math.exp((b - r) * T) * _norm_pdf(d1) * math.sqrt(T)


# ======================================================================
#                  BARONE-ADESI-WHALEY AMERICAN PRICE
# ======================================================================
# Reference: Barone-Adesi & Whaley (1987). Formulas follow Haug,
# "Complete Guide to Option Pricing Formulas" ch. 1.
#
# Notation (avoiding strike/parameter clash):
#     b   = r - q                        cost of carry
#     N   = 2 b / sigma^2
#     M   = 2 r / sigma^2
#     Kw  = 1 - exp(-r T)                (the "wedge" factor)
#     q2  = (-(N-1) + sqrt((N-1)^2 + 4M/Kw)) / 2
#     q1  = (-(N-1) - sqrt((N-1)^2 + 4M/Kw)) / 2
#
# Critical price S* (call) / S** (put) is the stock price at which immediate
# exercise exactly matches early-exercise-adjusted continuation. Solved by
# Newton-Raphson from Haug's suggested seed (perpetual-option limit blend).

_BAW_NEWTON_TOL = 1e-6
_BAW_NEWTON_MAX = 50


@njit(cache=True, inline="always")
def _baw_seed_call(S_inf: float, K: float, b: float, T: float,
                   sigma: float) -> float:
    h2 = -(b * T + 2.0 * sigma * math.sqrt(T)) * K / (S_inf - K)
    return K + (S_inf - K) * (1.0 - math.exp(h2))


@njit(cache=True, inline="always")
def _baw_seed_put(S_inf: float, K: float, b: float, T: float,
                  sigma: float) -> float:
    h1 = (b * T - 2.0 * sigma * math.sqrt(T)) * K / (K - S_inf)
    return S_inf + (K - S_inf) * math.exp(h1)


@njit(cache=True, inline="always")
def _bs_with_intrinsic_floor(S: float, K: float, T: float, r: float, q: float,
                             sigma: float, is_call: bool) -> float:
    euro = bs_price(S, K, T, r, q, max(sigma, 1e-8), is_call)
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    return euro if euro > intrinsic else intrinsic


@njit(cache=True)
def baw_price(S: float, K: float, T: float, r: float, q: float,
              sigma: float, is_call: bool) -> float:
    """American option price via Barone-Adesi-Whaley approximation."""
    # --- degenerate cases ---
    if T <= 0.0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    # For very small sigma the BAW critical-price seed becomes unstable.
    # In that regime, use the European price with an intrinsic floor.
    if sigma < 1e-2:
        return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, is_call)

    b = r - q

    # --- No early exercise premium for calls when b >= r (q <= 0). Rare. ---
    # For q > 0 (dividends) American call can be optimal to exercise.
    # For q <= 0 and vanilla call, American == European. For put, always check.
    if is_call and b >= r:
        return bs_price(S, K, T, r, q, sigma, True)

    s2 = sigma * sigma
    N_ = 2.0 * b / s2
    M_ = 2.0 * r / s2
    Kw = 1.0 - math.exp(-r * T)
    if abs(Kw) < 1e-12:
        return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, is_call)

    if is_call:
        # Solve for critical S*
        disc_inf = (N_ - 1.0) ** 2 + 4.0 * M_
        if disc_inf <= 0.0:
            return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, True)
        q2_inf = 0.5 * (-(N_ - 1.0) + math.sqrt(disc_inf))
        if (not math.isfinite(q2_inf)) or q2_inf <= 2.0 + 1e-12:
            return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, True)
        denom_inf = 1.0 - 2.0 / q2_inf
        if abs(denom_inf) < 1e-12:
            return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, True)
        S_inf = K / denom_inf
        if (not math.isfinite(S_inf)) or S_inf <= K:
            return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, True)
        Sx = _baw_seed_call(S_inf, K, b, T, sigma)
        if (not math.isfinite(Sx)) or Sx <= K:
            return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, True)

        disc = (N_ - 1.0) ** 2 + 4.0 * M_ / Kw
        if disc <= 0.0:
            return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, True)
        q2 = 0.5 * (-(N_ - 1.0) + math.sqrt(disc))
        if (not math.isfinite(q2)) or q2 <= 0.0:
            return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, True)

        for _ in range(_BAW_NEWTON_MAX):
            if (not math.isfinite(Sx)) or Sx <= 0.0:
                return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, True)
            vsqrt = sigma * math.sqrt(T)
            if vsqrt <= 0.0:
                return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, True)
            d1 = (math.log(Sx / K) + (b + 0.5 * s2) * T) / vsqrt
            LHS = Sx - K
            euro_c = bs_price(Sx, K, T, r, q, sigma, True)
            RHS = euro_c + (1.0 - math.exp((b - r) * T) * _norm_cdf(d1)) * Sx / q2
            # derivative of RHS - LHS w.r.t. Sx
            bi = (math.exp((b - r) * T) * _norm_cdf(d1) * (1.0 - 1.0 / q2)
                  + (1.0 - math.exp((b - r) * T) * _norm_pdf(d1) / vsqrt) / q2)
            f = LHS - RHS
            fp = 1.0 - bi
            if (not math.isfinite(f)) or (not math.isfinite(fp)) or abs(fp) < 1e-12:
                return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, True)
            if abs(f) < _BAW_NEWTON_TOL * K:
                break
            Sx = Sx - f / fp

        if S >= Sx:
            return S - K
        vsqrt = sigma * math.sqrt(T)
        d1 = (math.log(Sx / K) + (b + 0.5 * s2) * T) / vsqrt
        A2 = (Sx / q2) * (1.0 - math.exp((b - r) * T) * _norm_cdf(d1))
        if not math.isfinite(A2):
            return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, True)
        return bs_price(S, K, T, r, q, sigma, True) + A2 * (S / Sx) ** q2

    else:
        # --- American put ---
        disc_inf = (N_ - 1.0) ** 2 + 4.0 * M_
        if disc_inf <= 0.0:
            return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, False)
        q1_inf = 0.5 * (-(N_ - 1.0) - math.sqrt(disc_inf))
        if (not math.isfinite(q1_inf)) or abs(q1_inf) < 1e-12:
            return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, False)
        denom_inf = 1.0 - 2.0 / q1_inf
        if abs(denom_inf) < 1e-12:
            return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, False)
        S_inf = K / denom_inf
        if (not math.isfinite(S_inf)) or S_inf <= 0.0 or S_inf >= K:
            return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, False)
        Sxx = _baw_seed_put(S_inf, K, b, T, sigma)
        if (not math.isfinite(Sxx)) or Sxx <= 0.0 or Sxx >= K:
            return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, False)

        disc = (N_ - 1.0) ** 2 + 4.0 * M_ / Kw
        if disc <= 0.0:
            return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, False)
        q1 = 0.5 * (-(N_ - 1.0) - math.sqrt(disc))
        if (not math.isfinite(q1)) or abs(q1) < 1e-12:
            return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, False)

        for _ in range(_BAW_NEWTON_MAX):
            if (not math.isfinite(Sxx)) or Sxx <= 0.0:
                return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, False)
            vsqrt = sigma * math.sqrt(T)
            if vsqrt <= 0.0:
                return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, False)
            d1 = (math.log(Sxx / K) + (b + 0.5 * s2) * T) / vsqrt
            LHS = K - Sxx
            euro_p = bs_price(Sxx, K, T, r, q, sigma, False)
            RHS = euro_p - (1.0 - math.exp((b - r) * T) * _norm_cdf(-d1)) * Sxx / q1
            bi = (-math.exp((b - r) * T) * _norm_cdf(-d1) * (1.0 - 1.0 / q1)
                  - (1.0 + math.exp((b - r) * T) * _norm_pdf(-d1) / vsqrt) / q1)
            f = LHS - RHS
            fp = -1.0 - bi
            if (not math.isfinite(f)) or (not math.isfinite(fp)) or abs(fp) < 1e-12:
                return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, False)
            if abs(f) < _BAW_NEWTON_TOL * K:
                break
            Sxx = Sxx - f / fp

        if S <= Sxx:
            return K - S
        vsqrt = sigma * math.sqrt(T)
        d1 = (math.log(Sxx / K) + (b + 0.5 * s2) * T) / vsqrt
        A1 = -(Sxx / q1) * (1.0 - math.exp((b - r) * T) * _norm_cdf(-d1))
        if not math.isfinite(A1):
            return _bs_with_intrinsic_floor(S, K, T, r, q, sigma, False)
        return bs_price(S, K, T, r, q, sigma, False) + A1 * (S / Sxx) ** q1


# ======================================================================
#     COX-ROSS-RUBINSTEIN BINOMIAL WITH ESCROWED DISCRETE DIVIDENDS
# ======================================================================
# Haug's "escrowed dividend" construction:
#   S_hat(0) = S0 - sum_i D_i * exp(-r t_i)   (PV of divs in (0,T])
#   Build CRR tree on S_hat with constant r, sigma.
#   At each node, actual stock = S_hat_node + PV_remaining_divs(t_node, T).
#   Exercise decision uses actual stock; continuation is plain BS-style
#   risk-neutral expectation.
#
# This gives O(N^2) cost for N steps, no path-enumeration needed.

@njit(cache=True)
def _pv_divs_remaining(t_now: float, T_exp: float, r: float,
                       div_times: np.ndarray, div_amounts: np.ndarray) -> float:
    """Sum of D_i * exp(-r*(t_i - t_now)) for divs with t_now < t_i <= T_exp."""
    total = 0.0
    for i in range(div_times.shape[0]):
        ti = div_times[i]
        if ti > t_now and ti <= T_exp + 1e-12:
            total += div_amounts[i] * math.exp(-r * (ti - t_now))
    return total


@njit(cache=True)
def crr_price(S: float, K: float, T: float, r: float, sigma: float,
              div_times: np.ndarray, div_amounts: np.ndarray,
              is_call: bool, n_steps: int, american: bool = True) -> float:
    """CRR binomial with discrete cash dividends (escrowed-dividend model).

    Parameters
    ----------
    div_times, div_amounts : np.ndarray[float64]
        Ex-dates in years from valuation and cash amounts. Only entries with
        0 < t_i <= T are used. Pass empty arrays for no dividends.
    """
    if T <= 0.0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)

    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp(r * dt) - d) / (u - d)
    if p <= 0.0 or p >= 1.0:
        # Numerical failure — fall back to BS with q_eff
        pv = _pv_divs_remaining(0.0, T, r, div_times, div_amounts)
        q_eff = -math.log((S - pv) / S) / T if S > pv else 0.0
        return bs_price(S, K, T, r, q_eff, sigma, is_call)

    # Pseudo-stock (PV-div adjusted) at t=0
    pv0 = _pv_divs_remaining(0.0, T, r, div_times, div_amounts)
    S_hat0 = S - pv0

    # Terminal payoffs
    values = np.empty(n_steps + 1, dtype=np.float64)
    for j in range(n_steps + 1):
        # At t=T: no remaining divs, so actual stock == pseudo-stock
        ST = S_hat0 * (u ** j) * (d ** (n_steps - j))
        if is_call:
            values[j] = max(ST - K, 0.0)
        else:
            values[j] = max(K - ST, 0.0)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        t_now = i * dt
        pv_remain = _pv_divs_remaining(t_now, T, r, div_times, div_amounts)
        for j in range(i + 1):
            cont = disc * (p * values[j + 1] + (1.0 - p) * values[j])
            if american:
                S_hat_node = S_hat0 * (u ** j) * (d ** (i - j))
                S_node = S_hat_node + pv_remain
                if is_call:
                    intrinsic = S_node - K
                else:
                    intrinsic = K - S_node
                values[j] = cont if cont > intrinsic else intrinsic
            else:
                values[j] = cont
    return values[0]


# --- Continuous-yield CRR variant (no discrete divs) ------------------
# When we've extracted q continuously from put-call parity (backtest case),
# we don't want to synthesise fake discrete ex-dates. Here we just run a
# standard CRR tree with cost-of-carry (r - q) for the drift and r for
# discounting — the textbook generalised-BS formulation.

@njit(cache=True)
def crr_price_cts(S: float, K: float, T: float, r: float, q: float,
                  sigma: float, is_call: bool, n_steps: int,
                  american: bool = True) -> float:
    """CRR binomial with continuous dividend yield `q`."""
    if T <= 0.0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    # Risk-neutral with yield: E[S(t+dt)] = S(t) * exp((r-q)*dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if p <= 0.0 or p >= 1.0:
        return bs_price(S, K, T, r, q, sigma, is_call)

    values = np.empty(n_steps + 1, dtype=np.float64)
    for j in range(n_steps + 1):
        ST = S * (u ** j) * (d ** (n_steps - j))
        if is_call:
            values[j] = max(ST - K, 0.0)
        else:
            values[j] = max(K - ST, 0.0)

    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            cont = disc * (p * values[j + 1] + (1.0 - p) * values[j])
            if american:
                S_node = S * (u ** j) * (d ** (i - j))
                intrinsic = (S_node - K) if is_call else (K - S_node)
                values[j] = cont if cont > intrinsic else intrinsic
            else:
                values[j] = cont
    return values[0]


@njit(cache=True)
def implied_vol_crr_cts(price: float, S: float, K: float, T: float,
                        r: float, q: float, is_call: bool,
                        n_steps: int) -> float:
    if T <= 0.0 or price <= 0.0:
        return np.nan
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if price < intrinsic - 1e-10:
        return np.nan
    # Dummy div arrays for Brent's untyped signature compatibility
    empty = np.empty(0, dtype=np.float64)
    f_lo = crr_price_cts(S, K, T, r, q, _IV_LO, is_call, n_steps, True)
    f_hi = crr_price_cts(S, K, T, r, q, _IV_HI, is_call, n_steps, True)
    if abs(price - f_lo) < 1e-8 * max(f_lo, 1.0):
        return np.nan
    if (f_hi - f_lo) < 1e-6 * max(price, 1.0):
        return np.nan

    # Inline bisection/Brent for continuous-yield CRR
    a, b = _IV_LO, _IV_HI
    fa = f_lo - price
    fb = f_hi - price
    if fa * fb > 0.0:
        return np.nan
    for _ in range(_IV_MAX_ITER):
        m = 0.5 * (a + b)
        fm = crr_price_cts(S, K, T, r, q, m, is_call, n_steps, True) - price
        if abs(fm) < _IV_TOL or (b - a) < _IV_TOL:
            return m
        if fa * fm < 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


# ======================================================================
#                 IMPLIED VOLATILITY SOLVER (BRENT)
# ======================================================================
# We use a pure-numba Brent's method. This avoids scipy.optimize overhead
# inside hot loops and gives us ~0.5-2 us per solve.

_IV_LO = 1e-4
_IV_HI = 5.0
_IV_TOL = 1e-8
_IV_MAX_ITER = 100
_IV_SCAN_STEPS = 96


@njit(cache=True)
def _brent(f_val_lo: float, f_val_hi: float,
           target: float, S: float, K: float, T: float, r: float, q: float,
           is_call: bool, use_baw: bool,
           div_times: np.ndarray, div_amounts: np.ndarray, n_steps: int,
           lo: float, hi: float) -> float:
    """Classical Brent-Dekker root-finder specialised for IV solves.

    We inline the option-price function because numba can't take a Python
    callable through a jitted boundary without overhead.
    """
    a, b = lo, hi
    fa, fb = f_val_lo - target, f_val_hi - target

    if math.isnan(fa) or math.isnan(fb):
        return np.nan
    if fa * fb > 0.0:
        return np.nan                  # no sign change — no valid IV

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    mflag = True
    d = 0.0

    for _ in range(_IV_MAX_ITER):
        if abs(fb) < _IV_TOL:
            return b

        use_iqi = (fa != fb) and (fa != fc) and (fb != fc)
        if use_iqi:
            # inverse quadratic interpolation
            s = (a * fb * fc / ((fa - fb) * (fa - fc))
                 + b * fa * fc / ((fb - fa) * (fb - fc))
                 + c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # secant
            if fb == fa:
                s = 0.5 * (a + b)
            else:
                s = b - fb * (b - a) / (fb - fa)

        cond1 = not ((3.0 * a + b) / 4.0 < s < b or b < s < (3.0 * a + b) / 4.0)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2.0
        cond3 = (not mflag) and abs(s - b) >= abs(c - d) / 2.0
        cond4 = mflag and abs(b - c) < _IV_TOL
        cond5 = (not mflag) and abs(c - d) < _IV_TOL
        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = 0.5 * (a + b)
            mflag = True
        else:
            mflag = False

        if use_baw:
            fs = baw_price(S, K, T, r, q, s, is_call) - target
        else:
            fs = crr_price(S, K, T, r, s, div_times, div_amounts,
                           is_call, n_steps, True) - target

        d, c, fc = c, b, fb
        if fa * fs < 0.0:
            b, fb = s, fs
        else:
            a, fa = s, fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    return b


@njit(cache=True)
def implied_vol_baw(price: float, S: float, K: float, T: float,
                    r: float, q: float, is_call: bool) -> float:
    """Solve baw_price(sigma) = price for sigma in (_IV_LO, _IV_HI)."""
    if T <= 0.0 or price <= 0.0:
        return np.nan
    # Bracket sanity
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if price < intrinsic - 1e-10:
        return np.nan
    f_lo = baw_price(S, K, T, r, q, _IV_LO, is_call)
    f_hi = baw_price(S, K, T, r, q, _IV_HI, is_call)
    # If price sits on the exercise-dominated plateau (BAW returns pure
    # intrinsic across a range of sigma), IV is not uniquely recoverable.
    if abs(price - f_lo) < 1e-8 * max(f_lo, 1.0):
        return np.nan
    if (f_hi - f_lo) < 1e-6 * max(price, 1.0):
        return np.nan
    empty = np.empty(0, dtype=np.float64)

    # BAW can be non-monotone in sigma for some q > r, long-dated call
    # regimes. Scan from low sigma upward and use the first bracket so the
    # solver prefers the lower-vol root when multiple solutions exist.
    log_lo = math.log(_IV_LO)
    log_hi = math.log(_IV_HI)
    prev_sig = _IV_LO
    prev_val = f_lo
    prev_res = prev_val - price
    use_lo = _IV_LO
    use_hi = _IV_HI
    use_lo_val = f_lo
    use_hi_val = f_hi
    found = False

    for i in range(1, _IV_SCAN_STEPS + 1):
        sig = math.exp(log_lo + (log_hi - log_lo) * i / _IV_SCAN_STEPS)
        val = baw_price(S, K, T, r, q, sig, is_call)
        if math.isnan(val):
            prev_sig = sig
            prev_val = val
            prev_res = np.nan
            continue
        res = val - price
        if abs(res) < _IV_TOL:
            return sig
        if (not math.isnan(prev_res)) and prev_res * res <= 0.0:
            use_lo = prev_sig
            use_hi = sig
            use_lo_val = prev_val
            use_hi_val = val
            found = True
            break
        prev_sig = sig
        prev_val = val
        prev_res = res

    if not found:
        use_lo = _IV_LO
        use_hi = _IV_HI
        use_lo_val = f_lo
        use_hi_val = f_hi

    return _brent(use_lo_val, use_hi_val, price, S, K, T, r, q, is_call,
                  True, empty, empty, 0, use_lo, use_hi)


@njit(cache=True)
def implied_vol_crr(price: float, S: float, K: float, T: float, r: float,
                    div_times: np.ndarray, div_amounts: np.ndarray,
                    is_call: bool, n_steps: int) -> float:
    """Solve crr_price(sigma) = price. CRR carries divs natively."""
    if T <= 0.0 or price <= 0.0:
        return np.nan
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if price < intrinsic - 1e-10:
        return np.nan
    f_lo = crr_price(S, K, T, r, _IV_LO, div_times, div_amounts,
                     is_call, n_steps, True)
    f_hi = crr_price(S, K, T, r, _IV_HI, div_times, div_amounts,
                     is_call, n_steps, True)
    if abs(price - f_lo) < 1e-8 * max(f_lo, 1.0):
        return np.nan
    if (f_hi - f_lo) < 1e-6 * max(price, 1.0):
        return np.nan
    return _brent(f_lo, f_hi, price, S, K, T, r, 0.0, is_call,
                  False, div_times, div_amounts, n_steps, _IV_LO, _IV_HI)


# ======================================================================
#        VECTORIZED CHAIN PROCESSING (parallel prange)
# ======================================================================

@njit(cache=True, parallel=True)
def _process_chain_baw(mids: np.ndarray, Ss: np.ndarray, Ks: np.ndarray,
                       Ts: np.ndarray, rs: np.ndarray, qs: np.ndarray,
                       is_calls: np.ndarray) -> tuple:
    n = mids.shape[0]
    ivs = np.full(n, np.nan, dtype=np.float64)
    euros = np.full(n, np.nan, dtype=np.float64)
    for i in prange(n):
        iv = implied_vol_baw(mids[i], Ss[i], Ks[i], Ts[i], rs[i], qs[i], bool(is_calls[i]))
        if not math.isnan(iv):
            ivs[i] = iv
            euros[i] = bs_price(Ss[i], Ks[i], Ts[i], rs[i], qs[i], iv, bool(is_calls[i]))
    return ivs, euros


@njit(cache=True, parallel=True)
def _process_chain_crr(mids: np.ndarray, Ss: np.ndarray, Ks: np.ndarray,
                       Ts: np.ndarray, rs: np.ndarray, qs: np.ndarray,
                       is_calls: np.ndarray,
                       div_times: np.ndarray, div_amounts: np.ndarray,
                       n_steps: int) -> tuple:
    n = mids.shape[0]
    ivs = np.full(n, np.nan, dtype=np.float64)
    euros = np.full(n, np.nan, dtype=np.float64)
    for i in prange(n):
        # Filter divs to this option's life
        T_i = Ts[i]
        # Use full arrays; crr_price ignores divs outside (0, T] via the filter in _pv_divs_remaining
        iv = implied_vol_crr(mids[i], Ss[i], Ks[i], T_i, rs[i],
                             div_times, div_amounts, bool(is_calls[i]), n_steps)
        if not math.isnan(iv):
            ivs[i] = iv
            # Reprice European with equivalent yield q (same S, r, divs).
            # We use qs[i] which was computed from discrete divs in (0, T_i].
            euros[i] = bs_price(Ss[i], Ks[i], T_i, rs[i], qs[i], iv, bool(is_calls[i]))
    return ivs, euros


@njit(cache=True, parallel=True)
def _process_chain_crr_cts(mids: np.ndarray, Ss: np.ndarray, Ks: np.ndarray,
                           Ts: np.ndarray, rs: np.ndarray, qs: np.ndarray,
                           is_calls: np.ndarray, n_steps: int) -> tuple:
    n = mids.shape[0]
    ivs = np.full(n, np.nan, dtype=np.float64)
    euros = np.full(n, np.nan, dtype=np.float64)
    for i in prange(n):
        iv = implied_vol_crr_cts(mids[i], Ss[i], Ks[i], Ts[i],
                                 rs[i], qs[i], bool(is_calls[i]), n_steps)
        if not math.isnan(iv):
            ivs[i] = iv
            euros[i] = bs_price(Ss[i], Ks[i], Ts[i], rs[i], qs[i], iv,
                                bool(is_calls[i]))
    return ivs, euros


# ======================================================================
#                  YIELD CURVE: BOUNDED CUBIC SPLINE
# ======================================================================

@dataclass
class YieldCurve:
    """Bounded natural-cubic-spline over U.S. Treasury CMT rates.

    Implements the Cboe VIX methodology, section 2.1:
      1. Natural cubic spline over (tenor_days, BEY%)
      2. Clip the spline output to the band [min(CMT_i, CMT_{i+1}),
                                             max(CMT_i, CMT_{i+1})]
         on each interval.
      3. Linear extrapolation (also bounded) at the short end.
      4. BEY -> APY -> continuously compounded r.
    """
    tenors_days: np.ndarray           # [K]
    bey_pct: np.ndarray               # [K], CMT values in percent
    _spline: CubicSpline = field(init=False)
    _t_min: float = field(init=False)
    _lo_m: float = field(init=False)
    _lo_b: float = field(init=False)
    _hi_m: float = field(init=False)
    _hi_b: float = field(init=False)

    @classmethod
    def from_cmt_row(cls, row: pd.Series) -> "YieldCurve":
        """Build from one row of a rates CSV with columns _CMT_LABELS."""
        vals = []
        tens = []
        for t, lbl in zip(_CMT_TENORS_DAYS, _CMT_LABELS):
            v = row.get(lbl, np.nan)
            if pd.notna(v):
                vals.append(float(v))
                tens.append(float(t))
        if len(vals) < 3:
            raise ValueError(f"Need >=3 CMT points, got {len(vals)}")
        return cls(np.array(tens), np.array(vals))

    def __post_init__(self):
        order = np.argsort(self.tenors_days)
        self.tenors_days = self.tenors_days[order]
        self.bey_pct = self.bey_pct[order]
        self._spline = CubicSpline(
            self.tenors_days, self.bey_pct, bc_type="natural", extrapolate=True
        )
        self._t_min = float(self.tenors_days[0])

        # Linear extrapolation at short end per Cboe spec.
        # Lower bound slope: to next shorter point with CMT >= CMT_1.
        t1, c1 = self.tenors_days[0], self.bey_pct[0]
        m_lo = 0.0
        for i in range(1, len(self.tenors_days)):
            if self.bey_pct[i] >= c1:
                m_lo = (self.bey_pct[i] - c1) / (self.tenors_days[i] - t1)
                break
        self._lo_m = m_lo
        self._lo_b = c1 - m_lo * t1

        m_hi = 0.0
        for i in range(1, len(self.tenors_days)):
            if self.bey_pct[i] <= c1:
                m_hi = (self.bey_pct[i] - c1) / (self.tenors_days[i] - t1)
                break
        self._hi_m = m_hi
        self._hi_b = c1 - m_hi * t1

    def bey(self, t_days: float) -> float:
        """Bond-equivalent yield at tenor `t_days` (percent)."""
        if t_days >= self._t_min:
            # Interpolation — spline output, bounded by neighboring CMTs
            raw = float(self._spline(t_days))
            # Find bracketing knots
            j = np.searchsorted(self.tenors_days, t_days, side="right") - 1
            j = max(0, min(j, len(self.tenors_days) - 2))
            lo_cmt = min(self.bey_pct[j], self.bey_pct[j + 1])
            hi_cmt = max(self.bey_pct[j], self.bey_pct[j + 1])
            return min(hi_cmt, max(lo_cmt, raw))
        # Extrapolation at short end
        r_lo = self._lo_m * t_days + self._lo_b
        r_hi = self._hi_m * t_days + self._hi_b
        # Bounded between linear extrapolations
        return min(r_hi, max(r_lo, 0.5 * (r_lo + r_hi)))

    def r_continuous(self, t_years: float) -> float:
        """Continuously-compounded risk-free rate at tenor `t_years`."""
        if t_years <= 0.0:
            return 0.0
        bey = self.bey(t_years * 365.0) / 100.0          # pct -> decimal
        apy = (1.0 + bey / 2.0) ** 2 - 1.0               # Cboe BEY->APY
        return math.log(1.0 + apy)                        # APY->continuous


# ======================================================================
#                  DISCRETE DIVIDEND SCHEDULE
# ======================================================================

@dataclass
class DividendSchedule:
    """Cash dividends with known ex-dates, relative to a valuation date.

    Attributes
    ----------
    ex_times : np.ndarray[float64]   years from valuation date (strictly > 0)
    amounts  : np.ndarray[float64]   cash amount per share, same order
    """
    ex_times: np.ndarray
    amounts: np.ndarray

    def __post_init__(self):
        self.ex_times = np.asarray(self.ex_times, dtype=np.float64)
        self.amounts = np.asarray(self.amounts, dtype=np.float64)
        if self.ex_times.shape != self.amounts.shape:
            raise ValueError("ex_times and amounts must match in shape")
        order = np.argsort(self.ex_times)
        self.ex_times = self.ex_times[order]
        self.amounts = self.amounts[order]

    @classmethod
    def empty(cls) -> "DividendSchedule":
        return cls(np.zeros(0), np.zeros(0))

    @classmethod
    def from_ex_dates(cls, valuation_date: pd.Timestamp,
                      ex_dates: list, amounts: list) -> "DividendSchedule":
        valuation_date = pd.Timestamp(valuation_date)
        times = np.array([(pd.Timestamp(d) - valuation_date).days / 365.0
                          for d in ex_dates], dtype=np.float64)
        amts = np.asarray(amounts, dtype=np.float64)
        mask = times > 0.0
        return cls(times[mask], amts[mask])

    def pv(self, T: float, r: float) -> float:
        """PV at t=0 of divs in (0, T] using a flat discount rate r."""
        mask = (self.ex_times > 0.0) & (self.ex_times <= T + 1e-12)
        if not mask.any():
            return 0.0
        return float(np.sum(
            self.amounts[mask] * np.exp(-r * self.ex_times[mask])
        ))

    def pv_curve(self, T: float, curve: YieldCurve) -> float:
        """PV using the rate appropriate to each ex-date."""
        mask = (self.ex_times > 0.0) & (self.ex_times <= T + 1e-12)
        if not mask.any():
            return 0.0
        total = 0.0
        for t, d in zip(self.ex_times[mask], self.amounts[mask]):
            r = curve.r_continuous(float(t))
            total += d * math.exp(-r * t)
        return float(total)

    def q_equivalent(self, S: float, T: float, curve: YieldCurve) -> float:
        """Continuous yield q such that S*exp(-q T) == S - PV(divs)."""
        if T <= 0.0 or S <= 0.0:
            return 0.0
        pv = self.pv_curve(T, curve)
        if pv <= 0.0:
            return 0.0
        if pv >= S:
            return 0.0   # degenerate — shouldn't happen
        return -math.log((S - pv) / S) / T

    def slice_in(self, T: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (times, amounts) for divs in (0, T]. For numba consumption."""
        mask = (self.ex_times > 0.0) & (self.ex_times <= T + 1e-12)
        return self.ex_times[mask].copy(), self.amounts[mask].copy()


# ======================================================================
#                PARITY-IMPLIED SPOT (AVOID YAHOO ROUND-TRIP)
# ======================================================================

def implied_spot_from_parity(chain_df: pd.DataFrame, T: float,
                             curve: YieldCurve,
                             divs: Optional["DividendSchedule"] = None,
                             atm_band: float = 0.03) -> float:
    """Recover spot from ATM put-call parity on a single-expiry slice.

    For a BACKTEST (no lookahead), call this on the *shortest-dated*
    expiration in the chain and pass `divs=None`. The PV of dividends over
    very short horizons is negligible for SPY (quarterly), so the resulting
    S estimate has error well under 1 bp of spot — far less than the bid-ask
    quantisation of American ATM mids.

    If you happen to know the dividend schedule (e.g., from a forecast you
    built from past-only data), pass it as `divs` for a tighter estimate.

    Uses American mids close to ATM, where the early-exercise premium is
    small and approximately symmetric between puts and calls (so it largely
    cancels in P - C).

    Parameters
    ----------
    chain_df : DataFrame
        Must contain columns: strike, call_put ('Call'/'Put'), bid, ask.
        All rows should be for the same expiration.
    T : float
        Time to expiration in years.
    curve : YieldCurve
    divs : DividendSchedule | None
        If None, assume PV_div = 0 over (0, T]. Safe for the shortest
        expiry in a backtest.
    atm_band : float
        Fraction of strike range (±atm_band around best-guess ATM) to use
        for the regression.
    """
    calls = chain_df[chain_df["call_put"] == "Call"].copy()
    puts = chain_df[chain_df["call_put"] == "Put"].copy()
    merged = pd.merge(calls[["strike", "bid", "ask"]],
                      puts[["strike", "bid", "ask"]],
                      on="strike", suffixes=("_c", "_p"))
    merged["C"] = 0.5 * (merged["bid_c"] + merged["ask_c"])
    merged["P"] = 0.5 * (merged["bid_p"] + merged["ask_p"])
    # Initial guess: strike with smallest |C - P|
    merged["absdiff"] = (merged["C"] - merged["P"]).abs()
    atm_guess = float(merged.loc[merged["absdiff"].idxmin(), "strike"])

    lo = atm_guess * (1.0 - atm_band)
    hi = atm_guess * (1.0 + atm_band)
    near = merged[(merged["strike"] >= lo) & (merged["strike"] <= hi)]
    if len(near) < 2:
        near = merged
    r = curve.r_continuous(T)
    DF = math.exp(-r * T)
    PV_div = 0.0 if divs is None else divs.pv_curve(T, curve)

    # Regression: C - P = (S - PV_div) - DF * K
    # Slope is -DF (known); intercept is (S - PV_div). Solve for S.
    K = near["strike"].values.astype(np.float64)
    y = (near["C"] - near["P"]).values.astype(np.float64)
    # Weight near-ATM strikes more (inverse square distance from atm_guess)
    w = 1.0 / ((K - atm_guess) ** 2 + 1.0)
    # Fixing slope to -DF, the best intercept is weighted mean of (y + DF*K)
    intercept = float(np.average(y + DF * K, weights=w))
    S = intercept + PV_div
    return S


# ======================================================================
#   PARITY-IMPLIED DIVIDEND YIELD BY EXPIRATION (BACKTEST-SAFE)
# ======================================================================
# For a backtest we cannot use future dividend announcements as model input.
# Put-call parity on the *same-day* option chain gives us the market's own
# forward-dividend expectation, which is the most internally-consistent
# forecast for de-Americanising those same options.
#
#   P(K) - C(K) + S = DF(T) * K + PV_div(T)     (American ~ European at ATM)
#
# So with S known (Yahoo close) and DF(T) from the CMT curve, the intercept
# of (C - P + DF*K) across ATM strikes gives (S - PV_div(T)). Solve for
# PV_div(T), then convert to a continuous equivalent yield q(T).


def parity_implied_q_by_expiry(chain_df: pd.DataFrame,
                               spot: float,
                               curve: YieldCurve,
                               n_pairs: int = 1,
                               atm_band: float = 0.03,
                               min_pairs: int = 2) -> dict:
    """Return {T_years: q_continuous} extracted from put-call parity
    on each expiration's ATM strike(s). Uses only same-day data — safe for
    backtests.

    Parameters
    ----------
    chain_df : DataFrame
        Must contain: date, expiration, strike, call_put, bid, ask.
        May span multiple expirations.
    spot : float
        Underlying spot on `date`. For a backtest, this is the Yahoo close.
    curve : YieldCurve
    n_pairs : int
        Number of call-put pairs to use, ordered by |K/S - 1|. Default is 1
        — use only the single closest-to-ATM strike. Set to >1 to recover
        the old inverse-squared-weighted regression over the nearest
        ``n_pairs`` strikes (which also respects ``atm_band``).
    atm_band : float
        Only used when ``n_pairs > 1``. Moneyness band around ATM for the
        regression. 3% (default) includes ~3-5 SPY strikes for typical
        chains.
    min_pairs : int
        Only used when ``n_pairs > 1``. Minimum number of call-put pairs
        within the ATM band for the expiry to be included. Expiries with
        fewer pairs return q=0.

    Returns
    -------
    dict[float, float]
        Mapping from T_years (as stored in the DataFrame) to q_continuous.
        Pandas note: keys are Python floats, not pd.Timestamp.
    """
    df = chain_df.copy()
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["date"] = pd.to_datetime(df["date"])
    # Implicit: one trade date per caller. If multiple dates are in the frame,
    # we use the first (and flag).
    if df["date"].nunique() > 1:
        raise ValueError("chain_df must contain a single trade date")
    td = df["date"].iloc[0]

    out = {}
    for exp, grp in df.groupby("expiration"):
        T = (exp - td).total_seconds() / (365.0 * 86400.0)
        if T <= 0:
            continue
        # Build call-put pairs at each strike
        calls = grp[grp["call_put"].astype(str).str.lower().str.startswith("c")]
        puts = grp[grp["call_put"].astype(str).str.lower().str.startswith("p")]
        m = pd.merge(calls[["strike", "bid", "ask"]],
                     puts[["strike", "bid", "ask"]],
                     on="strike", suffixes=("_c", "_p"))
        m["C"] = 0.5 * (m["bid_c"] + m["ask_c"])
        m["P"] = 0.5 * (m["bid_p"] + m["ask_p"])
        # Filter to valid quotes. Allow bid == ask so synthetic European
        # chains from the iteration pass are accepted.
        m = m[(m["bid_c"] >= 0) & (m["ask_c"] >= m["bid_c"]) & (m["ask_c"] > 0)
              & (m["bid_p"] >= 0) & (m["ask_p"] >= m["bid_p"]) & (m["ask_p"] > 0)].copy()
        if m.empty:
            out[T] = 0.0
            continue
        m["mny"] = (m["strike"] / spot - 1.0).abs()

        r = curve.r_continuous(T)
        DF = math.exp(-r * T)

        if n_pairs <= 1:
            # Single closest-to-ATM pair: no band, no weighting.
            row = m.loc[m["mny"].idxmin()]
            s_minus_pv = float(row["C"] - row["P"] + DF * row["strike"])
        else:
            # Inverse-squared-weighted regression over the nearest n_pairs
            # strikes within ``atm_band``. Falls back to the top-n closest
            # if the band does not contain enough pairs.
            atm = m[m["mny"] <= atm_band]
            if len(atm) < min_pairs:
                atm = m.nsmallest(max(min_pairs, 3), "mny")
            atm = atm.nsmallest(n_pairs, "mny")
            w = 1.0 / (atm["mny"].values ** 2 + 1e-4)
            lhs = (atm["C"] - atm["P"] + DF * atm["strike"]).values
            s_minus_pv = float(np.average(lhs, weights=w))

        pv_div = spot - s_minus_pv
        # Clip to sensible range: no negative divs, no more than 5% of spot
        pv_div = max(0.0, min(pv_div, 0.05 * spot))
        if pv_div <= 0.0:
            out[T] = 0.0
        else:
            out[T] = -math.log((spot - pv_div) / spot) / T
    return out


def _q_lookup_from_dict(q_dict: dict):
    """Return a callable T -> q that piecewise-linearly interpolates on the
    provided map; flat-extrapolates at both ends."""
    if not q_dict:
        return lambda T: 0.0
    ts = np.array(sorted(q_dict.keys()), dtype=np.float64)
    qs = np.array([q_dict[t] for t in ts], dtype=np.float64)

    def fn(T: float) -> float:
        if T <= ts[0]:
            return float(qs[0])
        if T >= ts[-1]:
            return float(qs[-1])
        return float(np.interp(T, ts, qs))
    return fn


def anchor_q_curve_long_end(q_dict: dict,
                            long_end_q: float,
                            long_end_T: float = 1.0) -> dict:
    """Append a synthetic long-end point to a parity-implied q curve.

    Parity can only speak for maturities actually quoted in the chain. For
    SPY that is typically only a few weeks to a few months, which sees at
    most one or two quarterly ex-div events — flat-extrapolating that out to
    a multi-year autocall underestimates the true dividend drag because the
    other three quarterly distributions are invisible to parity.

    The fix used here is a single anchor point at ``long_end_T`` with value
    ``long_end_q`` (e.g., SPY TTM dividend yield). Downstream, linear
    interpolation of cumulative Q(T) = q(T)*T between the last parity expiry
    and this anchor gives a smooth blend; flat extrapolation beyond
    ``long_end_T`` holds q constant at the anchor.

    If ``long_end_T`` falls inside the existing grid, the function is a
    no-op: the quoted range already covers the anchor horizon.
    """
    if long_end_q is None or long_end_T is None:
        return dict(q_dict)
    if not q_dict:
        return {float(long_end_T): float(long_end_q)}
    max_T = max(q_dict.keys())
    if long_end_T <= max_T:
        return dict(q_dict)
    out = dict(q_dict)
    out[float(long_end_T)] = float(long_end_q)
    return out


def regularize_q_curve(q_dict: dict) -> dict:
    """Enforce monotone non-decreasing cumulative dividend yield Q(T)=q(T)*T.

    Raw parity-implied q(T) across expiries is noisy: bid/ask quote width,
    residual early-exercise premium, and iteration artifacts can drag q(T)
    below its neighbour. When those point estimates are differenced to build
    forward dividends over (t_{i-1}, t_i), the dips produce negative forwards
    that are economically implausible — real cumulative expected dividend
    yield is non-decreasing in T.

    The fix is the simplest projection onto the non-decreasing cone: a
    running forward maximum on Q_i = q_i * T_i. The first maturity is left
    untouched; each subsequent Q is pulled up to at least the preceding Q,
    then q is recovered as Q/T.
    """
    if not q_dict:
        return {}
    ts = np.array(sorted(q_dict.keys()), dtype=np.float64)
    qs = np.array([q_dict[t] for t in ts], dtype=np.float64)
    Q = qs * ts
    Q = np.maximum.accumulate(Q)
    qs_clean = np.where(ts > 0.0, Q / np.where(ts > 0.0, ts, 1.0), qs)
    return {float(t): float(q) for t, q in zip(ts, qs_clean)}


def _q_lookup_from_dict_Q(q_dict: dict):
    """Return a callable T -> q that piecewise-linearly interpolates the
    cumulative dividend yield Q(T) = q(T)*T, then recovers q(T) = Q(T)/T.

    Compared with ``_q_lookup_from_dict`` (which interpolates q linearly),
    this preserves monotonicity of Q between grid points: if the input q_dict
    already has non-decreasing Q on its grid (e.g. after
    ``regularize_q_curve``), then Q(T) is non-decreasing for every T, so
    forward dividend yields differenced off this function are guaranteed
    non-negative.

    Extrapolation: flat q at both ends (same convention as
    ``_q_lookup_from_dict``), i.e. Q(T) linear for T<ts[0] and linear for
    T>ts[-1] with the short- and long-end slopes set to q_first and q_last.
    """
    if not q_dict:
        return lambda T: 0.0
    ts = np.array(sorted(q_dict.keys()), dtype=np.float64)
    qs = np.array([q_dict[t] for t in ts], dtype=np.float64)
    Q = qs * ts

    def fn(T: float) -> float:
        if T <= 0.0:
            return 0.0
        if T <= ts[0]:
            return float(qs[0])
        if T >= ts[-1]:
            return float(qs[-1])
        Q_T = float(np.interp(T, ts, Q))
        return Q_T / T
    return fn


def parity_implied_q_iterated(chain_df: pd.DataFrame,
                              spot: float,
                              curve: YieldCurve,
                              n_iter: int = 2,
                              n_pairs: int = 1,
                              atm_band: float = 0.03,
                              min_pairs: int = 2,
                              method: str = "baw") -> dict:
    """Iteratively refined parity-implied yield-by-expiry.

    Standard parity on American mids under-states (S - PV_div) because
    American put mids carry an early-exercise premium that European C-P
    parity doesn't expect. This routine iterates:

        1. Start with q_dict = parity on raw American mids.
        2. De-Americanize the chain with that q_dict to get European mids.
        3. Re-extract q_dict from parity on the European mids.
        4. Repeat; typically 2 passes suffice.

    Returns the converged q_dict. Strictly non-lookahead (all inputs
    come from same-day data).

    Parameters
    ----------
    chain_df : DataFrame
        Single trade date; same schema as `deamericanize_chain`.
    spot, curve : as usual.
    n_iter : int
        Number of refinement passes. 2 is usually enough; residual shift
        after pass 2 is typically <10 bp.
    n_pairs : int
        Forwarded to ``parity_implied_q_by_expiry``. Default 1 → single
        closest-to-ATM pair per expiry.
    method : {'baw', 'crr'}
        Pricer for the intermediate de-Am step. BAW is ~20x faster.
    """
    # Pass 0: raw American parity
    q_dict = parity_implied_q_by_expiry(chain_df, spot, curve,
                                        n_pairs=n_pairs,
                                        atm_band=atm_band,
                                        min_pairs=min_pairs)

    for _ in range(n_iter):
        # Build European mids with current q estimate
        deam = deamericanize_chain(
            chain_df, spot=spot, curve=curve,
            q_per_expiry=q_dict, method=method,
        )
        # Build synthetic European chain (strike, bid=ask=euro mid)
        # and re-extract q from parity.
        euro_chain = deam[["date", "expiration", "strike", "call_put",
                           "european_mid"]].copy().dropna(subset=["european_mid"])
        euro_chain["bid"] = euro_chain["european_mid"]
        euro_chain["ask"] = euro_chain["european_mid"]
        q_dict = parity_implied_q_by_expiry(
            euro_chain[["date", "expiration", "strike",
                        "call_put", "bid", "ask"]],
            spot, curve, n_pairs=n_pairs,
            atm_band=atm_band, min_pairs=min_pairs,
        )
    return q_dict


# ======================================================================
#                  TOP-LEVEL CHAIN DEAMERICANIZATION
# ======================================================================

def deamericanize_chain(chain_df: pd.DataFrame,
                        spot: float,
                        curve: YieldCurve,
                        divs: Optional[DividendSchedule] = None,
                        q_per_expiry: Optional[dict] = None,
                        valuation_date: Optional[pd.Timestamp] = None,
                        method: str = "baw",
                        n_crr_steps: int = 400) -> pd.DataFrame:
    """De-Americanize a full option chain.

    There are two ways to supply the dividend input (BACKTEST vs KNOWN):

    * **q_per_expiry** (dict {T_years: q_continuous}) — recommended for
      backtests. Pass the output of `parity_implied_q_by_expiry(...)`. This
      is market-implied and uses only same-day option quotes, so it is
      strictly non-lookahead. Internally, CRR then uses a continuous-yield
      tree (no discrete div schedule needed).

    * **divs** (DividendSchedule) — use when you have an explicit forecast
      of future ex-dates and amounts (e.g., from a historical-only projection
      you built outside this module). CRR treats each cash dividend as a
      discrete drop; BAW uses the equivalent continuous yield.

    Exactly one of `divs` / `q_per_expiry` must be supplied. If both are
    None, the function auto-computes `q_per_expiry` from the chain itself
    using `parity_implied_q_by_expiry` — this is the fully-automatic
    backtest-safe default.

    Parameters
    ----------
    chain_df : DataFrame
        Columns required: date, expiration, strike, call_put, bid, ask.
        All rows must share the same `date` (= valuation date).
    spot : float
        Underlying spot at valuation. For backtests, use same-day close
        (e.g., Yahoo) — no lookahead.
    curve : YieldCurve
    divs : DividendSchedule | None
    q_per_expiry : dict | None
    valuation_date : pd.Timestamp | None
        Defaults to the unique `date` found in the chain.
    method : {'baw', 'crr'}
    n_crr_steps : int
        Tree steps for CRR (ignored for BAW). 400 is bid/ask-tight for
        weekly / monthly expiries; 800 for multi-year.

    Returns
    -------
    DataFrame
        Copy of input with extra columns:
            T_years, r, q_eff, mid, american_iv, european_mid
    """
    if divs is not None and q_per_expiry is not None:
        raise ValueError("Pass exactly one of `divs` or `q_per_expiry`.")

    out = chain_df.copy()
    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    out["expiration"] = pd.to_datetime(out["expiration"])

    if valuation_date is None:
        uniq = pd.to_datetime(out["date"]).unique()
        if len(uniq) != 1:
            raise ValueError(
                "chain_df spans multiple trade dates; pass valuation_date"
            )
        valuation_date = pd.Timestamp(uniq[0])
    else:
        valuation_date = pd.Timestamp(valuation_date)

    out["T_years"] = (
        out["expiration"] - valuation_date
    ).dt.total_seconds() / (365.0 * 86400.0)

    # Drop degenerate rows early
    mask = (out["T_years"] > 0) & (out["bid"] > 0) & (out["ask"] > out["bid"])
    out = out.loc[mask].reset_index(drop=True)

    # --- resolve q_eff per row ---
    if divs is None and q_per_expiry is None:
        # default: parity-implied, fully non-lookahead
        # Build an input for parity_implied_q_by_expiry that has exactly
        # one trade date (required by that function).
        chain_for_q = chain_df.copy()
        chain_for_q["date"] = valuation_date
        q_per_expiry = parity_implied_q_by_expiry(
            chain_for_q, spot=spot, curve=curve
        )

    out["r"] = out["T_years"].apply(curve.r_continuous)
    if divs is not None:
        out["q_eff"] = [divs.q_equivalent(spot, T, curve)
                        for T in out["T_years"]]
    else:
        q_fn = _q_lookup_from_dict(q_per_expiry)
        out["q_eff"] = [q_fn(float(T)) for T in out["T_years"]]

    is_call = (out["call_put"].astype(str).str.lower().str[0] == "c").values

    mids = out["mid"].values.astype(np.float64)
    Ss = np.full(len(out), spot, dtype=np.float64)
    Ks = out["strike"].values.astype(np.float64)
    Ts = out["T_years"].values.astype(np.float64)
    rs = out["r"].values.astype(np.float64)
    qs = out["q_eff"].values.astype(np.float64)
    is_calls = is_call.astype(np.bool_)

    method = method.lower()
    if method == "baw":
        ivs, euros = _process_chain_baw(mids, Ss, Ks, Ts, rs, qs, is_calls)
    elif method == "crr":
        # CRR path: if we have an explicit DividendSchedule, use discrete
        # cash dividends. If we have q_per_expiry (continuous), use zero
        # discrete divs and rely on qs[] for the drift in the tree — but
        # that requires crr_price to support continuous q. We add that below.
        if divs is not None:
            T_max = float(out["T_years"].max())
            div_t, div_a = divs.slice_in(T_max)
            ivs, euros = _process_chain_crr(
                mids, Ss, Ks, Ts, rs, qs, is_calls,
                div_t.astype(np.float64), div_a.astype(np.float64),
                n_crr_steps,
            )
        else:
            # Continuous-yield CRR (backtest case: q extracted from parity)
            ivs, euros = _process_chain_crr_cts(
                mids, Ss, Ks, Ts, rs, qs, is_calls, n_crr_steps,
            )
    else:
        raise ValueError(f"method must be 'baw' or 'crr', got {method!r}")

    out["american_iv"] = ivs
    out["european_mid"] = euros
    return out
