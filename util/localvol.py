"""
localvol.py
===========
Build a Dupire local-volatility surface sigma_loc(K, T) from a
de-Americanized European option chain, and wrap it in a grid object the
C++ pricer can consume.

Pipeline
--------
    de-Am chain (strike x expiry)
            -->  per-expiry total-variance smile fits  [IVSmile]
            -->  dense 2-D total-variance grid w(log_m, T)   [ImpliedVolSurface]
            -->  Gatheral-form Dupire                          ->  sigma_loc
            -->  LocalVolSurface (log_m_grid, t_grid, sigma_loc[n_m, n_t])

Design choices
--------------
*   Smile fit per expiry: natural cubic spline on w(k) with flat wings.
    Natural bc keeps the fit tame; SPY data is dense and already de-Am'd, so
    smoothing is not needed. We guard butterfly arb only by flooring negative
    total variance post-fit and by the final sigma_loc floor/cap.

*   Across expiries: linear interpolation in total variance (standard).
    A monotone-in-T projection enforces calendar arb: w(log_m, T_{j+1}) >=
    w(log_m, T_j) at every log_m, via cumulative max. This is the cheapest
    post-hoc guard and matches what traders do in practice.

*   Dupire in Gatheral form (total variance, log-forward-moneyness y):

        sigma_loc^2(y, T) = w_T / [1 - (y/w) w_y
                                    + (1/4)(-1/4 - 1/w + y^2/w^2) w_y^2
                                    + (1/2) w_yy]

    The storage axis is log_m = log(K/S0) for kernel convenience. At each
    (log_m, T) grid node we compute y = log_m - (cumR - cumQ)(T); derivatives
    ``w_y`` equal finite differences on log_m (at fixed T); ``w_T`` is
    corrected via ``w_T|_y = w_T|_log_m + (r-q) * w_y``.

*   Output: sigma_loc floored at 1% vol, capped at 200% vol.
    Stored as float64 [n_m, n_t], plus log_m_grid [n_m], t_grid [n_t].

Important caveat
----------------
Dupire extracts a surface from the observed chain. For grid nodes beyond the
longest observed expiry, only the flat-forward-variance extrapolation feeds
the differentiation, so the resulting ``sigma_loc(T)`` at long T is
essentially the short-end forward vol — not a real long-dated calibration.
To price a 5Y autocall against a real local-vol surface, the input chain
must actually contain LEAPS / long-dated expiries. Otherwise, the sub-stepped
pricer in ``prod/`` will price a smoothed extrapolation, not a calibration.

Public API
----------
    IVSmile(ks, ws, T)               per-expiry smile object
    ImpliedVolSurface.from_chain(...)  fit all slices from a de-Am DataFrame
    LocalVolSurface.from_iv_surface(...)  Dupire conversion onto a grid
    build_local_vol_grid_from_chain(...)  one-shot convenience wrapper
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

try:
    from .deamerican import YieldCurve, _q_lookup_from_dict
except ImportError:  # allow `python util/test_localvol.py`
    from deamerican import YieldCurve, _q_lookup_from_dict


# ======================================================================
#                        CONSTANTS / HELPERS
# ======================================================================

_SIGMA_FLOOR = 0.01        # 1% vol minimum
_SIGMA_CAP   = 2.00        # 200% vol maximum
_W_FLOOR     = 1e-8        # total variance minimum (numerical)
_K_PAD       = 1e-6        # avoid duplicate-x in spline inputs


def _cum_r_minus_q(T: float, curve: YieldCurve, q_fn) -> float:
    """Integral from 0 to T of (r(s) - q(s)) ds, piecewise-constant in our
    convention. Since r(T) and q(T) here are *term* continuous rates at
    maturity T, the integral evaluates trivially to (r(T) - q(T)) * T.
    (This is consistent with how build_forward_rates / build_forward_divs
    difference term quantities.)"""
    r = curve.r_continuous(T)
    q = float(q_fn(T))
    return (r - q) * T


# ======================================================================
#                        PER-EXPIRY SMILE
# ======================================================================

@dataclass
class IVSmile:
    """Total-variance smile for one expiry.

    Attributes
    ----------
    ks : np.ndarray     log-forward-moneyness k = log(K / F(T)), strictly increasing.
    ws : np.ndarray     total variance w = sigma_imp^2 * T at each k.
    T  : float          maturity in years.
    """
    ks: np.ndarray
    ws: np.ndarray
    T: float

    def __post_init__(self):
        ks = np.asarray(self.ks, dtype=np.float64)
        ws = np.asarray(self.ws, dtype=np.float64)
        order = np.argsort(ks)
        ks = ks[order]
        ws = np.clip(ws[order], _W_FLOOR, None)
        # De-duplicate (rare, but can happen with repeated strikes)
        keep = np.concatenate([[True], np.diff(ks) > _K_PAD])
        ks = ks[keep]
        ws = ws[keep]
        if len(ks) < 2:
            raise ValueError(f"IVSmile at T={self.T}: need >=2 points, got {len(ks)}")
        self.ks = ks
        self.ws = ws
        if len(ks) >= 4:
            self._spline = CubicSpline(ks, ws, bc_type="natural", extrapolate=False)
        else:
            # Too few points for cubic: fall back to linear interpolation.
            self._spline = None
        self._k_lo = float(ks[0])
        self._k_hi = float(ks[-1])
        self._w_lo = float(ws[0])
        self._w_hi = float(ws[-1])

    def w(self, k: float) -> float:
        """Total variance at log-forward-moneyness k. Flat extrapolation."""
        if k <= self._k_lo:
            return self._w_lo
        if k >= self._k_hi:
            return self._w_hi
        if self._spline is not None:
            v = float(self._spline(k))
        else:
            v = float(np.interp(k, self.ks, self.ws))
        return v if v > _W_FLOOR else _W_FLOOR

    def w_array(self, ks: np.ndarray) -> np.ndarray:
        """Vectorised w(k) for an array of k values."""
        ks = np.asarray(ks, dtype=np.float64)
        out = np.empty_like(ks)
        lo = ks <= self._k_lo
        hi = ks >= self._k_hi
        mid = ~(lo | hi)
        out[lo] = self._w_lo
        out[hi] = self._w_hi
        if mid.any():
            if self._spline is not None:
                out[mid] = self._spline(ks[mid])
            else:
                out[mid] = np.interp(ks[mid], self.ks, self.ws)
        return np.clip(out, _W_FLOOR, None)


# ======================================================================
#                 IMPLIED-VOL SURFACE (STACK OF SLICES)
# ======================================================================

@dataclass
class ImpliedVolSurface:
    """Collection of per-expiry IVSmile objects + forward shift lookup.

    The shift F(T)/S0 = exp((r-q) T) is needed to convert between our
    storage axis log_m = log(K/S0) and Gatheral's y = log(K/F(T)).
    """
    slices: list                      # list[IVSmile], sorted by T
    spot: float
    forward_shift: dict               # {T: log(F(T)/S0) = (r-q)T}

    @classmethod
    def from_chain(cls,
                   deam_df: pd.DataFrame,
                   spot: float,
                   curve: YieldCurve,
                   q_provider,
                   iv_col: str = "american_iv",
                   otm_only: bool = True,
                   min_points_per_expiry: int = 4) -> "ImpliedVolSurface":
        """Fit one IVSmile per expiration from a de-Americanized chain.

        Parameters
        ----------
        deam_df : DataFrame
            Output of ``deamericanize_chain`` with columns
            ``T_years, strike, call_put, <iv_col>``.
        spot : float
        curve : YieldCurve
        q_provider : callable or dict
            Continuous yield as a function of T. Same shape as in
            ``autocall_prep.build_autocall_inputs``.
        iv_col : str
            Column with the European IV (default ``american_iv`` — see the
            note in the deamerican module: that column is the BS IV matching
            the de-Am European mid).
        otm_only : bool
            Use OTM puts for K <= F and OTM calls for K > F (tighter mids,
            less early-exercise residual). Default ``True``.
        min_points_per_expiry : int
            Skip expiries with fewer usable points.
        """
        if isinstance(q_provider, dict):
            q_fn = _q_lookup_from_dict(q_provider)
        else:
            q_fn = q_provider

        df = deam_df.dropna(subset=[iv_col]).copy()
        if df.empty:
            raise ValueError("chain has no rows with finite IV")
        df = df[df[iv_col] > 0].copy()

        slices: list[IVSmile] = []
        forward_shift: dict[float, float] = {}
        for T, g in df.groupby("T_years", sort=True):
            T = float(T)
            if T <= 0:
                continue
            r = curve.r_continuous(T)
            q = float(q_fn(T))
            log_FoverS0 = (r - q) * T           # log(F/S0)
            F = spot * math.exp(log_FoverS0)
            strikes = g["strike"].to_numpy(dtype=np.float64)
            ivs = g[iv_col].to_numpy(dtype=np.float64)
            cps = g["call_put"].astype(str).str.lower().str[0].to_numpy()
            if otm_only:
                # OTM put: K < F and cp=='p'; OTM call: K > F and cp=='c'; both at ATM.
                mask = (
                    ((strikes <= F) & (cps == "p")) |
                    ((strikes >= F) & (cps == "c")) |
                    (np.isclose(strikes, F, rtol=1e-3))
                )
                strikes_o = strikes[mask]
                ivs_o = ivs[mask]
                if len(strikes_o) < min_points_per_expiry:
                    # Fall back to whole slice (average duplicate strikes).
                    strikes_o = strikes
                    ivs_o = ivs
            else:
                strikes_o = strikes
                ivs_o = ivs
            # Collapse any remaining put/call duplicates by averaging IV.
            order = np.argsort(strikes_o)
            strikes_o = strikes_o[order]
            ivs_o = ivs_o[order]
            if len(strikes_o) > 1:
                unique_K, inv = np.unique(strikes_o, return_inverse=True)
                avg_iv = np.zeros_like(unique_K)
                cnt = np.zeros_like(unique_K)
                for i, iv in zip(inv, ivs_o):
                    avg_iv[i] += iv
                    cnt[i] += 1
                iv_unique = avg_iv / np.maximum(cnt, 1)
                strikes_o = unique_K
                ivs_o = iv_unique
            if len(strikes_o) < min_points_per_expiry:
                warnings.warn(
                    f"Skipping T={T:.4f}: only {len(strikes_o)} usable points "
                    f"(need >= {min_points_per_expiry}).",
                    RuntimeWarning,
                )
                continue
            ks = np.log(strikes_o / F)
            ws = (ivs_o ** 2) * T
            slices.append(IVSmile(ks, ws, T))
            forward_shift[T] = log_FoverS0

        if len(slices) < 2:
            raise ValueError(
                f"need >= 2 expirations for a local-vol surface, got {len(slices)}"
            )
        slices.sort(key=lambda s: s.T)
        return cls(slices=slices, spot=float(spot), forward_shift=forward_shift)

    # ----- evaluation helpers ---------------------------------------

    def w_at(self, log_m: float, T: float) -> float:
        """Total variance at (log_m, T) via per-expiry evaluation + linear-in-T
        interpolation. log_m = log(K/S0)."""
        slices = self.slices
        Ts = np.array([s.T for s in slices])
        if T <= Ts[0]:
            s = slices[0]
            k = log_m - (s.T <= 0 and 0.0 or self.forward_shift[s.T])
            # Scale linearly in T for very short horizons (total variance -> 0 at T=0)
            if T <= 0.0:
                return 0.0
            return s.w(k) * (T / s.T)
        if T >= Ts[-1]:
            # Flat-extrapolate forward variance from the last interval.
            s = slices[-1]
            k = log_m - self.forward_shift[s.T]
            if len(slices) >= 2:
                s_prev = slices[-2]
                k_prev = log_m - self.forward_shift[s_prev.T]
                fwd_var = max(
                    (s.w(k) - s_prev.w(k_prev)) / (s.T - s_prev.T),
                    _W_FLOOR,
                )
                return s.w(k) + fwd_var * (T - s.T)
            return s.w(k)
        # Straddle
        j = int(np.searchsorted(Ts, T, side="right") - 1)
        s_lo = slices[j]
        s_hi = slices[j + 1]
        k_lo = log_m - self.forward_shift[s_lo.T]
        k_hi = log_m - self.forward_shift[s_hi.T]
        w_lo = s_lo.w(k_lo)
        w_hi = s_hi.w(k_hi)
        alpha = (T - s_lo.T) / (s_hi.T - s_lo.T)
        return (1.0 - alpha) * w_lo + alpha * w_hi

    def w_grid(self, log_m_grid: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        """Evaluate w on a dense 2-D grid. Shape [n_m, n_t]."""
        n_m = len(log_m_grid)
        n_t = len(t_grid)
        w = np.empty((n_m, n_t), dtype=np.float64)
        for j, T in enumerate(t_grid):
            for i, lm in enumerate(log_m_grid):
                w[i, j] = self.w_at(float(lm), float(T))
        return w

    def forward_log(self, T: float, curve: YieldCurve, q_fn) -> float:
        """log(F(T)/S0) at arbitrary T, consistent with the rate/div provider
        (interpolated, not cached). Used by the Dupire builder."""
        return _cum_r_minus_q(T, curve, q_fn)


# ======================================================================
#                     LOCAL-VOL SURFACE (DUPIRE)
# ======================================================================

@dataclass
class LocalVolSurface:
    """Grid of Dupire local vol on (log_m, t).

    Attributes
    ----------
    log_m_grid : np.ndarray [n_m]        log(K/S0) axis
    t_grid     : np.ndarray [n_t]        time axis (years, must include t=0)
    sigma_loc  : np.ndarray [n_m, n_t]   local vol at each grid node
    """
    log_m_grid: np.ndarray
    t_grid: np.ndarray
    sigma_loc: np.ndarray

    @classmethod
    def from_iv_surface(cls,
                        iv_surface: ImpliedVolSurface,
                        curve: YieldCurve,
                        q_provider,
                        log_m_bounds: tuple = (-0.8, 0.8),
                        n_m: int = 161,
                        t_max: float = None,
                        n_t: int = 101,
                        t_min: float = 1.0 / 365.0,
                        enforce_calendar: bool = True) -> "LocalVolSurface":
        """Apply the Gatheral-form Dupire transform to an implied-vol surface.

        Parameters
        ----------
        iv_surface : ImpliedVolSurface
        curve, q_provider : as in ImpliedVolSurface.from_chain
        log_m_bounds : (lo, hi)
            Storage range for log(K/S0). Default +/-80% log-moneyness is wide
            enough for 5Y SPY paths up to ~3 sigma.
        n_m : int       grid size in log-moneyness (odd preferred so ATM is a node).
        t_max : float   maximum maturity to cover; defaults to the last slice + 10%.
        n_t : int       grid size in time.
        t_min : float   smallest positive time node (avoid T=0 singularity).
            t_grid[0] = 0 is inserted; t_grid[1] = t_min.
        enforce_calendar : bool
            Project w(log_m, T) to be non-decreasing in T at each log_m
            (cumulative max). Guards Dupire denominator stability.
        """
        if isinstance(q_provider, dict):
            q_fn = _q_lookup_from_dict(q_provider)
        else:
            q_fn = q_provider

        if t_max is None:
            t_max = iv_surface.slices[-1].T * 1.10

        log_m_grid = np.linspace(log_m_bounds[0], log_m_bounds[1], n_m)
        # Time grid: t=0 first (no variance), then log-spaced from t_min to t_max
        t_body = np.geomspace(max(t_min, 1e-4), t_max, n_t - 1)
        t_grid = np.concatenate([[0.0], t_body])

        # 1. Dense total-variance grid w[n_m, n_t]
        w = iv_surface.w_grid(log_m_grid, t_grid)
        w[:, 0] = 0.0   # variance at T=0 is exactly 0 by definition

        # 2. Calendar-arb guard: cumulative max along T at each log_m.
        if enforce_calendar:
            for i in range(n_m):
                w[i, :] = np.maximum.accumulate(w[i, :])

        # 3. Finite-difference derivatives.
        #    w_log_m (at fixed T)        — central diff on log_m
        #    w_log_m_log_m (at fixed T)  — 2nd diff on log_m
        #    w_T_fixed_log_m            — central diff on T at fixed log_m
        dlm = log_m_grid[1] - log_m_grid[0]
        w_lm = np.gradient(w, dlm, axis=0, edge_order=2)
        w_lm_lm = np.gradient(w_lm, dlm, axis=0, edge_order=2)

        # dT varies on log-spaced grid; use non-uniform gradient.
        w_T_lm = np.gradient(w, t_grid, axis=1, edge_order=2)

        # 4. Convert w_T at fixed log_m -> w_T at fixed y.
        #    y = log_m - (r-q)T  =>  dy/dT|_log_m = -(r-q)
        #    w_T|_y = w_T|_log_m - w_y * dy/dT|_log_m = w_T|_log_m + (r-q)*w_lm
        #    (w_y == w_lm at fixed T.)
        sigma_loc = np.empty_like(w)
        for j, T in enumerate(t_grid):
            if T <= 0.0:
                # At T=0: set sigma_loc to short-dated ATM vol (continuity).
                # We'll fill this row from j=1 after the loop.
                continue
            r_T = curve.r_continuous(T)
            q_T = float(q_fn(T))
            rq = r_T - q_T
            log_F_over_S0 = rq * T
            for i, lm in enumerate(log_m_grid):
                y = lm - log_F_over_S0
                w_ij = max(w[i, j], _W_FLOOR)
                wy = w_lm[i, j]
                wyy = w_lm_lm[i, j]
                wT_y = w_T_lm[i, j] + rq * wy
                # Gatheral denominator
                denom = (1.0
                         - (y / w_ij) * wy
                         + 0.25 * (-0.25 - 1.0 / w_ij + (y * y) / (w_ij * w_ij)) * wy * wy
                         + 0.5 * wyy)
                # Guard denom: if very small or negative (butterfly arb
                # violation), fall back to sqrt(w_T|_y) / 1.
                if denom < 0.1:
                    denom = 0.1
                sigma2 = wT_y / denom
                if (not math.isfinite(sigma2)) or sigma2 <= 0.0:
                    # fall back to ATM-equivalent via w_T|_y
                    sigma2 = max(wT_y, _SIGMA_FLOOR ** 2)
                sigma2 = min(max(sigma2, _SIGMA_FLOOR ** 2), _SIGMA_CAP ** 2)
                sigma_loc[i, j] = math.sqrt(sigma2)
        # Fill T=0 row with first positive-T column (short-dated limit).
        sigma_loc[:, 0] = sigma_loc[:, 1]

        return cls(log_m_grid=log_m_grid, t_grid=t_grid, sigma_loc=sigma_loc)

    # ----- lookup & diagnostics ------------------------------------------

    def sigma(self, log_m: float, t: float) -> float:
        """Bilinear lookup with flat extrapolation (diagnostic)."""
        lm = self.log_m_grid
        tg = self.t_grid
        i = int(np.clip(np.searchsorted(lm, log_m) - 1, 0, len(lm) - 2))
        j = int(np.clip(np.searchsorted(tg, t) - 1, 0, len(tg) - 2))
        # clip log_m/t to grid
        log_m_c = min(max(log_m, lm[0]), lm[-1])
        t_c = min(max(t, tg[0]), tg[-1])
        alpha = (log_m_c - lm[i]) / (lm[i + 1] - lm[i])
        beta = (t_c - tg[j]) / max(tg[j + 1] - tg[j], 1e-12)
        a = self.sigma_loc[i, j]
        b = self.sigma_loc[i + 1, j]
        c = self.sigma_loc[i, j + 1]
        d = self.sigma_loc[i + 1, j + 1]
        return ((1 - alpha) * (1 - beta) * a
                + alpha * (1 - beta) * b
                + (1 - alpha) * beta * c
                + alpha * beta * d)

    def summary(self) -> pd.DataFrame:
        """Per-column (T slice) quantile summary of sigma_loc, for QC."""
        rows = []
        for j, T in enumerate(self.t_grid):
            col = self.sigma_loc[:, j]
            rows.append({
                "T": T,
                "min": col.min(),
                "p10": np.quantile(col, 0.10),
                "atm": float(np.interp(0.0, self.log_m_grid, col)),
                "p90": np.quantile(col, 0.90),
                "max": col.max(),
            })
        return pd.DataFrame(rows)


# ======================================================================
#                  CONVENIENCE ONE-SHOT BUILDER
# ======================================================================

def build_local_vol_grid_from_chain(deam_df: pd.DataFrame,
                                    spot: float,
                                    curve: YieldCurve,
                                    q_provider,
                                    log_m_bounds: tuple = (-0.8, 0.8),
                                    n_m: int = 161,
                                    t_max: float = None,
                                    n_t: int = 101,
                                    iv_col: str = "american_iv") -> LocalVolSurface:
    """End-to-end: fit smiles from a de-Am chain, then Dupire to a grid."""
    iv_surface = ImpliedVolSurface.from_chain(
        deam_df, spot=spot, curve=curve, q_provider=q_provider, iv_col=iv_col,
    )
    return LocalVolSurface.from_iv_surface(
        iv_surface, curve=curve, q_provider=q_provider,
        log_m_bounds=log_m_bounds, n_m=n_m, t_max=t_max, n_t=n_t,
    )
