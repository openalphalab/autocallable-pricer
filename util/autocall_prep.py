"""
autocall_prep.py
================
Turn a de-Americanized SPY option chain into the exact input arrays that
`autocall_pricer_prod.make_single_spec(...)` requires.

The autocall pricer wants three per-interval vectors on the observation grid
`obs_times = [t_1, t_2, ..., t_N]`:

    fwd_vols[i]   — forward vol over (t_{i-1}, t_i)
    fwd_rates[i]  — forward rate over (t_{i-1}, t_i)
    fwd_divs[i]   — forward dividend yield over (t_{i-1}, t_i)

We build each one from the no-arbitrage additive relation on total variance /
discount / PV-dividend, taken at the observation endpoints:

    sigma_fwd^2 (t_{i-1}, t_i) * Delta_i = sigma_term^2(t_i) * t_i
                                         - sigma_term^2(t_{i-1}) * t_{i-1}
    r_fwd       (t_{i-1}, t_i) * Delta_i = r_term(t_i) * t_i
                                         - r_term(t_{i-1}) * t_{i-1}
    q_fwd       (t_{i-1}, t_i) * Delta_i = q_term(t_i) * t_i
                                         - q_term(t_{i-1}) * t_{i-1}

where sigma_term(T) is the ATM implied vol at maturity T (extracted from the
de-Americanized chain), r_term(T) comes from the YieldCurve, and q_term(T) is
the continuous-equivalent yield implied by discrete dividends in (0, T].

If sigma_term^2(t) * t decreases over an interval (inverted term structure in
variance), the naive formula yields a negative forward variance. We floor it
at a small positive value (1e-6) and emit a warning — this indicates the vol
surface needs regularisation before being used.

Public API
----------
    ATMVolTermStructure.from_chain(df, spot)
        Extract a (T_i, sigma_atm_i) curve from a de-Americanized chain.
    build_autocall_inputs(obs_times, vol_ts, curve, divs, spot)
        Produce the fwd_vols / fwd_rates / fwd_divs arrays.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from .deamerican import (
        YieldCurve,
        _q_lookup_from_dict,
        _q_lookup_from_dict_Q,
        anchor_q_curve_long_end,
        regularize_q_curve,
    )
    from .localvol import LocalVolSurface, build_local_vol_grid_from_chain
except ImportError:  # allow `python util/xxx.py`
    from deamerican import (
        YieldCurve,
        _q_lookup_from_dict,
        _q_lookup_from_dict_Q,
        anchor_q_curve_long_end,
        regularize_q_curve,
    )
    from localvol import LocalVolSurface, build_local_vol_grid_from_chain


# ======================================================================
#          ATM IMPLIED-VOL TERM STRUCTURE FROM A OPTION CHAIN
# ======================================================================

@dataclass
class ATMVolTermStructure:
    """ATM implied volatility as a function of maturity T.

    Interpolation is done in **total variance** space (sigma^2 * T) and then
    converted back — this preserves no-arbitrage in calendar spreads.

    Attributes
    ----------
    expirations : np.ndarray[float64]    maturities in years, strictly increasing
    atm_vols    : np.ndarray[float64]    ATM implied vol at each T
    """
    expirations: np.ndarray
    atm_vols: np.ndarray

    def __post_init__(self):
        self.expirations = np.asarray(self.expirations, dtype=np.float64)
        self.atm_vols = np.asarray(self.atm_vols, dtype=np.float64)
        order = np.argsort(self.expirations)
        self.expirations = self.expirations[order]
        self.atm_vols = self.atm_vols[order]
        # Cache total variance
        self._total_var = self.atm_vols ** 2 * self.expirations

    @classmethod
    def from_chain(cls, chain_df: pd.DataFrame, spot: float,
                   band: float = 0.02,
                   iv_col: str = "american_iv") -> "ATMVolTermStructure":
        """Extract ATM vol per expiration by averaging near-ATM call/put IVs.

        Parameters
        ----------
        chain_df : DataFrame
            Output of `deamericanize_chain` — must have columns
            `T_years`, `strike`, `call_put`, and `iv_col`.
        spot : float
            Underlying spot for ATM band.
        band : float
            |K/S - 1| threshold for ATM bucket. 2% is tight; use 5% for thin
            markets.
        iv_col : str
            Column holding the IV to use. 'american_iv' after de-Am is
            standard; pass a different column if you've overridden.
        """
        df = chain_df.copy()
        df = df.dropna(subset=[iv_col])
        df["moneyness"] = df["strike"] / spot
        atm_mask = (df["moneyness"] - 1.0).abs() <= band
        atm = df.loc[atm_mask].copy()
        if atm.empty:
            raise ValueError(f"No ATM strikes within {band:.0%} of spot={spot}")

        # Average call and put IVs at each expiration, weighted inverse of
        # (distance from ATM + 1bp). Gives robust central estimate.
        atm["w"] = 1.0 / ((atm["moneyness"] - 1.0).abs() + 1e-4)
        grp = atm.groupby("T_years", sort=True).apply(
            lambda g: np.average(g[iv_col].to_numpy(), weights=g["w"].to_numpy()),
            include_groups=False,
        ).reset_index()
        grp.columns = ["T", "sigma_atm"]
        grp = grp.sort_values("T").reset_index(drop=True)
        if len(grp) < 2:
            raise ValueError(
                f"Need at least 2 expirations for a term structure; got {len(grp)}"
            )
        return cls(grp["T"].values, grp["sigma_atm"].values)

    def total_variance(self, T: float) -> float:
        """sigma^2 * T at arbitrary T, piecewise-linear in total variance."""
        if T <= 0.0:
            return 0.0
        # Linear interp in total-variance space; flat extrapolation at edges
        exps = self.expirations
        tv = self._total_var
        if T <= exps[0]:
            # Extrapolate to 0 assuming sigma_atm constant at the short end
            return self.atm_vols[0] ** 2 * T
        if T >= exps[-1]:
            # Extrapolate with the last-observed forward variance
            if len(exps) >= 2:
                fwd_var_last = (tv[-1] - tv[-2]) / (exps[-1] - exps[-2])
                return tv[-1] + fwd_var_last * (T - exps[-1])
            return self.atm_vols[-1] ** 2 * T
        return float(np.interp(T, exps, tv))

    def sigma(self, T: float) -> float:
        """ATM term vol at maturity T."""
        if T <= 0.0:
            return 0.0
        return math.sqrt(self.total_variance(T) / T)


# ======================================================================
#        FORWARD-QUANTITY BUILDERS FOR THE AUTOCALL SCHEDULE
# ======================================================================

def _forward_from_term(term_vals_times: np.ndarray,
                       term_vals: np.ndarray,
                       label: str,
                       floor: float = 1e-6) -> np.ndarray:
    """Convert cumulative term quantity `X_term(t) * t` into per-interval
    forwards over obs_times.

    `term_vals_times` are the observation times [t_1,...,t_N]; we assume
    t_0 = 0 with cumulative value 0 (consistent for variance, discount,
    dividend PV).
    """
    n = len(term_vals_times)
    fwd = np.empty(n, dtype=np.float64)
    prev_t = 0.0
    prev_tv = 0.0  # cumulative X * t at t=0
    for i in range(n):
        t = term_vals_times[i]
        # term_vals[i] is X_term(t_i), so cumulative = X_term(t_i) * t_i
        cum = term_vals[i] * t
        dt = t - prev_t
        if dt <= 0.0:
            raise ValueError(f"obs_times must be strictly increasing")
        x = (cum - prev_tv) / dt
        if x < floor:
            warnings.warn(
                f"{label}: negative/tiny forward on interval "
                f"({prev_t:.4f}, {t:.4f}) — floored to {floor:.1e}. "
                f"Suggests term-structure inversion; check input.",
                RuntimeWarning,
            )
            x = floor
        fwd[i] = x
        prev_t = t
        prev_tv = cum
    return fwd


def build_forward_vols(obs_times: np.ndarray,
                       vol_ts: ATMVolTermStructure) -> np.ndarray:
    """Forward vol per obs interval, from ATM term structure.

    fwd_vol_i = sqrt( (sigma_term(t_i)^2 * t_i - sigma_term(t_{i-1})^2 * t_{i-1}) / dt_i )
    """
    obs_times = np.asarray(obs_times, dtype=np.float64)
    # Evaluate term variance at each obs time via the ATM surface
    term_vols = np.array([vol_ts.sigma(t) for t in obs_times], dtype=np.float64)
    fwd_var = _forward_from_term(obs_times, term_vols ** 2, "fwd_var",
                                 floor=1e-6)
    return np.sqrt(fwd_var)


def build_forward_rates(obs_times: np.ndarray,
                        curve: YieldCurve) -> np.ndarray:
    """Continuously-compounded forward rate per obs interval.

    fwd_r_i = (r(t_i)*t_i - r(t_{i-1})*t_{i-1}) / dt_i
    """
    obs_times = np.asarray(obs_times, dtype=np.float64)
    term_r = np.array([curve.r_continuous(t) for t in obs_times],
                      dtype=np.float64)
    return _forward_from_term(obs_times, term_r, "fwd_rate", floor=-np.inf)


def build_forward_divs(obs_times: np.ndarray,
                       q_provider,
                       regularize: bool = True,
                       long_end_q_anchor=None) -> np.ndarray:
    """Continuous-equivalent forward dividend yield per obs interval.

    Parameters
    ----------
    obs_times : array-like of float
    q_provider : callable OR dict
        Either a callable `T -> q_continuous(T)` (e.g., from
        `deamerican._q_lookup_from_dict(parity_implied_q_by_expiry(...))`),
        or a dict `{T: q_continuous}` which will be wrapped automatically.
    regularize : bool, default True
        When ``q_provider`` is a dict, project the cumulative yield
        Q(T)=q(T)*T onto the non-decreasing cone before building forwards,
        and interpolate in Q-space so that monotonicity is preserved between
        grid points too. This eliminates the negative forward dividends that
        parity-implied q(T) quote noise otherwise produces. Set to False to
        preserve the legacy (linear-in-q, no-regularization) behaviour for
        diagnostics. Callable providers are used as-is either way.
    long_end_q_anchor : tuple(T_anchor, q_anchor) or float, optional
        Only honoured when ``q_provider`` is a dict. Appends a synthetic
        long-end q point so the curve blends toward an economically
        realistic yield (e.g., SPY TTM yield) past the last quoted expiry,
        instead of flat-extrapolating parity noise at the few-month horizon.
        A scalar is interpreted as ``q_anchor`` at ``T_anchor=1.0``.

    The forward is obtained by differencing cumulative `q_term(T) * T`:
        fwd_q_i * dt_i = q_term(t_i) * t_i - q_term(t_{i-1}) * t_{i-1}
    """
    obs_times = np.asarray(obs_times, dtype=np.float64)
    if isinstance(q_provider, dict):
        q_dict = q_provider
        if long_end_q_anchor is not None:
            if isinstance(long_end_q_anchor, (tuple, list)):
                T_anc, q_anc = long_end_q_anchor
            else:
                T_anc, q_anc = 1.0, float(long_end_q_anchor)
            q_dict = anchor_q_curve_long_end(q_dict, q_anc, T_anc)
        if regularize:
            q_dict = regularize_q_curve(q_dict)
            q_fn = _q_lookup_from_dict_Q(q_dict)
        else:
            q_fn = _q_lookup_from_dict(q_dict)
    else:
        q_fn = q_provider
    term_q = np.array([q_fn(float(t)) for t in obs_times], dtype=np.float64)
    floor = 0.0 if (regularize and isinstance(q_provider, dict)) else -np.inf
    return _forward_from_term(obs_times, term_q, "fwd_div", floor=floor)


# ======================================================================
#              ONE-SHOT AUTOCALL INPUT BUILDER
# ======================================================================

def build_autocall_inputs(obs_times: np.ndarray,
                          vol_ts: ATMVolTermStructure,
                          curve: YieldCurve,
                          q_provider,
                          spot: float = None,
                          regularize_q: bool = True,
                          long_end_q_anchor=None) -> dict:
    """Produce the full set of per-interval arrays required by
    `autocall_pricer_prod.make_single_spec`.

    The caller still owns product-level choices: `strike`, `ki_barrier`,
    `ac_barriers`, `coupons`, `notional`, `continuous_ki`. This function
    supplies only the market-implied pieces.

    Parameters
    ----------
    obs_times : array-like of float
        Observation dates in years from valuation date, strictly increasing.
    vol_ts : ATMVolTermStructure
    curve : YieldCurve
    q_provider : callable or dict
        Continuous yield as a function of maturity T. The standard
        backtest-safe construction is:
            q_dict = deamerican.parity_implied_q_by_expiry(chain, S0, curve)
        and pass `q_dict` here. For a forward-starting product whose obs
        dates extend beyond the option surface, q is flat-extrapolated.
    spot : float, optional
        Unused here; kept for signature symmetry with earlier drafts.

    Returns
    -------
    dict with keys
        obs_times, fwd_vols, fwd_rates, fwd_divs
    """
    obs_times = np.asarray(obs_times, dtype=np.float64)
    if obs_times.ndim != 1:
        raise ValueError("obs_times must be 1-D")
    if len(obs_times) < 1:
        raise ValueError("obs_times must be non-empty")
    if np.any(np.diff(obs_times) <= 0):
        raise ValueError("obs_times must be strictly increasing")
    if obs_times[0] <= 0:
        raise ValueError("obs_times[0] must be > 0")

    fv = build_forward_vols(obs_times, vol_ts)
    fr = build_forward_rates(obs_times, curve)
    fd = build_forward_divs(obs_times, q_provider,
                            regularize=regularize_q,
                            long_end_q_anchor=long_end_q_anchor)

    return {
        "obs_times": obs_times.copy(),
        "fwd_vols": fv,
        "fwd_rates": fr,
        "fwd_divs": fd,
    }


# ======================================================================
#                         DIAGNOSTICS
# ======================================================================

def build_autocall_inputs_lv(obs_times: np.ndarray,
                             deam_df: pd.DataFrame,
                             spot: float,
                             curve: YieldCurve,
                             q_provider,
                             log_m_bounds: tuple = (-0.8, 0.8),
                             n_m: int = 161,
                             n_t: int = 101,
                             t_max: float = None,
                             iv_col: str = "american_iv",
                             regularize_q: bool = True,
                             long_end_q_anchor=None) -> dict:
    """Like ``build_autocall_inputs`` but additionally returns a Dupire
    local-vol surface for the prod/ sub-stepped pricer.

    Returns a dict containing everything ``build_autocall_inputs`` returns
    (``obs_times``, ``fwd_vols``, ``fwd_rates``, ``fwd_divs`` — still useful
    as a fallback / control variate baseline), *plus*:

        log_m_grid : np.ndarray [n_m]
        t_grid     : np.ndarray [n_t]
        sigma_loc  : np.ndarray [n_m, n_t]     (row-major C-contiguous)
        use_local_vol : bool (always True)

    The pricer consumes these as ``make_single_spec(..., sigma_loc_grid=...)``
    (Phase 2 extension). If the pricer is called without the LV grid, the
    piecewise-constant ``fwd_vols`` will be used instead.

    Parameters
    ----------
    obs_times : array-like
        Observation dates in years from valuation, strictly increasing.
    deam_df : DataFrame
        De-Americanized chain (output of ``deamericanize_chain``) — the same
        object from which ``ATMVolTermStructure.from_chain`` is built.
    spot : float
    curve : YieldCurve
    q_provider : callable or dict
    log_m_bounds, n_m, n_t, t_max : grid spec (see LocalVolSurface.from_iv_surface).
    iv_col : str
        Column with European BS IV. Default ``american_iv`` (see deamerican.py).
    """
    obs_times = np.asarray(obs_times, dtype=np.float64)

    # Reuse the ATM-term-structure path to get fwd_vols (still useful as a
    # baseline and to keep the CV analytic exact in fallback mode).
    vol_ts = ATMVolTermStructure.from_chain(deam_df, spot=spot)
    base = build_autocall_inputs(obs_times, vol_ts, curve, q_provider,
                                 regularize_q=regularize_q,
                                 long_end_q_anchor=long_end_q_anchor)

    if t_max is None:
        # Cover a bit past the last observation to keep grid derivatives
        # meaningful at the terminal node.
        t_max = float(obs_times[-1]) * 1.10

    lv = build_local_vol_grid_from_chain(
        deam_df, spot=spot, curve=curve, q_provider=q_provider,
        log_m_bounds=log_m_bounds, n_m=n_m, t_max=t_max, n_t=n_t,
        iv_col=iv_col,
    )

    base.update({
        "log_m_grid": np.ascontiguousarray(lv.log_m_grid, dtype=np.float64),
        "t_grid":     np.ascontiguousarray(lv.t_grid, dtype=np.float64),
        "sigma_loc":  np.ascontiguousarray(lv.sigma_loc, dtype=np.float64),
        "use_local_vol": True,
    })
    return base


def summarize_inputs(inputs: dict) -> pd.DataFrame:
    """Human-readable per-interval summary of an autocall input bundle."""
    t = inputs["obs_times"]
    n = len(t)
    prev_t = np.concatenate([[0.0], t[:-1]])
    return pd.DataFrame({
        "t_start": prev_t,
        "t_end": t,
        "dt": t - prev_t,
        "fwd_vol": inputs["fwd_vols"],
        "fwd_rate": inputs["fwd_rates"],
        "fwd_div": inputs["fwd_divs"],
    })
