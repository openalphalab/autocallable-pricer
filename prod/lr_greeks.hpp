// lr_greeks.hpp
// Likelihood-ratio (LR) method for Greeks on products with discontinuous payoffs.
//
// Principle: instead of differentiating the payoff (which has Dirac-delta
// contributions at barriers), differentiate the probability density of the
// simulated path w.r.t. the parameter. This gives an unbiased estimator with
// finite variance.
//
// For GBM under Black dynamics, the log-return X_1 = log(S_{t_1}/S_0)
// over the first interval has density:
//     X_1 ~ N(mu_1, nu_1^2)   where mu_1 = (r-q-0.5*sig^2)*dt_1,  nu_1 = sig*sqrt(dt_1)
//
// The LR weight for Delta is the score of this density w.r.t. S_0:
//     psi_Delta = d/dS_0 log p(X_1 | S_0)
//               = d/dS_0 [- (X_1 - mu_1 + log(S_0))^2 / (2*nu_1^2)]   (via change of vars)
// After simplification for GBM:
//     psi_Delta = Z_1 / (S_0 * nu_1)
// where Z_1 = (X_1 - mu_1) / nu_1 is the standardized first-interval normal.
//
// Gamma LR weight:
//     psi_Gamma = (Z_1^2 - Z_1*nu_1 - 1) / (S_0^2 * nu_1^2)
//
// Then:
//     Delta = E[payoff * psi_Delta]
//     Gamma = E[payoff * psi_Gamma]
//
// These are UNBIASED for any payoff (no smoothing needed). Variance is
// typically higher than pathwise for smooth payoffs, which is why we use
// pathwise for Vega/Rho (which are smooth) and LR for Delta/Gamma.
//
// NOTE: LR only "sees" the first-interval randomness. This is correct under
// Markov dynamics -- the rest of the path depends on S_0 only through S_{t_1}.

#pragma once
#include <cmath>
#include <vector>

namespace lr_greeks {

// Compute LR weights for a single path given the first normal Z_1 and
// first-interval vol * sqrt(dt).
struct LRWeights {
    double delta;   // psi for Delta
    double gamma;   // psi for Gamma
};

static inline LRWeights compute_lr_weights(double Z1, double S0, double nu1) {
    LRWeights w;
    w.delta = Z1 / (S0 * nu1);
    w.gamma = (Z1 * Z1 - Z1 * nu1 - 1.0) / (S0 * S0 * nu1 * nu1);
    return w;
}

// ---------------------------------------------------------------
// Vega via pathwise differentiation.
// For autocallables, Vega is not purely "smooth" because barriers move with
// the path, but the payoff itself is not a direct function of vol -- only
// through the path. We use the pathwise method here, integrating over the
// path:
//     dPV/dsig_i = pathwise sensitivity propagated through log_s.
// This term-structure vega (vega bucket per interval) is what traders want.
//
// Implementation: during path simulation, track d(log_s)/d(sig_i) which is
// simply sig_i * dt_i contribution from the log-return formula.
// Since log_s = sum_j [(r_j - q_j - 0.5*sig_j^2)*dt_j + sig_j*sqrt(dt_j)*Z_j]
// we have d(log_s)/dsig_i = -sig_i*dt_i + sqrt(dt_i)*Z_i  (only at step i and after)
//
// This is the "smooth part"; at barrier crossings it's discontinuous, but we
// can still use pathwise IF we smooth the barrier indicator with a narrow
// sigmoid, OR we can use a LR weight for each vol bucket too.
// ---------------------------------------------------------------

// Score function for a single vol bucket's LR weight (applies when we
// differentiate the density of Z_i given (sig_i, dt_i, S_{t_{i-1}}))
// Under constant-parameter GBM for interval i:
//   X_i = log(S_{t_i}/S_{t_{i-1}}) ~ N((r-q-0.5*sig^2)*dt, sig^2*dt)
// Score w.r.t. sig:
//   psi_sig_i = [ (X_i - (r-q)*dt)^2 / sig^3 / dt - 1/sig + (X_i - (r-q)*dt)/sig ]
// Equivalent form in standardized Z_i = (X_i - mu_i)/nu_i:
//   psi_sig = (Z_i^2 - 1) / sig_i  -  Z_i * sqrt(dt_i)
// This is the LR Vega for vol bucket i.
static inline double lr_vega_weight(double Zi, double sig_i, double dt_i) {
    double sqdt = std::sqrt(dt_i);
    return (Zi * Zi - 1.0) / sig_i - Zi * sqdt;
}

// Score function for rate (continuously compounded r)
//   psi_r_i = Z_i * dt_i / (sig_i * sqrt(dt_i))  =  Z_i * sqrt(dt_i) / sig_i
// But we also get a direct term from the discounting:  -dt_i * payoff.
// Caller handles the discount part; this returns only the "path" part.
static inline double lr_rho_path_weight(double Zi, double sig_i, double dt_i) {
    return Zi * std::sqrt(dt_i) / sig_i;
}

} // namespace lr_greeks
