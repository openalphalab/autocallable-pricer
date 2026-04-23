// brownian_bridge.hpp
// Brownian bridge construction: given N i.i.d. N(0,1) draws Z[0..N-1],
// produces correlated increments dW[0..N-1] between observation times
// t_0=0 < t_1 < ... < t_N, such that W(t_i) = sum_{j<=i} dW[j] follows
// a Brownian motion.
//
// The key trick: we assign Sobol dimensions in a bridge order so that the
// first Sobol dimensions (which have the best equidistribution) fix the
// coarse structure of W (terminal value first, then midpoints recursively).
// This concentrates variance into low dimensions = huge QMC convergence gain.

#pragma once
#include <vector>
#include <cmath>
#include <cstdint>

class BrownianBridge {
public:
    // times[0..N-1] are observation times t_1,...,t_N (with implicit t_0 = 0).
    // We construct a bridge schedule: a permutation of {0,..,N-1} such that
    // dimension 0 samples W(t_N), dimension 1 samples the midpoint, etc.
    explicit BrownianBridge(const std::vector<double>& times) : t_(times) {
        int N = (int)times.size();
        if (N == 0) return;
        // Sort checks
        for (int i = 0; i < N; ++i) {
            if (t_[i] <= (i == 0 ? 0.0 : t_[i-1])) {
                // non-increasing -> fall back to sequential (no bridge)
                bridge_build_ = false;
                return;
            }
        }
        bridge_build_ = true;
        // Build bridge schedule.
        // schedule_[k] = (idx, left, right, sigma_mid) where:
        //   idx   = time index being sampled at step k (0..N-1)
        //   left  = index of previously sampled point on the left (or -1 for t=0)
        //   right = index of previously sampled point on the right (or -1 for terminal)
        //   For terminal: W(t_idx) ~ N(0, t_idx), so sigma = sqrt(t_idx)
        //   For bridge:   W(t_idx) | W(t_L), W(t_R) ~ N(mu, sigma^2)
        //       where mu = W(t_L) + (t_idx - t_L)/(t_R - t_L) * (W(t_R) - W(t_L))
        //       sigma^2 = (t_idx - t_L)(t_R - t_idx)/(t_R - t_L)
        schedule_.clear();
        schedule_.reserve(N);
        std::vector<bool> assigned(N, false);
        // Step 0: sample terminal
        schedule_.push_back({N - 1, -1, -1, 0.0, 0.0, std::sqrt(t_[N-1])});
        assigned[N - 1] = true;

        // BFS-style bisection: maintain a queue of (left_idx, right_idx) intervals
        // where left_idx can be -1 meaning t=0.
        struct Interval { int L, R; };
        std::vector<Interval> queue = {{-1, N - 1}};

        while ((int)schedule_.size() < N) {
            std::vector<Interval> next_q;
            for (auto iv : queue) {
                int L = iv.L, R = iv.R;
                // pick middle unassigned index between L and R
                int lo = L + 1;
                int hi = R - 1;
                if (lo > hi) continue;
                int mid = (lo + hi) / 2;
                if (assigned[mid]) continue;

                double tL = (L < 0) ? 0.0 : t_[L];
                double tR = t_[R];
                double tM = t_[mid];
                double a = (tM - tL) / (tR - tL);  // weight on W(t_R)
                double b = 1.0 - a;                 // weight on W(t_L)
                double sigma = std::sqrt((tM - tL) * (tR - tM) / (tR - tL));
                schedule_.push_back({mid, L, R, a, b, sigma});
                assigned[mid] = true;

                next_q.push_back({L, mid});
                next_q.push_back({mid, R});
            }
            queue = next_q;
            if (queue.empty()) break;  // safety
        }
    }

    int dim() const { return (int)schedule_.size(); }
    bool enabled() const { return bridge_build_; }

    // Given Z[0..N-1] i.i.d. N(0,1), fill W[0..N-1] = W(t_1)..W(t_N)
    // using the bridge construction.
    void build_path(const double* Z, double* W) const {
        int N = (int)schedule_.size();
        if (!bridge_build_) {
            // Fallback: sequential (plain Brownian motion)
            double prev_t = 0.0;
            double prev_w = 0.0;
            for (int i = 0; i < N; ++i) {
                double dt = t_[i] - prev_t;
                prev_w += std::sqrt(dt) * Z[i];
                W[i] = prev_w;
                prev_t = t_[i];
            }
            return;
        }
        for (int k = 0; k < N; ++k) {
            const auto& s = schedule_[k];
            if (s.L < 0 && s.R == (int)t_.size() - 1 && k == 0) {
                // Terminal draw
                W[s.idx] = s.sigma * Z[k];
            } else {
                double wL = (s.L < 0) ? 0.0 : W[s.L];
                double wR = W[s.R];
                // mean = (1-a)*wL + a*wR   where a = (tM-tL)/(tR-tL)
                double mean = s.b * wL + s.a * wR;
                W[s.idx] = mean + s.sigma * Z[k];
            }
        }
    }

private:
    struct Step {
        int idx;     // index in t_ being sampled at this step
        int L, R;    // left/right already-sampled indices (L can be -1 = t=0)
        double a, b; // bridge interp weights: mean = b*W_L + a*W_R
        double sigma;// conditional std dev
    };
    std::vector<double> t_;
    std::vector<Step> schedule_;
    bool bridge_build_ = false;
};
