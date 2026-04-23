# "Vibe" Autocallable Pricing

A weekend project: an ultra-fast Monte Carlo autocallable pricer for SPY,
built end-to-end through conversation with Claude Opus 4.7. From raw
American-style option prices to a calibrated pricer, a coupon solver, a
seasoned-note backtest, and a delta-replicated hedging strategy.

> **Disclaimer.** Essentially every line of code in this repo was written by
> an LLM. It prices and backtests cleanly on the bundled data, but this is an
> experiment — **not production software, not trading advice**. Review it
> carefully before using it for anything that matters.

## What's in the box

- **De-Americanisation** of the SPY option chain (BAW) with parity-implied
  spot recovery and per-expiry dividend yields — see
  [util/deamerican.py](util/deamerican.py).
- **Forward-vol / forward-rate / forward-div** construction on the observation
  grid from the cleaned chain, with TTM-dividend long-end anchoring —
  [util/autocall_prep.py](util/autocall_prep.py).
- **Compiled Monte Carlo pricer** (pybind11 + AVX2/AVX-512) with:
  - Piecewise-constant σ per observation interval, `K` Euler sub-steps per
    interval, continuous knock-in monitoring via Brownian bridge.
  - Philox or scrambled Sobol QMC (with automatic fallback when Sobol
    dimension > 200).
  - Control-variate variance reduction.
  - LR Delta / Gamma, pathwise Vega per bucket, Rho.
  - Optional local-vol surface σ(S, t) hook (see [prod/README.md](prod/README.md)).

  Source lives in [prod/](prod/); a prebuilt CPython 3.10 Windows x64
  extension ships as [`prod/autocall_pricer_lv.cp310-win_amd64.pyd`](prod/).
- **Worked examples** covering the three milestones from the write-up:
  1. [example.py](example.py) — single-date pricing with a coupon solver.
  2. [example_backtest.py](example_backtest.py) — seasoned-note backtest of
     one autocallable plus a self-financing delta-replicating portfolio.
  3. [backtest_two_notes.py](backtest_two_notes.py) — flat-barrier vs
     laddered-down barrier comparison on the same issue date, with
     performance stats (CAGR, vol, Sharpe vs 3M CMT, max drawdown) and a
     spot-normalised side-by-side time series.

## Repo layout

```
.
├── example.py               # single-date pricing + coupon solver
├── example_backtest.py      # seasoned-note backtest + delta replication
├── backtest_two_notes.py    # flat vs laddered barrier comparison
├── get_spot.py              # refresh data/SPY_spot.csv from Yahoo Finance
├── prod/                    # compiled MC pricer (C++ + pybind11)
├── util/                    # de-Americanisation, prep, local-vol, spot data
└── data/                    # bundled market data (SPY options, spot, rates)
```

## Install

Requires Python 3.10 on Windows x64 to use the bundled prebuilt extension
as-is. On other platforms / Python versions, rebuild the extension.

```bash
python -m venv .venv
.venv\Scripts\activate         # PowerShell: .venv\Scripts\Activate.ps1
pip install numpy pandas scipy numba yfinance pybind11
```

Rebuild the pricer (any time you're not on CPython 3.10 Windows x64):

```bash
python prod/setup.py build_ext --inplace
```

AVX2+FMA is required; AVX-512F+DQ is autodetected at runtime.

## Quickstart

```bash
# Price one default 5Y annual-obs SPY autocallable at the latest trade date
# in the bundled chain, solving the per-obs coupon to par:
python example.py

# Override the trade date, note terms, or MC settings:
python example.py --date 2020-01-03 --ac-barriers 1.0,0.95,0.90,0.85,0.80 \
                  --ki-barrier 0.70 --paths 200000
```

```bash
# Seasoned-note backtest + self-financing delta hedge,
# writing outputs/autocall_backtest_<issue>.xlsx
python example_backtest.py --issue-date 2020-01-03

# Flat vs laddered comparison (two tabs + stats),
# writing outputs/autocall_two_notes_<issue>.xlsx
python backtest_two_notes.py
```

## Data setup

The repo ships a bundled slice of SPY options ([data/SPY.csv](data/SPY.csv)),
SPY spot closes ([data/SPY_spot.csv](data/SPY_spot.csv)), and US Treasury
CMT rates ([data/rates.csv](data/rates.csv)) so the examples run out of the
box. To refresh or extend:

- **Options and rates** — pull fresh CSVs from DoltHub per
  [data/README.md](data/README.md).
- **Spot closes** — `python get_spot.py` refreshes `data/SPY_spot.csv` from
  Yahoo Finance across the date range of the options file.

## Known limitations

- The bundled DoltHub option chain is a free community dataset, not an
  institutional feed. Good enough for experiments, not for anything live.
- SPY options are American-style; we de-Americanise with BAW, which is a
  reasonable but imperfect approximation.
- Expirations in the chain typically only go out ~2 months, so long-dated
  autocallables (e.g. 5Y) lean heavily on extrapolation. The long end of the
  dividend curve is anchored with TTM yield from Yahoo to partly compensate;
  it is still the weakest link.
- Worst-of multi-asset baskets are supported without sub-stepping only
  (`K=1`); per-asset Cholesky at every sub-step is deferred.
- The Sobol stream caps at dimension 200. Large `K · N` (or `2 · K · N` with
  continuous-KI) falls back to Philox automatically.


## License

MIT — see [LICENSE](LICENSE).
