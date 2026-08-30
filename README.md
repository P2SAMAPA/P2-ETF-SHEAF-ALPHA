# P2-ETF-SHEAF-ALPHA

Financial inconsistency alpha via **cellular sheaves** over ETF and macro relationships.

## The idea

A graph says *these nodes are connected*. A cellular sheaf says more: each node
carries a locally-defined observation, and each edge carries a **restriction
map** describing how two connected observations should relate if the system
is internally consistent. When observed data violates that relation, the
violation is measurable and localizable — and, if markets take time to
arbitrage dislocations away, potentially predictive.

This repo builds a lightweight cellular sheaf over ETF returns and macro
changes:

- **Nodes** — each ETF's daily log-return, plus each macro series' daily
  change (VIX, T10Y2Y, DXY, IG/HY credit spreads). All node values are
  standardized within the training window so every node lives on a
  comparable "one-day shock" scale.
- **Edges** — each node connects to its `k` most-correlated other nodes
  (above a minimum correlation threshold), forming a sparse, interpretable
  graph rather than a fully-connected one.
- **Restriction maps** — for standardized variables, the natural linear
  consistency map between two connected nodes is exactly their fitted
  correlation `ρ`, which works symmetrically in either direction.
- **Sheaf disagreement ("the gap")** — a node's neighbors, run through their
  restriction maps, each imply a value for that node. The confidence-weighted
  average of those implied values, minus the node's actual value, is the
  local section disagreement: literally *local state disagreement*.
- **The alpha** — whether disagreement at time `t` predicts a node's return
  at `t+1` is tested empirically, by pooling `(gap, next-return)` pairs
  across every ticker and every day in the training window and fitting a
  single regression. This reports directly whether the learned relationship
  is real signal (non-trivial slope, non-trivial R²) or effectively noise —
  it never assumes the effect exists.
- **Sheaf energy** — the sum of squared edge disagreements on a given day is
  the sheaf Laplacian quadratic form for that day, a standard measure of how
  far the day's data is from a consistent "global section." Reported as a
  market-dislocation index alongside the trading signal.

### Honest scope note

The framing that motivated this ("ETF price / implied vol / underlying
basket / sector index / futures / options as different local observations
of one underlying state") describes a richer sheaf than what's built here —
this repo's dataset only has ETF **prices** and a handful of **macro**
series, not options, futures, or an implied-vol surface. The sheaf here is
built over what's actually available: cross-ETF and ETF-macro correlation
structure. That's a real, defensible cellular sheaf — just a narrower one
than the full multi-instrument version the idea describes. Extending it to
options/futures data would be a natural next step if that data becomes
available.

## Repo structure

Same shape as the sister repo (`P2-ETF-SINDY-ETF-DYNAMICS`), swapping the
algorithm:

```
config.py          Universes, windows, HF repo IDs, SHEAF_CONFIG
data_manager.py     Loads the master parquet from HF (unchanged from SINDy repo)
sheaf_model.py       SheafModel: graph construction, restriction maps, gap
                      computation, pooled regression, sheaf energy
trainer.py           Walk-forward backtesting across window sizes + per-window
                      ETF picks, mirrors the SINDy trainer's orchestration shape
push_results.py      Uploads results JSON to the HF results dataset (unchanged)
us_calendar.py        Trading-day utilities (unchanged)
streamlit_app.py      Dashboard: live picks + backtest diagnostics, including
                       an actual sheaf network graph (no bar charts)
requirements.txt      Adds networkx (for the network graph layout) to the
                       SINDy repo's dependency set
```

## Running it

```bash
pip install -r requirements.txt
python trainer.py          # walk-forward backtest + live picks, pushes to HF
streamlit run streamlit_app.py
```

`HF_TOKEN` must be set in the environment for `push_results.py` to upload.
`config.RESULTS_REPO` currently points at `P2SAMAPA/p2-sheaf-alpha-results`
— create that dataset repo on HF before the first run, or change the value.

> **Note on this delivery**: this sandbox's network access does not include
> `huggingface.co`, so the code here was validated with synthetic
> data (unit tests of the sheaf mechanism, full-pipeline dry runs, and a
> headless Streamlit smoke test) rather than against the real master
> parquet or a live HF push. Please run `trainer.py` yourself against the
> real data as the first real test.

## What the backtest reports, per window per universe

Same metric set as the SINDy repo (correlation, MSE, directional accuracy,
mean/std return, n_predictions) plus sheaf-specific diagnostics:

- `sharpe` / `sharpe_gross` — Sharpe **net** of trading costs (see below)
  and **before** costs, so the cost drag is visible directly rather than
  hidden inside one number.
- `avg_daily_cost_bps` — average trading cost actually paid per day
  (turnover-driven; a signal that rarely changes its mind costs little
  even at a nonzero `TRADING_COST_BPS`, and vice versa).
- `avg_sheaf_energy` — average market-dislocation index over the test period.
- `avg_reg_slope` / `avg_reg_r2` — how strong and how reliable the
  gap → next-return relationship was, averaged across walk-forward steps.
- `energy_series` — the full per-day sheaf energy time series for the best
  window's test period, plotted in the dashboard.
- `k_neighbors` / `min_abs_corr` — the sheaf hyperparameter combination
  that won this window's search (see below).
- `hyperparam_search` — the full comparison table of every combination
  tried for this window, not just the winner.

## Trading costs

`backtest_window` applies a **turnover-based** cost (`config.TRADING_COST_BPS`,
default 15bps), charged only when a ticker's position actually changes
sign from one day to the next — a position held unchanged costs nothing
extra. This matters: in testing, several windows' apparent edge collapsed
almost entirely once costs were applied (e.g. one window went from +0.44
gross Sharpe to essentially 0 net Sharpe), which is exactly the kind of
thing a cost-free backtest would hide.

## How the "best window" and sheaf hyperparameters are selected

Two related design choices, both aimed at the same problem: **Sharpe is
not used to select anything**, because it reflects realized P&L, which
can look good from a window whose predictions barely explain any real
variance (this was observed directly — the highest-gross-Sharpe window in
an early run had the *weakest* R² of any window tested, consistent with
it mostly riding the test period's market drift rather than detecting
genuine disagreement).

- **Best window** (`config.BEST_WINDOW_METRIC`, default `"correlation"`):
  selected by predicted-vs-actual return correlation — a direct measure
  of whether the predictions themselves are any good.
- **Sheaf hyperparameters** (`config.SHEAF_GRID`): for each window, a
  small grid of `(k_neighbors, min_abs_corr)` combinations is backtested,
  and the one with the best out-of-sample correlation is used both for
  that window's reported metrics AND for generating that window's live
  picks (so the two are always consistent with each other).

**Multiple-comparisons caveat, stated plainly**: searching more
combinations increases the chance that the "best" one simply got lucky on
this particular test period, even if no combination is actually better
than any other. This is why the full `hyperparam_search` comparison table
is kept and surfaced in the dashboard (Tab 2, per window) rather than
only reporting the winner — if the winning combination isn't clearly
ahead of the rest of the grid, its apparent edge should be treated as
noise, not as a validated finding.

## Backtest coverage (`SHEAF_CONFIG["burn_in_fraction"]`)

The walk-forward test still can't manufacture more real market history
than exists in the source dataset, but it can use much more of what's
already there. Each window is only ever trained on the `window` days
immediately preceding a given test point (enforced by a per-window guard,
regardless of this setting) — `burn_in_fraction` just controls how early
in the dataset's history walk-forward testing is *allowed* to start. The
previous default (0.8) reserved 80% of history purely as burn-in before
any testing began, producing only 427-874 out-of-sample predictions per
universe — with `n` that small, the standard error on a correlation
estimate (~1/√n) is too large to reliably tell real signal from noise.
Lowering this to 0.05 roughly tripled to quintupled the usable
out-of-sample sample size in testing, with no reduction in per-window
training data and no leakage introduced.

## Caveats (same spirit as the SINDy repo's)

- Correlation-based restriction maps are linear approximations; genuine
  regime shifts (correlation breakdowns) will degrade them until the next
  refit.
- A positive backtest Sharpe over a few hundred walk-forward days is weak
  statistical evidence of durable edge, not strong evidence — these are
  daily-correlated observations, not independent trials. Extending
  backtest coverage helps, but doesn't eliminate this.
- The multiple-comparisons risk from the hyperparameter grid search
  (above) applies on top of the usual sample-size caveat.
- This is not financial advice, and none of this should be traded live
  without further validation and sanity-checking that results aren't
  driven by a handful of outlier days.

