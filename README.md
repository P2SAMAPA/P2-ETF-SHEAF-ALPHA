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
Sharpe, mean/std return, n_predictions) plus sheaf-specific diagnostics:

- `avg_sheaf_energy` — average market-dislocation index over the test period
- `avg_reg_slope` / `avg_reg_r2` — how strong and how reliable the
  gap → next-return relationship was, averaged across walk-forward steps
- `energy_series` — the full per-day sheaf energy time series for the best
  window's test period, plotted in the dashboard

## Caveats (same spirit as the SINDy repo's)

- No transaction costs or slippage are modeled.
- Correlation-based restriction maps are linear approximations; genuine
  regime shifts (correlation breakdowns) will degrade them until the next
  refit.
- A positive backtest Sharpe over a few hundred walk-forward days is weak
  statistical evidence of durable edge, not strong evidence — these are
  daily-correlated observations, not independent trials.
- This is not financial advice, and none of this should be traded live
  without further validation (out-of-sample period extension, cost
  modeling, and sanity-checking that results aren't driven by a handful of
  outlier days).
