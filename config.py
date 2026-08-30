"""
config.py  —  Configuration for P2 SHEAF-ALPHA
"""

import os

HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-sheaf-alpha-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"],
    "COMBINED": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV", "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"]
}

# Multiple windows for analysis
WINDOWS = [126, 252, 504, 756, 1008]

# Sheaf-Alpha configuration
#
# k_neighbors      : how many of each node's most-correlated neighbors form
#                     its sheaf edges (a sparse, interpretable graph rather
#                     than a fully-connected one).
# min_abs_corr      : neighbors with |correlation| below this are dropped
#                     even if they'd otherwise make the top-k cut — a weak
#                     restriction map is not a meaningful consistency
#                     constraint, and including it just adds noise to the
#                     gap signal.
# use_macro         : include macro variables (VIX, yield curve, DXY, credit
#                      spreads, ...) as extra nodes in the sheaf, so ETFs can
#                      be flagged inconsistent with macro conditions, not
#                      just with each other.
# min_train_samples : minimum rows required in a training window before a
#                      graph + regression is even attempted.
SHEAF_CONFIG = {
    "k_neighbors": 5,
    "min_abs_corr": 0.15,
    "use_macro": True,
    "min_train_samples": 30,
    # Fraction of a universe's history reserved purely as a global
    # "burn-in" before walk-forward testing starts. This is deliberately
    # small: it does NOT reduce how much data each individual model is
    # trained on (that's always exactly `window` days immediately
    # preceding each test point, enforced separately) -- it only controls
    # how far back testing is ALLOWED to start. A large fraction here
    # (0.8, the old default) wastes most of the dataset's history: it was
    # producing only 427-874 out-of-sample predictions per universe, with
    # standard errors on the resulting correlation estimates (~1/sqrt(n))
    # too large to distinguish real signal from noise. Each window is
    # still individually protected from starting before it has `window`
    # valid days of prior data (see the train_start < 0 guard in
    # backtest_window), so lowering this is safe -- it just lets every
    # window use as much of the eligible history as it actually can.
    "burn_in_fraction": 0.05,
}

# Small hyperparameter grid searched PER WINDOW, per universe, selecting
# the combination with the best out-of-sample correlation (consistent
# with BEST_WINDOW_METRIC below -- prediction quality drives selection,
# not backtested P&L). Only k_neighbors / min_abs_corr are searched, since
# those are the two levers that most directly control what counts as a
# "consistency constraint" in the sheaf -- how many neighbors contribute
# to a node's consensus estimate, and how strong a relationship has to be
# before it's trusted at all.
#
# IMPORTANT CAVEAT: searching more combinations increases the risk of
# picking one that looks good by pure chance (the "best of N noisy
# estimates" problem) -- this grows with grid size. The trainer reports
# the FULL comparison table, not just the winner, specifically so this
# can be checked: if the winning combination isn't meaningfully better
# than the rest of the grid, its apparent edge should be treated as noise
# regardless of which number happened to come out on top.
SHEAF_GRID = [
    {"k_neighbors": 3, "min_abs_corr": 0.15},
    {"k_neighbors": 5, "min_abs_corr": 0.15},
    {"k_neighbors": 8, "min_abs_corr": 0.15},
    {"k_neighbors": 5, "min_abs_corr": 0.10},
    {"k_neighbors": 5, "min_abs_corr": 0.25},
    {"k_neighbors": 5, "min_abs_corr": 0.35},
]

TOP_N = 3

# Round-trip trading cost assumption, in basis points, applied to every
# position change (turnover) in the backtest. 15 bps is a reasonable
# generic estimate for liquid ETF bid-ask spread + slippage; adjust to
# match your actual execution costs.
TRADING_COST_BPS = 15

# How the "best window" per universe is chosen for Tab 1's live picks.
# "correlation" selects the window whose return PREDICTIONS were most
# accurate historically (predicted vs. actual return correlation) -- a
# direct measure of prediction quality. The previous default, "sharpe",
# reflects backtested P&L, which can be inflated by a window simply
# riding the test period's market drift even when its predictions barely
# explain anything (this was observed directly: the highest-Sharpe window
# for EQUITY_SECTORS had by far the weakest R² of any window tested).
BEST_WINDOW_METRIC = "correlation"
