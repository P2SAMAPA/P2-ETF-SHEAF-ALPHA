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
}

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
