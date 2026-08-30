"""
trainer.py  —  P2 SHEAF-ALPHA Trainer with Multi-Window Backtesting

Builds a cellular sheaf over each universe's ETFs (plus macro variables as
extra nodes), and tests whether the resulting "local state disagreement"
signal (see sheaf_model.py) predicts next-day returns, across several
training window sizes.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_manager import load_master_data, validate_data
from sheaf_model import SheafModel, get_sheaf_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def backtest_window(V: np.ndarray, ticker_indices: List[int], window: int,
                     sheaf_config: Dict) -> Dict:
    """
    Walk-forward backtest of the sheaf-disagreement signal on one window size.

    V: (n_samples, n_nodes) node values -- ETF returns + macro changes,
       row-aligned across the whole universe (tickers occupy the first
       len(ticker_indices) columns).
    """
    n_samples = len(V)
    if n_samples < window + 50:
        return {"error": "Insufficient data", "window": window}

    train_size = int(n_samples * 0.8)
    min_train = sheaf_config.get("min_train_samples", 30)

    predictions = []
    actuals = []
    energies = []
    slopes = []
    r2s = []

    for i in range(train_size, n_samples - 1):
        train_start = i - window
        if train_start < 0:
            continue

        V_train = V[train_start:i]
        if len(V_train) < min_train:
            continue

        try:
            model = SheafModel(
                k_neighbors=sheaf_config.get("k_neighbors", 5),
                min_abs_corr=sheaf_config.get("min_abs_corr", 0.15),
            )
            fit_result = model.fit(V_train, node_names=[""] * V.shape[1], ticker_indices=ticker_indices)

            V_current = V[i]
            pred_returns, _ = model.predict(V_current, ticker_indices)
            Z_current = model._apply_standardize(V_current.reshape(1, -1))[0]
            energy = model.sheaf_energy(Z_current)

            actual_returns = V[i + 1, ticker_indices]

            predictions.append(pred_returns)
            actuals.append(actual_returns)
            energies.append(energy)
            slopes.append(fit_result["reg_slope"])
            r2s.append(fit_result["reg_r2"])
        except Exception:
            continue

    if len(predictions) < 10:
        return {"error": "Not enough predictions", "window": window}

    predictions = np.array(predictions)  # (n_steps, n_tickers)
    actuals = np.array(actuals)

    correlation = np.corrcoef(predictions.flatten(), actuals.flatten())[0, 1]
    mse = np.mean((predictions - actuals) ** 2)

    pred_sign = np.sign(predictions)      # (n_steps, n_tickers)
    actual_sign = np.sign(actuals)
    directional_accuracy = np.mean(pred_sign.flatten() == actual_sign.flatten())

    gross_returns = actuals * pred_sign   # (n_steps, n_tickers), pre-cost

    # Turnover-based trading costs: a cost is only paid when a ticker's
    # position (long/flat/short, from pred_sign) actually CHANGES from one
    # day to the next -- not on every day a position is simply held. The
    # first day is charged as trading in from flat (previous position 0).
    # cost_bps is applied per unit of position change, so a full flip
    # (long -> short) costs 2x a simple entry, matching the extra notional
    # actually traded.
    cost_bps = sheaf_config.get("trading_cost_bps", 15)
    prev_position = np.zeros((1, pred_sign.shape[1]))
    position_history = np.vstack([prev_position, pred_sign])  # (n_steps+1, n_tickers)
    turnover = np.abs(np.diff(position_history, axis=0))       # (n_steps, n_tickers)
    trading_cost = (cost_bps / 10000.0) * turnover

    net_returns = gross_returns - trading_cost

    sharpe_gross = np.mean(gross_returns.flatten()) / (np.std(gross_returns.flatten()) + 1e-8) * np.sqrt(252)
    sharpe_net = np.mean(net_returns.flatten()) / (np.std(net_returns.flatten()) + 1e-8) * np.sqrt(252)

    return {
        "window": window,
        "n_predictions": len(predictions),
        "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
        "mse": float(mse),
        "directional_accuracy": float(directional_accuracy),
        "sharpe": float(sharpe_net),
        "sharpe_gross": float(sharpe_gross),
        "mean_return": float(np.mean(net_returns.flatten())),
        "mean_return_gross": float(np.mean(gross_returns.flatten())),
        "std_return": float(np.std(net_returns.flatten())),
        "avg_daily_cost_bps": float(np.mean(trading_cost.flatten()) * 10000.0),
        "trading_cost_bps_assumed": cost_bps,
        "avg_sheaf_energy": float(np.mean(energies)),
        "avg_reg_slope": float(np.mean(slopes)),
        "avg_reg_r2": float(np.mean(r2s)),
        "energy_series": [round(float(e), 4) for e in energies],
    }


def _confidence_from_r2(r2: float) -> str:
    if r2 > 0.01:
        return "High"
    elif r2 > 0.002:
        return "Medium"
    return "Low"


def compute_ticker_picks(V: np.ndarray, node_names: List[str], ticker_indices: List[int],
                          tickers: List[str], window: int, sheaf_config: Dict,
                          top_n: int) -> Tuple[List[Dict], Dict, Dict]:
    """
    Fit a sheaf model on the most recent `window` rows and return the top-N
    ETF picks by predicted next-day return, plus the full per-ticker result
    dict and the run's diagnostics.
    """
    cfg = sheaf_config.copy()
    cfg["window"] = window

    try:
        result = get_sheaf_predictions(V, node_names, ticker_indices, cfg)
    except Exception as e:
        logger.error(f"  Sheaf fit failed (window={window}): {e}")
        return [], {}, {}

    pred_returns = result["next_returns"]
    gaps = result["gaps"]
    model: SheafModel = result["model"]
    confidence = _confidence_from_r2(result["reg_r2"])

    ticker_results = {}
    for idx, ticker in enumerate(tickers):
        node_idx = ticker_indices[idx]
        nbrs = sorted(model.neighbors.get(node_idx, []), key=lambda x: -abs(x[1]))[:3]
        related = [{"node": node_names[n], "rho": round(rho, 3)} for n, rho in nbrs]
        ticker_results[ticker] = {
            "next_return": float(pred_returns[idx]),
            "gap": float(gaps[idx]),
            "related": related,
        }

    sorted_picks = sorted(ticker_results.items(), key=lambda x: x[1]["next_return"], reverse=True)
    top_picks = sorted_picks[:top_n]

    picks = []
    for ticker, info in top_picks:
        picks.append({
            "ticker": ticker,
            "expected_return": round(info["next_return"] * 100, 2),
            "gap": round(info["gap"], 3),
            "confidence": confidence,
            "related": info["related"],
        })

    diagnostics = {
        "n_edges": result["n_edges"],
        "avg_degree": round(result["avg_degree"], 2),
        "reg_slope": round(result["reg_slope"], 4),
        "reg_r2": round(result["reg_r2"], 5),
        "sheaf_energy": round(result["sheaf_energy"], 4),
        "top_edges": [
            {"a": a, "b": b, "rho": round(rho, 3)}
            for a, b, rho in model.get_top_edges(10)
        ],
    }

    return picks, ticker_results, diagnostics


def run_trainer() -> Dict:
    """Main Sheaf-Alpha trainer with multi-window backtesting."""

    logger.info("🔄 Loading data...")
    try:
        prices_df, macro_df = load_master_data()
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {}

    macro_names = list(macro_df.columns) if macro_df is not None else []
    use_macro_globally = config.SHEAF_CONFIG.get("use_macro", True) and len(macro_names) > 0
    if use_macro_globally:
        logger.info(f"🕸️  Sheaf will include macro nodes: {macro_names}")
    else:
        logger.info("🕸️  Macro nodes disabled or unavailable.")

    run_date = datetime.now().strftime("%Y-%m-%d")
    results = {
        "run_date": run_date,
        "algorithm": "SHEAF-ALPHA",
        "top_picks": {},
        "backtest_results": {},
        "best_window": {},
        "universes": {},
        "window_picks": {},
        "diagnostics": {},
    }

    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Analyzing {universe_name}...")

        available = [t for t in tickers if t in prices_df.columns]
        if not available:
            continue

        universe_prices_df = prices_df[available]
        valid_mask = ~universe_prices_df.isna().any(axis=1)
        universe_prices_df = universe_prices_df[valid_mask]

        if len(universe_prices_df) < 200:
            logger.warning(f"Not enough data for {universe_name}")
            continue

        returns = np.diff(np.log(universe_prices_df.values), axis=0)
        node_names = list(available)

        if use_macro_globally:
            universe_macro_df = macro_df.loc[universe_prices_df.index]
            macro_diff = np.diff(universe_macro_df.values, axis=0)
            V = np.hstack([returns, macro_diff])
            node_names = node_names + macro_names
        else:
            V = returns

        ticker_indices = list(range(len(available)))
        sheaf_config = config.SHEAF_CONFIG.copy()
        sheaf_config["trading_cost_bps"] = config.TRADING_COST_BPS

        # Backtest each window
        window_results = {}
        for window in config.WINDOWS:
            logger.info(f"  Testing window {window}...")
            result = backtest_window(V, ticker_indices, window, sheaf_config)

            if "error" not in result:
                window_results[window] = result
                logger.info(f"    Correlation: {result['correlation']:.3f}, "
                           f"Directional: {result['directional_accuracy']:.2%}, "
                           f"Sharpe (net of {config.TRADING_COST_BPS}bps costs): {result['sharpe']:.2f} "
                           f"(gross: {result['sharpe_gross']:.2f}), "
                           f"Sheaf-R²: {result['avg_reg_r2']:.4f}")
            else:
                logger.warning(f"    {result['error']}")

        # Best window is selected by RETURN-PREDICTION quality (correlation
        # between predicted and actual returns), not by backtested Sharpe.
        # Sharpe reflects realized P&L, which can look good even when a
        # window's predictions barely explain anything -- e.g. a window
        # whose predictions are nearly flat can still post a decent Sharpe
        # just by riding the test period's market drift. Correlation
        # directly measures whether the predictions themselves are good.
        select_metric = config.BEST_WINDOW_METRIC
        if window_results:
            best_window = max(window_results.items(), key=lambda x: x[1].get(select_metric, -999))
            results["best_window"][universe_name] = {
                "window": best_window[0],
                "metrics": best_window[1],
                "selected_by": select_metric,
            }
            logger.info(f"  ✅ Best window for {universe_name}: {best_window[0]} "
                       f"(selected by {select_metric}={best_window[1][select_metric]:.4f}; "
                       f"Sharpe: {best_window[1]['sharpe']:.2f})")

        results["backtest_results"][universe_name] = window_results

        best_win = results["best_window"].get(universe_name, {}).get("window", 252)

        results["window_picks"][universe_name] = {}
        results["diagnostics"][universe_name] = {}
        best_win_ticker_results = {}
        best_win_diag = {}

        for window in config.WINDOWS:
            picks, ticker_results, diag = compute_ticker_picks(
                V, node_names, ticker_indices, available, window, sheaf_config, config.TOP_N
            )
            results["window_picks"][universe_name][window] = picks
            results["diagnostics"][universe_name][window] = diag
            if window == best_win:
                best_win_ticker_results = ticker_results
                best_win_diag = diag

        picks = results["window_picks"][universe_name].get(best_win, [])
        if not best_win_ticker_results:
            picks, best_win_ticker_results, best_win_diag = compute_ticker_picks(
                V, node_names, ticker_indices, available, best_win, sheaf_config, config.TOP_N
            )

        results["top_picks"][universe_name] = picks
        results["universes"][universe_name] = {
            "tickers": available,
            "best_window": best_win,
            "ticker_results": best_win_ticker_results,
            "diagnostics": best_win_diag,
        }

        logger.info(f"  ✅ Top picks for {universe_name}:")
        for pick in picks:
            logger.info(f"     {pick['ticker']}: {pick['expected_return']}% (gap={pick['gap']}, {pick['confidence']})")

    output_path = f"sheaf_results_{run_date}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n💾 Saved: {output_path}")

    try:
        from push_results import upload_results
        upload_results(output_path, hf_token=config.HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not upload results: {e}")

    return results


if __name__ == "__main__":
    run_trainer()
