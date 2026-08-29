"""
sheaf_model.py  —  Sheaf-Alpha: financial inconsistency alpha via cellular sheaves

Theory
------
A graph says "these nodes are connected." A cellular sheaf says more: each
node carries a *local observation* (a stalk), and each edge carries a
*restriction map* describing how the two endpoints' observations should
relate if the system is internally consistent. When the observed values
violate that relation, that violation is measurable, localizable, and — if
markets take time to arbitrage away dislocations — potentially predictive.

This module builds a lightweight cellular sheaf over ETF returns and macro
changes:

  Nodes     Each ETF's daily log-return, plus each macro series' daily
            change (VIX, T10Y2Y, DXY, credit spreads, ...). All node values
            are standardized (z-scored) within the training window, so every
            node lives on a comparable "one-day shock" scale regardless of
            its native units.

  Edges     For each node, connect it to its k most-correlated other nodes
            (over the training window), above a minimum correlation
            threshold. This keeps the graph sparse and keeps only
            relationships strong enough to be a meaningful constraint,
            rather than a fully-connected graph of mostly-noise edges.

  Restriction maps
            For an edge (u, v) with standardized node values, the natural
            linear consistency map is z_v ≈ rho_uv * z_u, where rho_uv is
            the training-window correlation between u and v (for
            standardized variables this is exactly the OLS slope in either
            direction, so the same rho works symmetrically as the
            restriction map from u's stalk or from v's stalk onto the
            shared edge stalk).

  Sheaf disagreement (the "financial inconsistency" signal)
            For node u, its neighbors' current values each imply a value
            for u (rho_uv * z_v). Averaging those implied values (weighted
            by |rho_uv|, so stronger relationships carry more say) gives a
            "sheaf-consensus" estimate of what u's standardized move should
            be right now. The GAP between that consensus and u's actual
            realized value is the local section disagreement — exactly what
            a cellular sheaf's coboundary operator measures, restricted to
            node u's neighborhood.

  Alpha     Whether disagreement at node u predicts u's NEXT return (mean
            reversion / catch-up) is an empirical question, not an
            assumption. It's tested by pooling (gap, next-return) pairs
            across every ticker and every day in the training window and
            fitting a single linear regression. This also naturally
            reports whether the learned relationship is a genuine edge
            (non-trivial slope) or effectively noise.

Aggregate sheaf energy (a market-dislocation index, not directly used for
the return forecast but reported alongside it): the sum of squared edge
disagreements across the whole graph on a given day is the sheaf Laplacian
quadratic form of that day's observation, x^T L x — a standard measure of
how far the day's data is from being a consistent "global section."
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class SheafModel:
    def __init__(self, k_neighbors: int = 5, min_abs_corr: float = 0.15):
        self.k_neighbors = k_neighbors
        self.min_abs_corr = min_abs_corr

        self.node_names: List[str] = []
        self.n_nodes = 0
        self.train_mean_: Optional[np.ndarray] = None
        self.train_std_: Optional[np.ndarray] = None
        self.edges: List[Tuple[int, int, float]] = []       # (i, j, rho_ij)
        self.neighbors: Dict[int, List[Tuple[int, float]]] = {}  # node -> [(neighbor, rho)]
        self.reg_intercept_: float = 0.0
        self.reg_slope_: float = 0.0
        self.reg_r2_: float = 0.0
        self.gap_std_: float = 1.0

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #
    def _standardize(self, V: np.ndarray) -> np.ndarray:
        mean = V.mean(axis=0)
        std = V.std(axis=0)
        std[std < 1e-10] = 1e-10
        self.train_mean_ = mean
        self.train_std_ = std
        return (V - mean) / std

    def _apply_standardize(self, V: np.ndarray) -> np.ndarray:
        return (V - self.train_mean_) / self.train_std_

    def _build_graph(self, Z: np.ndarray) -> None:
        """Top-k correlation graph from standardized training data Z (n_samples, n_nodes)."""
        n_nodes = Z.shape[1]
        corr = np.corrcoef(Z, rowvar=False)
        np.fill_diagonal(corr, 0.0)
        corr = np.nan_to_num(corr, nan=0.0)

        self.neighbors = {i: [] for i in range(n_nodes)}
        edge_set = {}

        for i in range(n_nodes):
            row = corr[i].copy()
            row[np.abs(row) < self.min_abs_corr] = 0.0
            order = np.argsort(-np.abs(row))
            picked = 0
            for j in order:
                if picked >= self.k_neighbors:
                    break
                if row[j] == 0.0:
                    break
                self.neighbors[i].append((int(j), float(row[j])))
                key = (min(i, j), max(i, j))
                edge_set[key] = float(row[j])
                picked += 1

        self.edges = [(i, j, rho) for (i, j), rho in edge_set.items()]

    # ------------------------------------------------------------------ #
    # Sheaf disagreement ("gap") computation
    # ------------------------------------------------------------------ #
    def _compute_gaps(self, Z: np.ndarray, node_indices: List[int]) -> np.ndarray:
        """
        For each requested node index and each row (time) in Z, compute the
        neighbor-implied consensus value minus the node's actual value.

        Returns array of shape (n_samples, len(node_indices)).
        """
        n_samples = Z.shape[0]
        gaps = np.zeros((n_samples, len(node_indices)))

        for col, u in enumerate(node_indices):
            nbrs = self.neighbors.get(u, [])
            if not nbrs:
                continue  # isolated node -> no consensus available, gap stays 0
            weights = np.array([abs(rho) for _, rho in nbrs])
            w_sum = weights.sum()
            if w_sum < 1e-12:
                continue
            implied = np.zeros(n_samples)
            for (v, rho) in nbrs:
                implied += abs(rho) * (rho * Z[:, v])
            implied /= w_sum
            gaps[:, col] = implied - Z[:, u]

        return gaps

    def sheaf_energy(self, Z_row: np.ndarray) -> float:
        """
        x^T L x for a single day's standardized observation Z_row (n_nodes,):
        the total squared disagreement across every edge in the graph. A
        higher value means the day's data is further from a consistent
        "global section" -- i.e. more internal market disagreement.
        """
        energy = 0.0
        for (i, j, rho) in self.edges:
            d = Z_row[j] - rho * Z_row[i]
            energy += d ** 2
        return float(energy)

    # ------------------------------------------------------------------ #
    # Fit / predict
    # ------------------------------------------------------------------ #
    def fit(self, V_train: np.ndarray, node_names: List[str],
            ticker_indices: List[int]) -> Dict:
        """
        V_train: (n_samples, n_nodes) raw node values (ticker returns +
                 macro changes) for the training window.
        ticker_indices: indices (into node_names / V_train columns) of the
                 tradeable ETF nodes -- the ones we want gaps/predictions for.
        """
        self.node_names = node_names
        self.n_nodes = V_train.shape[1]

        Z_train = self._standardize(V_train)
        self._build_graph(Z_train)

        gaps_train = self._compute_gaps(Z_train, ticker_indices)  # (n_samples, n_tickers)

        # Pool (gap_u(t), standardized next-return_u(t+1)) across every
        # ticker and every day, then fit ONE global regression. Pooling
        # gives far more data points than fitting per-ticker on a single
        # window's worth of days, which makes the fitted relationship much
        # more statistically stable.
        X_pool = gaps_train[:-1, :].flatten()
        Y_pool = Z_train[1:, ticker_indices].flatten()

        gap_std = X_pool.std()
        self.gap_std_ = gap_std if gap_std > 1e-10 else 1.0
        X_pool_std = X_pool / self.gap_std_

        A = np.column_stack([np.ones_like(X_pool_std), X_pool_std])
        coef, *_ = np.linalg.lstsq(A, Y_pool, rcond=None)
        self.reg_intercept_, self.reg_slope_ = float(coef[0]), float(coef[1])

        fitted = A @ coef
        ss_res = np.sum((Y_pool - fitted) ** 2)
        ss_tot = np.sum((Y_pool - Y_pool.mean()) ** 2)
        self.reg_r2_ = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

        n_edges = len(self.edges)
        avg_degree = float(np.mean([len(v) for v in self.neighbors.values()])) if self.neighbors else 0.0

        return {
            "n_edges": n_edges,
            "avg_degree": avg_degree,
            "reg_slope": self.reg_slope_,
            "reg_r2": self.reg_r2_,
            "n_pooled_samples": len(X_pool),
        }

    def predict(self, V_current_row: np.ndarray, ticker_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        V_current_row: (n_nodes,) most recent raw node values.
        Returns (predicted_returns, gaps) both length len(ticker_indices),
        predicted_returns in RAW return units (un-standardized).
        """
        Z_row = self._apply_standardize(V_current_row.reshape(1, -1))[0]
        gaps = self._compute_gaps(Z_row.reshape(1, -1), ticker_indices)[0]  # (n_tickers,)

        gaps_std = gaps / self.gap_std_
        pred_std = self.reg_intercept_ + self.reg_slope_ * gaps_std  # standardized next-return

        mean_t = self.train_mean_[ticker_indices]
        std_t = self.train_std_[ticker_indices]
        pred_raw = pred_std * std_t + mean_t

        return pred_raw, gaps

    def get_top_edges(self, n: int = 10) -> List[Tuple[str, str, float]]:
        """Strongest edges by |rho|, as (name_i, name_j, rho)."""
        sorted_edges = sorted(self.edges, key=lambda e: -abs(e[2]))[:n]
        return [(self.node_names[i], self.node_names[j], rho) for i, j, rho in sorted_edges]


def get_sheaf_predictions(V: np.ndarray, node_names: List[str], ticker_indices: List[int],
                           config: Dict) -> Dict:
    """
    Fit a SheafModel on the most recent `window` rows of V and produce
    1-step-ahead return forecasts + diagnostics for every ticker node.
    """
    window = config.get("window", 252)
    if len(V) > window:
        V_train = V[-window:]
    else:
        V_train = V

    if len(V_train) < config.get("min_train_samples", 30):
        raise ValueError("Not enough data to fit a sheaf model.")

    model = SheafModel(
        k_neighbors=config.get("k_neighbors", 5),
        min_abs_corr=config.get("min_abs_corr", 0.15),
    )
    fit_result = model.fit(V_train, node_names, ticker_indices)

    V_current = V_train[-1]
    pred_returns, gaps = model.predict(V_current, ticker_indices)

    Z_current = model._apply_standardize(V_current.reshape(1, -1))[0]
    energy = model.sheaf_energy(Z_current)

    return {
        "next_returns": pred_returns,        # (n_tickers,)
        "gaps": gaps,                        # (n_tickers,) standardized disagreement
        "sheaf_energy": energy,
        "reg_slope": fit_result["reg_slope"],
        "reg_r2": fit_result["reg_r2"],
        "n_edges": fit_result["n_edges"],
        "avg_degree": fit_result["avg_degree"],
        "model": model,
    }
