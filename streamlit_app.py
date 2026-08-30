import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import glob
import os
import networkx as nx
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(
    page_title="P2 Sheaf-Alpha",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------
PRIMARY = "#7c3aed"      # violet -- distinct identity from the SINDy dashboard's blue
POSITIVE = "#16a34a"
NEGATIVE = "#dc2626"
NEUTRAL = "#d97706"
INK = "#0f172a"
SUBTLE = "#64748b"
CARD_BG = "#ffffff"
CARD_BORDER = "#e2e8f0"
PAGE_BG = "#f8fafc"
MACRO_COLOR = "#0891b2"
TICKER_COLOR = "#7c3aed"

CONF_COLORS = {"high": POSITIVE, "medium": NEUTRAL, "low": SUBTLE}

st.markdown(f"""
<style>
    .stApp {{ background-color: {PAGE_BG}; }}
    #MainMenu, footer {{visibility: hidden;}}

    .app-header {{ display: flex; align-items: baseline; gap: 0.75rem; margin-bottom: 0; }}
    .app-title {{ font-size: 1.9rem; font-weight: 800; color: {INK}; margin: 0; }}
    .app-subtitle {{ color: {SUBTLE}; font-size: 0.95rem; margin-top: 0.15rem; margin-bottom: 1.25rem; }}

    .universe-heading {{ font-size: 1.15rem; font-weight: 700; color: {INK}; margin: 0 0 0.15rem 0; }}
    .universe-caption {{ color: {SUBTLE}; font-size: 0.85rem; margin-bottom: 0.75rem; }}

    .pick-card {{
        background: {CARD_BG}; border: 1px solid {CARD_BORDER}; border-left: 4px solid var(--accent);
        border-radius: 12px; padding: 1.1rem 1.3rem; margin: 0.35rem 0;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .pick-ticker {{ font-size: 1.05rem; font-weight: 700; color: {INK}; letter-spacing: 0.02em; }}
    .pick-return {{ font-size: 1.9rem; font-weight: 800; color: {INK}; margin: 0.25rem 0 0.35rem 0; line-height: 1.1; }}
    .pick-badge {{
        display: inline-block; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.04em; padding: 0.15rem 0.55rem; border-radius: 999px; color: white; background: var(--accent);
    }}
    .pick-gap {{ font-size: 0.78rem; color: {SUBTLE}; margin-top: 0.55rem; }}
    .pick-related {{ margin-top: 0.4rem; }}
    .related-tag {{
        display: inline-block; font-size: 0.7rem; color: {INK}; background: #f1f5f9;
        border: 1px solid {CARD_BORDER}; border-radius: 999px; padding: 0.1rem 0.5rem; margin: 0.1rem 0.2rem 0 0;
    }}

    .kpi-card {{ background: {CARD_BG}; border: 1px solid {CARD_BORDER}; border-radius: 12px; padding: 0.9rem 1.1rem; text-align: center; }}
    .kpi-value {{ font-size: 1.4rem; font-weight: 800; color: {INK}; }}
    .kpi-label {{ font-size: 0.72rem; color: {SUBTLE}; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 0.15rem; }}

    .best-window-banner {{
        background: linear-gradient(90deg, #f5f3ff 0%, #faf5ff 100%); border: 1px solid #ddd6fe;
        border-left: 4px solid {PRIMARY}; border-radius: 10px; padding: 0.7rem 1rem;
        font-size: 0.9rem; color: #4c1d95; margin: 0.5rem 0 1rem 0;
    }}
    .theory-note {{
        background: #f8fafc; border: 1px dashed {CARD_BORDER}; border-radius: 10px;
        padding: 0.75rem 1rem; font-size: 0.82rem; color: {SUBTLE}; margin: 0.75rem 0 1.25rem 0;
    }}

    div[data-testid="stDataFrame"] {{ border: 1px solid {CARD_BORDER}; border-radius: 10px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
RESULTS_REPO = "P2SAMAPA/p2-sheaf-alpha-results"


def _find_latest_hf_result(repo_id: str):
    """List actual files in the HF dataset and return the newest results
    file by name (YYYY-MM-DD sorts correctly lexically), instead of
    guessing that today's date matches when the training run happened."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_files(repo_id, repo_type="dataset")
        result_files = sorted(f for f in files if f.startswith("sheaf_results_") and f.endswith(".json"))
        if not result_files:
            return None, f"No sheaf_results_*.json files found in {repo_id}."
        return result_files[-1], None
    except Exception as e:
        return None, f"Could not list files in {repo_id}: {e}"


@st.cache_data(show_spinner="Loading results...")
def load_data(_cache_key: str = ""):
    """
    Load the latest results, trying in order:
      1. A local sheaf_results_*.json in the working directory (fast path
         right after running trainer.py locally).
      2. The most recent sheaf_results_*.json actually present in the HF
         results dataset (not a guess at "today's" date -- training may
         have run on a different day than this dashboard is being viewed).
    Returns (data, notes) where notes is a list of human-readable strings
    describing what was tried, so a failure is diagnosable instead of a
    silent blank dashboard.
    """
    notes = []

    json_files = glob.glob("sheaf_results_*.json")
    if json_files:
        latest_local = sorted(json_files)[-1]
        try:
            with open(latest_local, "r") as f:
                return json.load(f), [f"Loaded local file: {latest_local}"]
        except Exception as e:
            notes.append(f"Found local file {latest_local} but couldn't parse it: {e}")

    latest_remote, list_err = _find_latest_hf_result(RESULTS_REPO)
    if list_err:
        notes.append(list_err)
    else:
        try:
            url = f"https://huggingface.co/datasets/{RESULTS_REPO}/resolve/main/{latest_remote}"
            headers = {}
            token = os.environ.get("HF_TOKEN")
            if token:
                headers["Authorization"] = f"Bearer {token}"
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                return response.json(), [f"Loaded from HF dataset: {latest_remote}"]
            notes.append(f"Fetching {latest_remote} from HF returned HTTP {response.status_code} "
                         f"(private dataset without a valid HF_TOKEN would show up this way).")
        except Exception as e:
            notes.append(f"Fetching {latest_remote} from HF failed: {e}")

    return None, notes


def conf_color(confidence: str) -> str:
    return CONF_COLORS.get((confidence or "low").lower(), SUBTLE)


def render_pick_cards(picks, key_prefix):
    if not picks:
        st.info("No ETF picks available for this selection.")
        return

    cols = st.columns(min(len(picks), 3))
    for i, pick in enumerate(picks):
        color = conf_color(pick["confidence"])
        gap = pick.get("gap", 0)
        gap_arrow = "▲" if gap > 0 else ("▼" if gap < 0 else "—")
        related = pick.get("related", [])
        related_html = "".join(
            f'<span class="related-tag">{r["node"]} (ρ={r["rho"]:+.2f})</span>' for r in related
        )
        with cols[i % len(cols)]:
            st.markdown(f"""
            <div class="pick-card" style="--accent: {color};">
                <div class="pick-ticker">{pick['ticker']}</div>
                <div class="pick-return">{pick['expected_return']:+.2f}%</div>
                <span class="pick-badge">{pick['confidence']} confidence</span>
                <div class="pick-gap">Sheaf disagreement {gap_arrow} {abs(gap):.2f}σ</div>
                <div class="pick-related">{related_html}</div>
            </div>
            """, unsafe_allow_html=True)


def render_network_graph(top_edges, ticker_set, key):
    """A real sheaf network diagram: nodes = tickers/macro, edges = restriction map strength."""
    if not top_edges:
        st.info("No sheaf edges to display for this selection.")
        return

    G = nx.Graph()
    for e in top_edges:
        G.add_edge(e["a"], e["b"], rho=e["rho"])

    pos = nx.spring_layout(G, seed=7, k=1.1 / max(len(G.nodes()) ** 0.5, 0.1))

    edge_traces = []
    for u, v, d in G.edges(data=True):
        rho = d["rho"]
        color = POSITIVE if rho > 0 else NEGATIVE
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(width=1 + 3 * abs(rho), color=color),
            opacity=0.55, hoverinfo="text",
            text=f"{u} ↔ {v}: ρ = {rho:+.2f}",
            showlegend=False,
        ))

    node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        is_ticker = node in ticker_set
        node_color.append(TICKER_COLOR if is_ticker else MACRO_COLOR)
        node_size.append(22 if is_ticker else 16)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_text, textposition="top center",
        textfont=dict(size=11, color=INK),
        marker=dict(size=node_size, color=node_color, line=dict(width=2, color="white")),
        hoverinfo="text", showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"network_{key}")
    st.markdown(
        f'<div style="font-size:0.78rem; color:{SUBTLE};">'
        f'<span style="color:{TICKER_COLOR};">●</span> ETF &nbsp;&nbsp;'
        f'<span style="color:{MACRO_COLOR};">●</span> Macro &nbsp;&nbsp;'
        f'<span style="color:{POSITIVE};">—</span> Positive restriction map &nbsp;&nbsp;'
        f'<span style="color:{NEGATIVE};">—</span> Negative restriction map'
        f'</div>', unsafe_allow_html=True
    )


def render_energy_chart(energy_series, key):
    """Sheaf inconsistency energy over the backtest test period -- a market-dislocation index."""
    if not energy_series:
        return
    x = list(range(len(energy_series)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=energy_series, mode="lines", fill="tozeroy",
        line=dict(color=PRIMARY, width=1.5), fillcolor="rgba(124, 58, 237, 0.12)",
        hovertemplate="Day %{x}: energy %{y:.2f}<extra></extra>",
    ))
    avg = float(np.mean(energy_series))
    fig.add_hline(y=avg, line_dash="dot", line_color=SUBTLE, annotation_text=f"avg {avg:.2f}", annotation_font_size=10)
    fig.update_layout(
        height=220, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Test-period trading day", gridcolor="#eef2f7"),
        yaxis=dict(title="Sheaf energy", gridcolor="#eef2f7"),
        font=dict(color=INK, size=11),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"energy_{key}")


def render_window_comparison(df_results, best_idx, key):
    """Net Sharpe + directional accuracy across window sizes -- a line chart, not a bar chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_results["Window"], y=df_results["Net Sharpe"],
        mode="lines+markers", name="Net Sharpe (after costs)",
        line=dict(color=PRIMARY, width=2.5), marker=dict(size=9),
    ))
    fig.add_trace(go.Scatter(
        x=df_results["Window"], y=df_results["Directional Accuracy"],
        mode="lines+markers", name="Directional Accuracy %",
        line=dict(color=POSITIVE, width=2.5, dash="dot"), marker=dict(size=9),
        yaxis="y2",
    ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Window Size (days)", gridcolor="#eef2f7"),
        yaxis=dict(title="Net Sharpe Ratio", gridcolor="#eef2f7"),
        yaxis2=dict(title="Directional Accuracy (%)", overlaying="y", side="right"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(color=INK, size=12),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"windowcmp_{key}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    if "cache_bust" not in st.session_state:
        st.session_state.cache_bust = 0

    header_col, refresh_col = st.columns([5, 1])
    with header_col:
        st.markdown('<div class="app-header"><span style="font-size:1.9rem;">🕸️</span>'
                     '<span class="app-title">P2 Sheaf-Alpha</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subtitle">Financial inconsistency alpha via cellular sheaves over ETF & macro relationships</div>',
                    unsafe_allow_html=True)
    with refresh_col:
        st.markdown("<div style='margin-top: 0.6rem;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh data", use_container_width=True,
                      help="Clear the cache and re-check for the latest results"):
            st.cache_data.clear()
            st.session_state.cache_bust += 1
            st.rerun()

    data, notes = load_data(_cache_key=str(st.session_state.cache_bust))

    if not data:
        st.error("No results found.")
        with st.expander("Why? (click for details)", expanded=True):
            if notes:
                for n in notes:
                    st.write(f"• {n}")
            else:
                st.write("• No diagnostic information was returned.")
            st.write(
                "Run `python trainer.py` to generate a local `sheaf_results_*.json`, "
                "or confirm the HF dataset repo has a results file and (if private) "
                "that `HF_TOKEN` is set in this app's environment. Then hit **🔄 Refresh data** above."
            )
        return

    run_date = data.get("run_date", "Unknown")
    st.caption(f"🕒 Results generated: **{run_date}**")
    if notes:
        st.caption(f"📡 {notes[0]}")

    tab1, tab2 = st.tabs(["🔮 Live Signal", "🕸️ Sheaf Diagnostics & Backtest"])

    # ------------------------------------------------------------------ #
    # TAB 1 — Live Signal
    # ------------------------------------------------------------------ #
    with tab1:
        st.markdown("""
        <div class="theory-note">
        A cellular sheaf ties each ETF's return to its most correlated peers and macro
        variables via fitted <i>restriction maps</i>. When an ETF's actual move disagrees
        with what its neighborhood implies, that gap is the raw signal below — tested
        empirically for whether it tends to resolve into next-day return.
        </div>
        """, unsafe_allow_html=True)

        top_picks = data.get("top_picks", {})
        best_window = data.get("best_window", {})

        if not top_picks:
            st.warning("No top-pick data available yet.")

        for universe, picks in top_picks.items():
            st.markdown(f'<div class="universe-heading">{universe.replace("_", " ").title()}</div>',
                        unsafe_allow_html=True)

            best = best_window.get(universe, {})
            if best:
                metrics = best.get("metrics", {})
                st.markdown(f"""
                <div class="best-window-banner">
                    ✅ Best training window: <b>{best.get('window', 'N/A')} days</b>
                    <span style="opacity:0.7;">(selected by return-prediction correlation, not Sharpe)</span>
                    &nbsp;|&nbsp; Correlation: <b>{metrics.get('correlation', 0):.4f}</b>
                    &nbsp;|&nbsp; Directional accuracy: <b>{metrics.get('directional_accuracy', 0):.1%}</b>
                    &nbsp;|&nbsp; Sheaf R²: <b>{metrics.get('avg_reg_r2', 0):.4f}</b>
                    <br/>
                    Net Sharpe (after {metrics.get('trading_cost_bps_assumed', 15)}bps trading cost):
                    <b>{metrics.get('sharpe', 0):.2f}</b>
                    <span style="opacity:0.7;">(gross, before costs: {metrics.get('sharpe_gross', 0):.2f})</span>
                </div>
                """, unsafe_allow_html=True)

            render_pick_cards(picks, key_prefix=f"picks_{universe}")
            st.markdown("<div style='margin: 0.5rem 0 1.5rem 0;'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------ #
    # TAB 2 — Sheaf Diagnostics & Backtest
    # ------------------------------------------------------------------ #
    with tab2:
        backtest_results = data.get("backtest_results", {})
        window_picks = data.get("window_picks", {})
        diagnostics = data.get("diagnostics", {})

        if not backtest_results:
            st.warning("No backtest data available yet.")

        for universe, window_results in backtest_results.items():
            st.markdown(f'<div class="universe-heading">{universe.replace("_", " ").title()}</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="universe-caption">Sheaf structure, disagreement diagnostics, and performance across training window sizes</div>',
                        unsafe_allow_html=True)

            if not window_results:
                st.warning("No backtest results available for this universe.")
                continue

            df_results = pd.DataFrame([
                {
                    "Window": int(w),
                    "Correlation": r.get("correlation", 0),
                    "Directional Accuracy": r.get("directional_accuracy", 0) * 100,
                    "Net Sharpe": r.get("sharpe", 0),
                    "Gross Sharpe": r.get("sharpe_gross", 0),
                    "Sheaf R²": r.get("avg_reg_r2", 0),
                    "Avg Cost (bps/day)": r.get("avg_daily_cost_bps", 0),
                    "Predictions": r.get("n_predictions", 0),
                }
                for w, r in window_results.items()
            ]).sort_values("Window").reset_index(drop=True)

            # Selected by return-prediction correlation (see config.BEST_WINDOW_METRIC),
            # not by Sharpe -- Sharpe reflects realized P&L, which can look
            # good from a window whose predictions barely explain anything.
            best_idx = df_results["Correlation"].idxmax()
            best_row = df_results.loc[best_idx]
            best_window_val = str(int(best_row["Window"]))
            cost_bps = window_results.get(best_window_val, window_results.get(int(best_window_val), {})).get("trading_cost_bps_assumed", 15)

            kpi_cols = st.columns(5)
            kpi_data = [
                ("Best Window", f"{int(best_row['Window'])}d"),
                ("Correlation", f"{best_row['Correlation']:.4f}"),
                ("Directional Acc.", f"{best_row['Directional Accuracy']:.1f}%"),
                ("Net Sharpe", f"{best_row['Net Sharpe']:.2f}"),
                ("Sheaf R²", f"{best_row['Sheaf R²']:.4f}"),
            ]
            for col, (label, value) in zip(kpi_cols, kpi_data):
                with col:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-value">{value}</div>
                        <div class="kpi-label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.caption(f"💸 Net Sharpe/returns above assume a {cost_bps}bps trading cost per position "
                       f"change (turnover), not a flat per-day charge — a position held unchanged "
                       f"day-to-day costs nothing extra. Best window highlighted below is chosen by "
                       f"return-prediction correlation, not by Sharpe.")

            st.markdown("<div style='margin-top: 0.9rem;'></div>", unsafe_allow_html=True)

            st.dataframe(
                df_results.style.apply(
                    lambda x: ["background-color: #ede9fe" if x.name == best_idx else "" for _ in x],
                    axis=1,
                ).format({
                    "Correlation": "{:.4f}",
                    "Directional Accuracy": "{:.1f}%",
                    "Net Sharpe": "{:.2f}",
                    "Gross Sharpe": "{:.2f}",
                    "Sheaf R²": "{:.4f}",
                    "Avg Cost (bps/day)": "{:.2f}",
                    "Predictions": "{:,.0f}",
                }),
                use_container_width=True,
                hide_index=True,
                column_config={"Window": "Window (days)"},
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("###### Performance across windows")
                render_window_comparison(df_results, best_idx, key=f"{universe}")
            with col_b:
                st.markdown("###### Sheaf inconsistency energy (best window, test period)")
                energy_series = window_results.get(int(best_window_val), {}).get("energy_series", [])
                if not energy_series:
                    energy_series = window_results.get(best_window_val, {}).get("energy_series", [])
                render_energy_chart(energy_series, key=f"{universe}")

            st.markdown("###### Sheaf network — strongest restriction maps (best window)")
            uni_diag = diagnostics.get(universe, {})
            best_diag = uni_diag.get(best_window_val) or uni_diag.get(int(best_window_val)) or {}
            ticker_set = set(data.get("universes", {}).get(universe, {}).get("tickers", []))
            render_network_graph(best_diag.get("top_edges", []), ticker_set, key=f"{universe}")

            st.markdown("###### ETF picks by window")
            universe_window_picks = window_picks.get(universe, {})
            if not universe_window_picks:
                st.info("No per-window ETF picks in this results file.")
            else:
                available_windows = sorted(universe_window_picks.keys(), key=lambda w: int(w))
                window_tabs = st.tabs([f"{w}d" for w in available_windows])
                for wtab, w in zip(window_tabs, available_windows):
                    with wtab:
                        render_pick_cards(universe_window_picks[w], key_prefix=f"wpicks_{universe}_{w}")

            st.markdown("<hr style='margin: 1.75rem 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
