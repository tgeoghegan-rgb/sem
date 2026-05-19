"""
SEM Driver Analysis Tool — Prototype v1

Single-file Streamlit app for driver importance analysis using
Johnson's (2000) relative weights regression on Likert survey data.

Future upgrade notes are marked NOTE(future) throughout.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SEM Driver Analysis",
    page_icon="📊",
    layout="wide",
)


# ── Synthetic sample data ──────────────────────────────────────────────────────

@st.cache_data
def generate_sample_data(n: int = 150, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic Likert survey data with realistic inter-driver correlations.
    10 drivers + 1 outcome (all 1–10 scale) + 2 demographic columns.
    Only the first four drivers have non-zero true relationships with the
    outcome, so importance scores vary meaningfully across drivers.
    """
    rng = np.random.default_rng(seed)

    # Shared factor creates moderate inter-driver correlation (~0.30)
    base_corr = np.full((10, 10), 0.30)
    np.fill_diagonal(base_corr, 1.0)
    L = np.linalg.cholesky(base_corr)
    raw = rng.standard_normal((n, 10)) @ L.T

    # Clip latent scores to 1–10 Likert scale
    X = np.clip(np.round(raw * 1.8 + 5.5), 1, 10).astype(int)

    # Outcome = weighted combination of first 4 drivers + noise
    w = np.array([0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.01, 0.01, 0.00, 0.00])
    y_lat = raw @ w + rng.standard_normal(n) * 0.6
    lo, hi = y_lat.min(), y_lat.max()
    y = np.clip(np.round((y_lat - lo) / (hi - lo) * 9 + 1), 1, 10).astype(int)

    df = pd.DataFrame(X, columns=[f"driver_{i+1:02d}" for i in range(10)])
    df["overall_satisfaction"] = y
    df["region"] = rng.choice(["North", "South", "East", "West"], size=n)
    df["age_band"] = rng.choice(
        ["18–24", "25–34", "35–44", "45–54", "55+"], size=n,
        p=[0.10, 0.25, 0.30, 0.20, 0.15],
    )
    return df


# ── Relative weights regression ────────────────────────────────────────────────

def relative_weights_regression(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Johnson's (2000) relative weights analysis.

    Partitions R² among correlated predictors so each driver receives credit
    for its unique contribution plus a proportional share of shared variance.

    Algorithm:
      1. Standardise X (predictors) and y (outcome).
      2. Eigendecompose the predictor correlation matrix: R_XX = A diag(λ) A'.
      3. Form orthogonal unit-variance predictors: Z = X_std A diag(λ^-½).
         (Z'Z = (n-1)·I, so columns are uncorrelated with unit sample variance.)
      4. OLS of y_std on Z → β_Z.  Since Z is orthogonal: R² = Σ_k β_Z_k².
      5. Relative weight for predictor j: ε_j = Σ_k A_jk² · β_Z_k².
         The weights sum to R², so ε_j / R² × 100 gives the % share.
      6. Standardised betas and p-values come from plain OLS on X_std
         (betas give direction; relative weights give importance magnitude).

    NOTE(future): Replace this function with full SEM via semopy.
    The interface (X ndarray, y ndarray → dict) is kept stable for that swap.

    Returns
    -------
    r_squared              float   — proportion of variance explained
    relative_weights_pct   ndarray — each driver's % share of R² (sums to 100)
    std_betas              ndarray — standardised OLS regression coefficients
    p_values               ndarray — two-tailed p-values for each std_beta
    """
    n, p = X.shape

    # Standardise; ddof=1 ensures y'y = n-1 (needed for R² identity below)
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)
    ys = (y - y.mean()) / y.std(ddof=1)

    # ── Eigendecompose predictor correlation matrix ──
    R_XX = np.corrcoef(Xs.T)
    eigenvalues, eigenvectors = np.linalg.eigh(R_XX)   # real & stable for symmetric A
    eigenvalues = np.maximum(eigenvalues, 1e-10)        # numerical floor

    # ── Orthogonal predictors Z ──
    Z = Xs @ eigenvectors @ np.diag(eigenvalues ** -0.5)

    # ── OLS β_Z; R² = Σ β_Z_k² (proven: SS_reg = (n-1)·β'β, SS_tot = n-1) ──
    beta_Z, _, _, _ = np.linalg.lstsq(Z, ys, rcond=None)
    r_squared = float(np.clip(np.sum(beta_Z ** 2), 0.0, 1.0))

    # ── Relative weights ε_j = Σ_k A_jk² β_Z_k² ──
    # Matrix form: (eigenvectors**2) @ (beta_Z**2) gives the p-vector in one line.
    raw_weights = (eigenvectors ** 2) @ (beta_Z ** 2)
    total_rw = raw_weights.sum()
    rw_pct = (raw_weights / total_rw * 100.0) if total_rw > 1e-10 else np.full(p, 100.0 / p)

    # ── Standardised betas + p-values (plain OLS on Xs) ──
    ols = LinearRegression(fit_intercept=False).fit(Xs, ys)
    std_betas = ols.coef_

    y_hat = Xs @ std_betas
    residuals = ys - y_hat
    df_resid = max(n - p - 1, 1)
    mse = np.sum(residuals ** 2) / df_resid
    try:
        XtX_inv = np.linalg.inv(Xs.T @ Xs)
        se = np.sqrt(mse * np.diag(XtX_inv))
        t_stat = np.where(se > 0, std_betas / se, 0.0)
        p_values = 2.0 * stats.t.sf(np.abs(t_stat), df=df_resid)
    except np.linalg.LinAlgError:
        p_values = np.ones(p)

    return {
        "r_squared": r_squared,
        "relative_weights_pct": rw_pct,
        "std_betas": std_betas,
        "p_values": p_values,
    }


# ── Significance helpers ───────────────────────────────────────────────────────

def sig_flag(p: float) -> str:
    if p < 0.05:
        return "✓"
    if p < 0.10:
        return "~"
    return "✗"


_ROW_COLORS = {"✓": "#d4edda", "~": "#fff3cd", "✗": "#f8d7da"}


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("SEM Driver Analysis Tool")
    st.caption("Prototype v1 — Relative weights regression on Likert survey data")

    tab_analysis, tab_guide = st.tabs(["📊 Analysis", "📖 User Guide"])

    with tab_guide:
        with open("user_guide.html", "r", encoding="utf-8") as fh:
            guide_html = fh.read()
        components.html(guide_html, height=3000, scrolling=True)

    with tab_analysis:
        # ── Data loading ──────────────────────────────────────────────────────
        sample_df = generate_sample_data()

        uploaded = st.file_uploader(
            "Upload your own CSV (optional — replaces built-in sample data)", type="csv"
        )
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded).dropna(how="all")
                st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")
            except Exception as exc:
                st.error(f"Could not parse CSV: {exc}")
                df = sample_df
        else:
            df = sample_df
            st.info("Using built-in sample dataset (150 respondents, 10 drivers, 1 outcome).")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        text_cols = df.select_dtypes(exclude="number").columns.tolist()

        if len(numeric_cols) < 2:
            st.error("At least 2 numeric columns required (1 outcome + ≥1 driver).")
            return

        # ── Sidebar ───────────────────────────────────────────────────────────
        # NOTE(future): Upgrade this sidebar to a drag-and-drop node-link canvas
        # (path diagram) so analysts can visually wire up a structural model —
        # e.g. using streamlit-agraph or a custom React component.
        with st.sidebar:
            st.header("Network Editor")

            # Segment filters from text columns (auto-detected on upload)
            active_df = df.copy()
            if text_cols:
                st.subheader("Segment filters")
                for col in text_cols:
                    opts = ["All"] + sorted(df[col].dropna().unique().tolist())
                    choice = st.selectbox(col, opts, key=f"seg_{col}")
                    if choice != "All":
                        active_df = active_df[active_df[col] == choice]
                st.caption(f"n = {len(active_df):,} respondents in view")
                st.divider()

            # Outcome selector — defaults to last numeric column
            st.subheader("Outcome variable")
            outcome_col = st.selectbox(
                "outcome",
                numeric_cols,
                index=len(numeric_cols) - 1,
                label_visibility="collapsed",
                key="outcome_col",
            )

            # Driver checkboxes — all checked by default
            st.subheader("Driver variables")
            candidate_drivers = [c for c in numeric_cols if c != outcome_col]
            selected_drivers: list = []
            for col in candidate_drivers:
                if st.checkbox(col, value=True, key=f"drv_{col}"):
                    selected_drivers.append(col)

            n_active = len(selected_drivers)
            n_total = len(candidate_drivers)
            st.caption(f"{n_active} of {n_total} drivers → {outcome_col}")
            st.divider()

            run_btn = st.button("Run analysis", type="primary", use_container_width=True)

        # ── Session state — results persist across widget interactions ────────
        if "results" not in st.session_state:
            st.session_state.results = None
        if "run_config" not in st.session_state:
            st.session_state.run_config = None

        # ── Trigger model only on button click ────────────────────────────────
        if run_btn:
            if not selected_drivers:
                st.error("Select at least one driver variable.")
            elif len(active_df) < 5:
                st.error("Too few rows to fit a model after filtering.")
            else:
                model_df = active_df[selected_drivers + [outcome_col]].dropna()
                X = model_df[selected_drivers].values.astype(float)
                y = model_df[outcome_col].values.astype(float)

                # Guard: zero-variance drivers cause division by zero in standardisation
                zero_var = [d for d, v in zip(selected_drivers, X.var(axis=0)) if v < 1e-10]
                if zero_var:
                    st.error(
                        f"Driver(s) with zero variance (all respondents gave the same rating): "
                        f"{zero_var}. Deselect them and retry."
                    )
                else:
                    with st.spinner("Running relative weights regression…"):
                        results = relative_weights_regression(X, y)

                    st.session_state.results = results
                    st.session_state.run_config = {
                        "outcome": outcome_col,
                        "drivers": selected_drivers,
                        "n": len(model_df),
                        "driver_means": model_df[selected_drivers].mean().to_dict(),
                    }

        # ── Display outputs (all gated on having run at least once) ───────────
        if st.session_state.results is None:
            st.info("Configure the network in the sidebar, then click **Run analysis**.")
            return

        res = st.session_state.results
        cfg = st.session_state.run_config
        drivers = cfg["drivers"]
        outcome = cfg["outcome"]
        n_resp = cfg["n"]
        r2 = res["r_squared"]
        rw_pct = res["relative_weights_pct"]
        std_betas = res["std_betas"]
        p_vals = res["p_values"]

        # ── R² metric card ────────────────────────────────────────────────────
        m_col, t_col = st.columns([1, 4])
        with m_col:
            st.metric("Model R²", f"{r2:.3f}")
        with t_col:
            pct_label = f"{round(r2 * 100)}%"
            st.markdown(f"**The model explains {pct_label} of variation in *{outcome}*.**")
            if r2 < 0.10:
                st.caption("Low explanatory power — consider adding or refining drivers.")
            elif r2 < 0.30:
                st.caption("Moderate explanatory power.")
            else:
                st.caption("Good explanatory power.")

        st.divider()

        # ── Driver importance table ───────────────────────────────────────────
        st.subheader("Driver importance")

        flags = [sig_flag(p) for p in p_vals]
        table_df = (
            pd.DataFrame(
                {
                    "Driver": drivers,
                    "Importance (%)": np.round(rw_pct, 1),
                    "Std. beta": np.round(std_betas, 3),
                    "p-value": np.round(p_vals, 3),
                    "Sig.": flags,
                }
            )
            .sort_values("Importance (%)", ascending=False)
            .reset_index(drop=True)
        )

        def _color_row(row: pd.Series) -> list:
            bg = _ROW_COLORS[row["Sig."]]
            return [f"background-color: {bg}"] * len(row)

        styled = (
            table_df.style
            .apply(_color_row, axis=1)
            .format({"Importance (%)": "{:.1f}", "Std. beta": "{:.3f}", "p-value": "{:.3f}"})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Sample adequacy note
        threshold = 10 * len(drivers)
        if n_resp < threshold:
            st.warning(
                f"Sample adequacy: n = {n_resp} may be small for {len(drivers)} drivers "
                f"(rule of thumb: n ≥ {threshold}). Interpret results cautiously."
            )
        else:
            st.caption(
                f"Sample adequacy: n = {n_resp} meets the n ≥ {threshold} guideline "
                f"for {len(drivers)} drivers ✓"
            )

        st.caption("Significance: ✓ p < 0.05 (green) · ~ p < 0.10 (amber) · ✗ p > 0.10 (red)")

        st.divider()

        # ── Charts ────────────────────────────────────────────────────────────
        left, right = st.columns(2)

        # Bubble chart: importance (y) vs. mean score / performance (x)
        with left:
            st.subheader("Importance vs. Performance")
            means = [cfg["driver_means"][d] for d in drivers]
            # Sqrt-scale bubble sizes to prevent extreme outliers dominating visually
            bubble_sizes = list(np.sqrt(np.maximum(rw_pct, 0)) * 5 + 8)

            fig_bubble = go.Figure(
                go.Scatter(
                    x=means,
                    y=list(rw_pct),
                    mode="markers+text",
                    text=drivers,
                    textposition="top center",
                    marker=dict(
                        size=bubble_sizes,
                        color=list(rw_pct),
                        colorscale="Blues",
                        showscale=True,
                        colorbar=dict(title="Importance %"),
                        line=dict(width=1, color="white"),
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Mean score: %{x:.2f}<br>"
                        "Importance: %{y:.1f}%<extra></extra>"
                    ),
                )
            )
            fig_bubble.update_layout(
                xaxis_title="Mean score (performance)",
                yaxis_title="Relative importance (%)",
                height=420,
                margin=dict(t=20, b=40, l=50, r=20),
            )
            st.plotly_chart(fig_bubble, use_container_width=True)

        # Horizontal bar chart: ranked by relative importance
        with right:
            st.subheader("Driver importance ranking")
            # Sort ascending so the highest-importance driver appears at the top of the chart
            ranked = table_df.sort_values("Importance (%)")
            fig_bar = go.Figure(
                go.Bar(
                    x=ranked["Importance (%)"],
                    y=ranked["Driver"],
                    orientation="h",
                    marker_color="#4C78A8",
                    text=[f"{v:.1f}%" for v in ranked["Importance (%)"]],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
                )
            )
            fig_bar.update_layout(
                title=dict(text=f"Outcome: {outcome}", font=dict(size=13), x=0),
                xaxis_title="Relative importance (%)",
                yaxis_title="",
                height=420,
                margin=dict(t=40, b=40, l=20, r=80),
            )
            st.plotly_chart(fig_bar, use_container_width=True)


main()
