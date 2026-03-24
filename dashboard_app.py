from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_echarts import st_echarts
    ECHARTS_AVAILABLE = True
except Exception:
    ECHARTS_AVAILABLE = False

try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    AGGRID_AVAILABLE = True
except Exception:
    AGGRID_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

st.set_page_config(
    page_title="Canada Electricity Transition Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "electricity-capacity-dataset.csv"

RENEWABLE = {"Hydro", "Wind", "Solar", "Biomass"}
MIX_COLORS = {"Renewable": "#2563eb", "Non-Renewable": "#dc2626"}


@st.cache_data
def load_base(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["Region", "Source", "Year", "Data"]).copy()
    df["Region"] = df["Region"].astype(str).str.strip()
    df["Source"] = df["Source"].astype(str).str.strip()
    df["Year"] = df["Year"].astype(int)
    df["Data"] = df["Data"].astype(float)
    return df.sort_values(["Region", "Source", "Year"]).reset_index(drop=True)


def fig_style(fig: go.Figure, *, height: int = 360, showlegend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        showlegend=showlegend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        hoverlabel={"namelength": -1},
    )
    return fig


def render_echarts(option: dict, *, height: int = 360, key: str | None = None) -> None:
    if ECHARTS_AVAILABLE:
        st_echarts(options=option, height=f"{height}px", key=key)
    else:
        st.info("Install streamlit-echarts to enable high-performance ECharts visuals.")


def render_aggrid(df: pd.DataFrame, *, key: str, height: int = 280) -> None:
    if AGGRID_AVAILABLE:
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(sortable=True, filter=True, resizable=True)
        gb.configure_grid_options(domLayout="normal")
        grid_options = gb.build()
        AgGrid(df, gridOptions=grid_options, height=height, key=key, fit_columns_on_grid_load=True)
    else:
        st.info("Install streamlit-aggrid to enable AG Grid tables.")
        st.dataframe(df, width="stretch", hide_index=True)


def model_commentary(metrics: pd.DataFrame, series: pd.DataFrame, region: str, source: str) -> str:
    ranked = metrics.sort_values("RMSE").reset_index(drop=True)
    if ranked.empty:
        return "No model diagnostics are available for this selection."

    best = ranked.iloc[0]
    second = ranked.iloc[1] if len(ranked) > 1 else ranked.iloc[0]
    gap = float(second["RMSE"] - best["RMSE"])
    rel_gap = float((gap / second["RMSE"] * 100.0) if second["RMSE"] > 0 else 0.0)

    s = series.sort_values("Year")["Data"].to_numpy(dtype=float)
    n = len(s)
    vol = float(np.std(s) / np.mean(s)) if n >= 2 and np.mean(s) > 0 else 0.0
    lag1 = float(np.corrcoef(s[1:], s[:-1])[0, 1]) if n >= 3 and np.std(s[1:]) > 1e-9 and np.std(s[:-1]) > 1e-9 else 0.0

    if rel_gap < 3:
        strength = "Model lead is narrow, so this is a close contest."
    elif rel_gap < 10:
        strength = "Model lead is moderate and defensible."
    else:
        strength = "Model lead is strong and clearly separated."

    if best["Model"] == "NaiveLag1":
        why = "Persistence dominates this source, so last-value information carries most predictive power."
    elif best["Model"] == "Blend_Naive_Drift_SES":
        why = "A pure time-series blend wins here, combining persistence, drift, and smoothing effects."
    elif best["Model"] == "LinearTrend":
        why = "Trend and lag structure are dominant, favoring a parsimonious linear model."
    elif best["Model"] == "Blend_Naive_Linear_RF":
        why = "A three-model blend wins here, combining persistence, trend structure, and nonlinear effects."
    else:
        why = "Nonlinear interactions appear meaningful for this source, favoring ensemble methods."

    data_note = (
        "Short sample + low volatility can suppress gains from complex models."
        if n < 15 or vol < 0.12
        else "Higher variation can increase the value of nonlinear learners."
    )

    return (
        f"For {region} - {source}, best model is {best['Model']} (RMSE {best['RMSE']:.2f}, MAE {best['MAE']:.2f}). "
        f"Runner-up is {second['Model']} with RMSE gap {gap:.2f} ({rel_gap:.1f}%). {strength} {why} "
        f"Lag-1 correlation is {lag1:.2f}; {data_note}"
    )


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 1e-9
    if not np.any(mask):
        return 0.0
    return float(np.mean(200.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def fit_model(name: str, x_train: np.ndarray, y_train: np.ndarray) -> tuple[object | None, dict]:
    if name == "NaiveLag1":
        return None, {}
    if name == "LinearTrend":
        model = LinearRegression().fit(x_train, y_train)
        return model, {}
    if name == "RandomForest":
        base = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid = {"n_estimators": [180, 260, 340], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 2]}
    else:
        raise ValueError(f"Unsupported tuned model: {name}")

    cv_splits = 2 if len(y_train) < 12 else 3
    cv = TimeSeriesSplit(n_splits=cv_splits)
    gs = GridSearchCV(base, grid, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1, refit=True)
    gs.fit(x_train, y_train)
    return gs.best_estimator_, gs.best_params_


def fit_ses_alpha(y_train: np.ndarray) -> float:
    if len(y_train) < 3:
        return 0.4
    alphas = [0.2, 0.35, 0.5, 0.65, 0.8]
    best_alpha = 0.4
    best_err = float("inf")
    for alpha in alphas:
        level = float(y_train[0])
        errs = []
        for i in range(1, len(y_train)):
            pred = level
            errs.append((float(y_train[i]) - pred) ** 2)
            level = alpha * float(y_train[i]) + (1.0 - alpha) * level
        mse = float(np.mean(errs)) if errs else float("inf")
        if mse < best_err:
            best_err = mse
            best_alpha = alpha
    return float(best_alpha)


def ses_next(y_train: np.ndarray, alpha: float) -> float:
    level = float(y_train[0])
    for i in range(1, len(y_train)):
        level = alpha * float(y_train[i]) + (1.0 - alpha) * level
    return float(level)


def drift_next(y_train: np.ndarray) -> float:
    if len(y_train) < 2:
        return float(y_train[-1])
    drift = (float(y_train[-1]) - float(y_train[0])) / max(1, len(y_train) - 1)
    return float(y_train[-1]) + drift


def build_model_from_params(name: str, params: dict) -> object | None:
    if name == "NaiveLag1":
        return None
    if name == "LinearTrend":
        return LinearRegression()
    if name == "RandomForest":
        return RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    raise ValueError(f"Unsupported model type: {name}")


def micro_forecast(series: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = series[["Year", "Data"]].sort_values("Year").copy()
    s["lag1"] = s["Data"].shift(1)
    s["lag2"] = s["Data"].shift(2)
    s["d1"] = s["Data"].diff().shift(1)
    s = s.dropna().reset_index(drop=True)

    if len(s) < 8:
        return (
            pd.DataFrame(columns=["Year", "Model", "Forecast_MW", "Lower80_MW", "Upper80_MW"]),
            pd.DataFrame(columns=["Model", "MAE", "RMSE", "sMAPE", "BacktestPoints", "BestParams"]),
        )

    x = s[["Year", "lag1", "lag2", "d1"]].to_numpy(dtype=float)
    y = s["Data"].to_numpy(dtype=float)
    candidates = [
        "NaiveLag1",
        "DriftTrend_TS",
        "SES_TS",
        "Blend_Naive_Drift_SES",
        "LinearTrend",
        "Blend_Naive_Linear_RF",
        "RandomForest",
    ]

    rows_eval: list[dict] = []
    rows_fc: list[dict] = []
    min_train = max(6, min(12, len(s) - 2))

    tuned_params: dict[str, dict] = {
        "NaiveLag1": {},
        "DriftTrend_TS": {},
        "SES_TS": {},
        "Blend_Naive_Drift_SES": {},
        "LinearTrend": {},
        "Blend_Naive_Linear_RF": {},
        "RandomForest": {},
    }
    init_x = x[:min_train]
    init_y = y[:min_train]
    for name in candidates:
        if name in {"RandomForest"}:
            _, tuned = fit_model(name, init_x, init_y)
            tuned_params[name] = tuned

    for name in candidates:
        y_true_bt: list[float] = []
        y_pred_bt: list[float] = []

        for t in range(min_train, len(s)):
            x_train = x[:t]
            y_train = y[:t]
            x_next = x[t:t + 1]
            y_next = y[t]

            if name == "NaiveLag1":
                pred_next = float(x_next[0, 1])
            elif name == "DriftTrend_TS":
                pred_next = drift_next(y_train)
            elif name == "SES_TS":
                alpha_bt = fit_ses_alpha(y_train)
                pred_next = ses_next(y_train, alpha_bt)
            elif name == "Blend_Naive_Drift_SES":
                alpha_bt = fit_ses_alpha(y_train)
                pred_next = (
                    float(x_next[0, 1])
                    + drift_next(y_train)
                    + ses_next(y_train, alpha_bt)
                ) / 3.0
            elif name == "Blend_Naive_Linear_RF":
                blend_lin = LinearRegression().fit(x_train, y_train)
                pred_lin = float(blend_lin.predict(x_next)[0])
                blend_rf = build_model_from_params("RandomForest", tuned_params.get("RandomForest", {}))
                blend_rf.fit(x_train, y_train)
                pred_rf = float(blend_rf.predict(x_next)[0])
                pred_next = (float(x_next[0, 1]) + pred_lin + pred_rf) / 3.0
            else:
                model_bt = build_model_from_params(name, tuned_params.get(name, {}))
                model_bt.fit(x_train, y_train)
                pred_next = float(model_bt.predict(x_next)[0])

            y_true_bt.append(float(y_next))
            y_pred_bt.append(max(0.0, pred_next))

        y_true_arr = np.array(y_true_bt, dtype=float)
        y_pred_arr = np.array(y_pred_bt, dtype=float)
        mae = float(np.mean(np.abs(y_true_arr - y_pred_arr))) if len(y_true_arr) else np.nan
        rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))) if len(y_true_arr) else np.nan
        smape_val = smape(y_true_arr, y_pred_arr) if len(y_true_arr) else np.nan
        resid = float(np.std(y_true_arr - y_pred_arr)) if len(y_true_arr) > 1 else max(1.0, mae)

        if name == "NaiveLag1":
            model = None
            best_params = {}
        elif name == "DriftTrend_TS":
            model = None
            best_params = {"trend": "linear drift from first to last"}
        elif name == "SES_TS":
            alpha_full = fit_ses_alpha(y)
            model = {"alpha": alpha_full}
            best_params = {"alpha": alpha_full}
        elif name == "Blend_Naive_Drift_SES":
            alpha_full = fit_ses_alpha(y)
            model = {"alpha": alpha_full}
            best_params = {"blend": "(naive + drift + ses) / 3", "alpha": alpha_full}
        elif name == "Blend_Naive_Linear_RF":
            model = {
                "linear": LinearRegression().fit(x, y),
                "rf": build_model_from_params("RandomForest", tuned_params.get("RandomForest", {})),
            }
            model["rf"].fit(x, y)
            best_params = {
                "blend": "(naive + linear + random_forest) / 3",
                "rf_params": tuned_params.get("RandomForest", {}),
            }
        else:
            best_params = tuned_params.get(name, {})
            model = build_model_from_params(name, best_params)
            model.fit(x, y)

        rows_eval.append(
            {
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "sMAPE": smape_val,
                "BacktestPoints": int(len(y_true_arr)),
                "BestParams": str(best_params),
            }
        )

        vals = series["Data"].astype(float).tolist()
        last_year = int(series["Year"].max())
        for h in range(1, horizon + 1):
            y_next = last_year + h
            lag1 = vals[-1]
            lag2 = vals[-2] if len(vals) > 1 else vals[-1]
            d1 = vals[-1] - vals[-2] if len(vals) > 1 else 0.0
            if name == "NaiveLag1":
                y_hat = float(max(0.0, lag1))
            elif name == "DriftTrend_TS":
                if len(vals) < 2:
                    y_hat = float(max(0.0, vals[-1]))
                else:
                    drift = (float(vals[-1]) - float(vals[0])) / max(1, len(vals) - 1)
                    y_hat = float(max(0.0, vals[-1] + drift))
            elif name == "SES_TS":
                alpha = float(model["alpha"])
                level = float(vals[0])
                for vi in vals[1:]:
                    level = alpha * float(vi) + (1.0 - alpha) * level
                y_hat = float(max(0.0, level))
            elif name == "Blend_Naive_Drift_SES":
                alpha = float(model["alpha"])
                level = float(vals[0])
                for vi in vals[1:]:
                    level = alpha * float(vi) + (1.0 - alpha) * level
                if len(vals) < 2:
                    drift_part = float(vals[-1])
                else:
                    drift_part = float(vals[-1] + (vals[-1] - vals[0]) / max(1, len(vals) - 1))
                y_hat = float(max(0.0, (float(lag1) + drift_part + level) / 3.0))
            elif name == "Blend_Naive_Linear_RF":
                x_next = np.array([[float(y_next), float(lag1), float(lag2), float(d1)]], dtype=float)
                y_lin = float(model["linear"].predict(x_next)[0])
                y_rf = float(model["rf"].predict(x_next)[0])
                y_hat = float(max(0.0, (lag1 + y_lin + y_rf) / 3.0))
            else:
                x_next = np.array([[float(y_next), float(lag1), float(lag2), float(d1)]], dtype=float)
                y_hat = float(max(0.0, model.predict(x_next)[0]))
            band = 1.28 * resid * (1.0 + 0.08 * (h - 1))
            rows_fc.append(
                {
                    "Year": y_next,
                    "Model": name,
                    "Forecast_MW": y_hat,
                    "Lower80_MW": max(0.0, y_hat - band),
                    "Upper80_MW": y_hat + band,
                }
            )
            vals.append(y_hat)

    met = pd.DataFrame(rows_eval)
    met = met.replace([np.inf, -np.inf], np.nan).dropna(subset=["MAE", "RMSE"]).reset_index(drop=True)
    return pd.DataFrame(rows_fc), met


def classify_signal(delta_pct: float) -> str:
    if delta_pct >= 20:
        return "High Growth"
    if delta_pct >= 8:
        return "Moderate Growth"
    if delta_pct <= -20:
        return "Sharp Decline"
    if delta_pct <= -8:
        return "Moderate Decline"
    return "Stable / Flat"


def summarize_prediction_signal(
    series: pd.DataFrame,
    fc: pd.DataFrame,
    met: pd.DataFrame,
    horizon: int,
) -> dict:
    if series.empty or fc.empty or met.empty:
        return {}

    best_model = str(met.sort_values(["RMSE", "MAE"]).iloc[0]["Model"])
    path = fc[fc["Model"] == best_model].sort_values("Year")
    if path.empty:
        return {}

    y0 = float(series["Data"].iloc[-1])
    yT = float(path["Forecast_MW"].iloc[-1])
    lowerT = float(path["Lower80_MW"].iloc[-1])
    upperT = float(path["Upper80_MW"].iloc[-1])

    delta = yT - y0
    delta_pct = (delta / y0 * 100.0) if y0 > 1e-9 else 0.0
    cagr = ((yT / y0) ** (1.0 / max(1, horizon)) - 1.0) * 100.0 if y0 > 1e-9 and yT > 1e-9 else 0.0
    uncert_ratio = ((upperT - lowerT) / max(yT, 1e-9)) * 100.0

    return {
        "BestModel": best_model,
        "CurrentMW": y0,
        "HorizonMW": yT,
        "DeltaMW": delta,
        "DeltaPct": delta_pct,
        "CAGR": cagr,
        "UncertaintyPct": uncert_ratio,
        "Signal": classify_signal(delta_pct),
    }


@st.cache_data(show_spinner=False)
def build_region_source_outlook(
    base_df: pd.DataFrame,
    region: str,
    sources: tuple[str, ...],
    horizon: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for src in sources:
        s = base_df[(base_df["Region"] == region) & (base_df["Source"] == src)][["Year", "Data"]].sort_values("Year")
        if len(s) < 10:
            continue
        fc, met = micro_forecast(s, horizon=horizon)
        summ = summarize_prediction_signal(s, fc, met, horizon)
        if not summ:
            continue
        rows.append(
            {
                "Source": src,
                "BestModel": summ["BestModel"],
                "CurrentMW": summ["CurrentMW"],
                f"Forecast_{horizon}y_MW": summ["HorizonMW"],
                "DeltaMW": summ["DeltaMW"],
                "DeltaPct": summ["DeltaPct"],
                "CAGR_pct": summ["CAGR"],
                "UncertaintyPct": summ["UncertaintyPct"],
                "Signal": summ["Signal"],
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(["DeltaPct", "Forecast_{}y_MW".format(horizon)], ascending=[False, False]).reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False)
def build_live_reliability(
    base_df: pd.DataFrame,
    region: str,
    sources: tuple[str, ...],
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_source_rows: list[dict] = []
    for src in sources:
        s = base_df[(base_df["Region"] == region) & (base_df["Source"] == src)][["Year", "Data"]].sort_values("Year")
        if len(s) < 10:
            continue
        _, met = micro_forecast(s, horizon=horizon)
        if met.empty:
            continue
        top = met.sort_values(["RMSE", "MAE"]).iloc[0]
        per_source_rows.append(
            {
                "Source": src,
                "BestModel": str(top["Model"]),
                "RMSE": float(top["RMSE"]),
                "MAE": float(top["MAE"]),
                "sMAPE": float(top["sMAPE"]),
                "BacktestPoints": int(top["BacktestPoints"]),
            }
        )

    if not per_source_rows:
        return pd.DataFrame(), pd.DataFrame()

    per_source = pd.DataFrame(per_source_rows)
    by_model = per_source.groupby("BestModel", as_index=False).agg(
        WinCount=("BestModel", "count"),
        MeanRMSE=("RMSE", "mean"),
        MeanMAE=("MAE", "mean"),
        MeansMAPE=("sMAPE", "mean"),
    ).sort_values(["WinCount", "MeanRMSE"], ascending=[False, True]).reset_index(drop=True)
    return per_source, by_model


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
    :root {
        --ink-900: #0b1324;
        --ink-700: #1f3349;
        --line: rgba(15, 23, 42, 0.10);
        --card: rgba(255, 255, 255, 0.78);
        --accent: #1d4ed8;
    }
    .stApp {
        background:
            radial-gradient(1000px 450px at 8% -10%, rgba(37,99,235,.10), transparent 65%),
            radial-gradient(900px 450px at 95% 0%, rgba(220,38,38,.08), transparent 65%),
            linear-gradient(180deg, #f8fbff 0%, #f4f8fb 100%);
    }
    h1,h2,h3,p,li,span,label {
        font-family: 'Manrope', sans-serif;
        letter-spacing: 0.01em;
    }
    h2 {
        font-weight: 800;
        color: var(--ink-900);
    }
    h3 {
        font-weight: 700;
        color: var(--ink-700);
        margin-top: 0.1rem;
    }
    .hero {
        border-radius: 18px;
        padding: 20px 22px;
        background: linear-gradient(135deg,#0b1324 0%,#123a54 55%,#1d4ed8 100%);
        color: #e5eefb;
        border: 1px solid rgba(15,23,42,.08);
        box-shadow: 0 10px 24px rgba(15,23,42,.14);
        margin-bottom: 14px;
        animation: rise-in 420ms ease-out;
    }
    .section-lead {
        border: 1px solid var(--line);
        background: var(--card);
        backdrop-filter: blur(4px);
        border-radius: 14px;
        padding: 10px 14px;
        margin: 6px 0 10px 0;
        color: var(--ink-700);
        font-size: 0.95rem;
    }
    [data-testid="stMetric"] {
        border: 1px solid var(--line);
        background: var(--card);
        border-radius: 14px;
        padding: 8px 10px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    }
    [data-testid="stMetricLabel"] {
        font-weight: 700;
        color: #334155;
        font-size: 0.82rem;
        line-height: 1.2;
        white-space: normal;
        overflow-wrap: anywhere;
    }
    [data-testid="stMetricValue"] {
        font-weight: 800;
        color: #0f172a;
        font-size: 1.45rem;
        line-height: 1.15;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 10px;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.65);
        font-weight: 700;
    }
    .stTabs [aria-selected="true"] {
        background: #e8efff;
        border-color: rgba(29, 78, 216, 0.38);
    }
    section[data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
        padding-top: 1rem;
    }
    @keyframes rise-in {
        from { transform: translateY(6px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

base = load_base(DATA_PATH)

all_years = sorted(base["Year"].unique())
all_regions = sorted([r for r in base["Region"].unique() if r != "Canada"])
all_sources = sorted(base["Source"].unique())

with st.sidebar:
    st.subheader("Scenario Controls")
    if st.button("Refresh", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    year_min, year_max = st.slider("Historical window", min(all_years), max(all_years), (min(all_years), max(all_years)))
    selected_regions = st.multiselect("Regions", all_regions, default=all_regions)
    selected_sources = st.multiselect("Sources", all_sources, default=all_sources)
    auto_play = st.toggle("Auto-play transition animation", value=True)
    anim_speed = st.slider("Animation speed (seconds)", min_value=0.2, max_value=1.0, value=0.45, step=0.05)

    st.markdown("---")
    st.subheader("Forecast Controls")
    lab_region = st.selectbox("Region", ["Canada"] + all_regions, index=0)
    lab_source = st.selectbox("Source", all_sources, index=0)
    lab_horizon = st.slider("Forecast horizon", 3, 10, 6)
    prediction_view = st.radio("Prediction table", ["Best model only", "All models (comparison)"], index=0)

if not selected_regions:
    selected_regions = all_regions
if not selected_sources:
    selected_sources = all_sources

flt = base[
    base["Year"].between(year_min, year_max)
    & base["Region"].isin(selected_regions + ["Canada"])
    & base["Source"].isin(selected_sources)
].copy()

can = flt[flt["Region"] == "Canada"].groupby("Year", as_index=False)["Data"].sum()
latest_year = int(can["Year"].max())
latest_total = float(can[can["Year"] == latest_year]["Data"].sum())
renew_latest = float(
    flt[(flt["Region"] == "Canada") & (flt["Year"] == latest_year) & (flt["Source"].isin(RENEWABLE))]["Data"].sum()
)
renew_share_latest = float(renew_latest / latest_total) if latest_total > 0 else 0.0

st.markdown(
    """
<div class='hero'>
    <h2 style='margin:0;'>Canada Electricity Transition Intelligence Dashboard</h2>
    <p style='margin:6px 0 0 0;'>Question: where is capacity shifting next, and which regions need action first?</p>
</div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Dataset rows", f"{len(base):,}")
m2.metric("Year range", f"{base['Year'].min()}-{base['Year'].max()}")
m3.metric("Latest Canada capacity", f"{latest_total:,.0f} MW")
m4.metric("Latest renewable share", f"{renew_share_latest:.1%}")
st.caption("Guideline flow: evidence from EDA -> transition diagnosis -> model selection and tuning -> forecast validation -> policy decision support.")

region_energy = flt[flt["Region"].isin(selected_regions)].copy()
region_energy["EnergyType"] = np.where(region_energy["Source"].isin(RENEWABLE), "Renewable", "Non-Renewable")
region_energy = region_energy.groupby(["Year", "Region", "EnergyType"], as_index=False)["Data"].sum()
region_piv = region_energy.pivot(index=["Year", "Region"], columns="EnergyType", values="Data").fillna(0.0).reset_index()
if "Renewable" not in region_piv.columns:
    region_piv["Renewable"] = 0.0
if "Non-Renewable" not in region_piv.columns:
    region_piv["Non-Renewable"] = 0.0
region_piv["RenewableShare"] = np.where(
    (region_piv["Renewable"] + region_piv["Non-Renewable"]) > 0,
    100.0 * region_piv["Renewable"] / (region_piv["Renewable"] + region_piv["Non-Renewable"]),
    0.0,
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1) Evidence Base",
        "2) Transition Diagnostics",
        "3) Forecast Lab",
        "4) Reliability & Policy",
    ]
)

with tab1:
    st.markdown(
        "<div class='section-lead'><strong>Evidence goal:</strong> establish baseline structure of regional mix and source-share shifts before modeling decisions.</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Regional Mix Snapshot (100% Stacked)")
        st.caption("Compares renewable-heavy regions against regions still dominated by non-renewables.")
        cat = flt[(flt["Region"].isin(selected_regions)) & (flt["Year"] == latest_year)].copy()
        cat["EnergyType"] = np.where(cat["Source"].isin(RENEWABLE), "Renewable", "Non-Renewable")
        mix = cat.groupby(["Region", "EnergyType"], as_index=False)["Data"].sum()
        totals = mix.groupby("Region", as_index=False)["Data"].sum().rename(columns={"Data": "Tot"})
        mix = mix.merge(totals, on="Region", how="left")
        mix["SharePct"] = np.where(mix["Tot"] > 0, 100.0 * mix["Data"] / mix["Tot"], 0.0)
        order = (
            mix[mix["EnergyType"] == "Renewable"]
            .sort_values("SharePct", ascending=False)["Region"]
            .tolist()
        )
        if ECHARTS_AVAILABLE:
            ren_map = mix[mix["EnergyType"] == "Renewable"].set_index("Region")["SharePct"].to_dict()
            non_map = mix[mix["EnergyType"] == "Non-Renewable"].set_index("Region")["SharePct"].to_dict()
            option_mix = {
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "legend": {"top": 0},
                "grid": {"left": 40, "right": 20, "top": 45, "bottom": 30, "containLabel": True},
                "xAxis": {"type": "category", "data": order, "name": f"Region ({latest_year})"},
                "yAxis": {"type": "value", "name": "Share (%)", "max": 100},
                "series": [
                    {
                        "name": "Renewable",
                        "type": "bar",
                        "stack": "total",
                        "itemStyle": {"color": MIX_COLORS["Renewable"]},
                        "data": [round(float(ren_map.get(r, 0.0)), 2) for r in order],
                    },
                    {
                        "name": "Non-Renewable",
                        "type": "bar",
                        "stack": "total",
                        "itemStyle": {"color": MIX_COLORS["Non-Renewable"]},
                        "data": [round(float(non_map.get(r, 0.0)), 2) for r in order],
                    },
                ],
            }
            render_echarts(option_mix, height=390, key="ech_mix")
        else:
            fig_mix = px.bar(
                mix,
                x="Region",
                y="SharePct",
                color="EnergyType",
                barmode="stack",
                color_discrete_map=MIX_COLORS,
                category_orders={"Region": order, "EnergyType": ["Renewable", "Non-Renewable"]},
                labels={"SharePct": "Share (%)"},
            )
            fig_mix.update_yaxes(range=[0, 100])
            fig_mix.update_layout(xaxis_title=f"Region ({latest_year})")
            st.plotly_chart(fig_style(fig_mix, height=390, showlegend=True))

    with c2:
        st.subheader("Source Shift: Start vs Latest")
        st.caption("Shows which generation technologies gained or lost share over the historical window.")
        start_year = int(flt["Year"].min())
        end_year = int(flt["Year"].max())
        a = flt[(flt["Region"] == "Canada") & (flt["Year"] == start_year)].groupby("Source", as_index=False)["Data"].sum()
        b = flt[(flt["Region"] == "Canada") & (flt["Year"] == end_year)].groupby("Source", as_index=False)["Data"].sum()
        src = sorted(set(a["Source"]).union(set(b["Source"])))
        a = a.set_index("Source").reindex(src, fill_value=0).reset_index()
        b = b.set_index("Source").reindex(src, fill_value=0).reset_index()
        ta = float(a["Data"].sum())
        tb = float(b["Data"].sum())
        comp = pd.DataFrame(
            {
                "Source": src,
                f"Share {start_year}": [100.0 * float(v) / ta if ta > 0 else 0.0 for v in a["Data"]],
                f"Share {end_year}": [100.0 * float(v) / tb if tb > 0 else 0.0 for v in b["Data"]],
            }
        )
        comp["DeltaPctPt"] = comp[f"Share {end_year}"] - comp[f"Share {start_year}"]
        long = comp.melt(id_vars=["Source", "DeltaPctPt"], value_vars=[f"Share {start_year}", f"Share {end_year}"], var_name="Year", value_name="SharePct")
        if ECHARTS_AVAILABLE:
            option_comp = {
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "legend": {"top": 0},
                "grid": {"left": 55, "right": 20, "top": 45, "bottom": 30, "containLabel": True},
                "xAxis": {"type": "value", "name": "Share (%)"},
                "yAxis": {"type": "category", "data": src},
                "series": [
                    {
                        "name": f"Share {start_year}",
                        "type": "bar",
                        "itemStyle": {"color": "#94a3b8"},
                        "data": [round(float(v), 2) for v in comp[f"Share {start_year}"].tolist()],
                    },
                    {
                        "name": f"Share {end_year}",
                        "type": "bar",
                        "itemStyle": {"color": "#2563eb"},
                        "data": [round(float(v), 2) for v in comp[f"Share {end_year}"].tolist()],
                    },
                ],
            }
            render_echarts(option_comp, height=390, key="ech_share_compare")
        else:
            fig_comp = px.bar(
                long,
                x="SharePct",
                y="Source",
                color="Year",
                barmode="group",
                color_discrete_map={f"Share {start_year}": "#94a3b8", f"Share {end_year}": "#2563eb"},
                labels={"SharePct": "Share (%)"},
            )
            st.plotly_chart(fig_style(fig_comp, height=390, showlegend=True))

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Canada Total Capacity Trend")
        can_total = flt[flt["Region"] == "Canada"].groupby("Year", as_index=False)["Data"].sum()
        if ECHARTS_AVAILABLE:
            option_can_total = {
                "tooltip": {"trigger": "axis"},
                "grid": {"left": 45, "right": 20, "top": 20, "bottom": 35, "containLabel": True},
                "xAxis": {"type": "category", "data": [int(y) for y in can_total["Year"].tolist()]},
                "yAxis": {"type": "value", "name": "Capacity (MW)"},
                "series": [
                    {
                        "type": "line",
                        "smooth": True,
                        "lineStyle": {"width": 3, "color": "#0f172a"},
                        "areaStyle": {"color": "rgba(15,23,42,0.10)"},
                        "data": [round(float(v), 2) for v in can_total["Data"].tolist()],
                    }
                ],
            }
            render_echarts(option_can_total, height=320, key="ech_can_total")
        else:
            fig_can_total = px.line(can_total, x="Year", y="Data", labels={"Data": "Capacity (MW)"})
            st.plotly_chart(fig_style(fig_can_total, height=320, showlegend=False))

    with c4:
        st.subheader("Renewable vs Non-Renewable Capacity")
        can_mix_time = (
            flt[flt["Region"] == "Canada"]
            .assign(EnergyType=lambda d: np.where(d["Source"].isin(RENEWABLE), "Renewable", "Non-Renewable"))
            .groupby(["Year", "EnergyType"], as_index=False)["Data"]
            .sum()
        )
        piv_mix = can_mix_time.pivot(index="Year", columns="EnergyType", values="Data").fillna(0.0)
        if "Renewable" not in piv_mix.columns:
            piv_mix["Renewable"] = 0.0
        if "Non-Renewable" not in piv_mix.columns:
            piv_mix["Non-Renewable"] = 0.0
        if ECHARTS_AVAILABLE:
            option_mix_time = {
                "tooltip": {"trigger": "axis"},
                "legend": {"top": 0},
                "grid": {"left": 45, "right": 20, "top": 35, "bottom": 35, "containLabel": True},
                "xAxis": {"type": "category", "data": [int(y) for y in piv_mix.index.tolist()]},
                "yAxis": {"type": "value", "name": "Capacity (MW)"},
                "series": [
                    {
                        "name": "Renewable",
                        "type": "line",
                        "stack": "total",
                        "areaStyle": {},
                        "lineStyle": {"color": "#2563eb"},
                        "data": [round(float(v), 2) for v in piv_mix["Renewable"].tolist()],
                    },
                    {
                        "name": "Non-Renewable",
                        "type": "line",
                        "stack": "total",
                        "areaStyle": {},
                        "lineStyle": {"color": "#dc2626"},
                        "data": [round(float(v), 2) for v in piv_mix["Non-Renewable"].tolist()],
                    },
                ],
            }
            render_echarts(option_mix_time, height=320, key="ech_mix_time")
        else:
            mix_plot = piv_mix.reset_index().melt(id_vars="Year", value_vars=["Renewable", "Non-Renewable"], var_name="EnergyType", value_name="Capacity")
            fig_mix_time = px.area(mix_plot, x="Year", y="Capacity", color="EnergyType")
            st.plotly_chart(fig_style(fig_mix_time, height=320, showlegend=True))

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Canada Source Mix Donut")
        share_latest = (
            flt[(flt["Region"] == "Canada") & (flt["Year"] == latest_year)]
            .groupby("Source", as_index=False)["Data"]
            .sum()
            .sort_values("Data", ascending=False)
        )
        total_latest = float(share_latest["Data"].sum()) if not share_latest.empty else 0.0
        share_latest["SharePct"] = np.where(total_latest > 0, 100.0 * share_latest["Data"] / total_latest, 0.0)

        if ECHARTS_AVAILABLE:
            option_donut = {
                "tooltip": {"trigger": "item", "formatter": "{b}<br/>{c} MW ({d}%)"},
                "legend": {"type": "scroll", "orient": "vertical", "right": 5, "top": 20, "bottom": 20},
                "series": [
                    {
                        "name": f"{latest_year} Source Share",
                        "type": "pie",
                        "radius": ["42%", "72%"],
                        "center": ["38%", "52%"],
                        "avoidLabelOverlap": True,
                        "label": {"show": True, "formatter": "{b}: {d}%"},
                        "data": [
                            {"name": str(r["Source"]), "value": round(float(r["Data"]), 2)}
                            for _, r in share_latest.iterrows()
                        ],
                    }
                ],
                "title": {
                    "text": f"{latest_year}",
                    "left": "38%",
                    "top": "48%",
                    "textAlign": "center",
                    "textStyle": {"fontSize": 16, "fontWeight": "bold"},
                },
            }
            render_echarts(option_donut, height=340, key="ech_source_donut")
        else:
            fig_donut = px.pie(
                share_latest,
                names="Source",
                values="Data",
                hole=0.52,
            )
            fig_donut.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_style(fig_donut, height=340, showlegend=True))

    with c6:
        st.subheader("Animated Source Capacity Race (Canada)")
        race_src = flt[flt["Region"] == "Canada"].groupby(["Year", "Source"], as_index=False)["Data"].sum()
        race_src_years = sorted(race_src["Year"].unique())
        race_src_placeholder = st.empty()
        if auto_play and race_src_years:
            for yr in race_src_years:
                fr = race_src[race_src["Year"] == yr].sort_values("Data", ascending=True)
                if ECHARTS_AVAILABLE:
                    option_sr = {
                        "title": {"text": f"Year {int(yr)}", "left": "center"},
                        "grid": {"left": 90, "right": 20, "top": 45, "bottom": 25, "containLabel": True},
                        "xAxis": {"type": "value", "name": "Capacity (MW)"},
                        "yAxis": {"type": "category", "data": fr["Source"].tolist()},
                        "series": [
                            {
                                "type": "bar",
                                "data": [round(float(v), 2) for v in fr["Data"].tolist()],
                                "itemStyle": {"color": "#1d4ed8"},
                            }
                        ],
                    }
                    with race_src_placeholder:
                        render_echarts(option_sr, height=340, key=f"ech_src_race_{int(yr)}")
                else:
                    fig_sr = px.bar(fr, x="Data", y="Source", orientation="h", title=f"Year {int(yr)}")
                    race_src_placeholder.plotly_chart(fig_style(fig_sr, height=340, showlegend=False))
                time.sleep(float(anim_speed))
        elif race_src_years:
            yr = race_src_years[-1]
            fr = race_src[race_src["Year"] == yr].sort_values("Data", ascending=True)
            if ECHARTS_AVAILABLE:
                option_sr = {
                    "title": {"text": f"Year {int(yr)}", "left": "center"},
                    "grid": {"left": 90, "right": 20, "top": 45, "bottom": 25, "containLabel": True},
                    "xAxis": {"type": "value", "name": "Capacity (MW)"},
                    "yAxis": {"type": "category", "data": fr["Source"].tolist()},
                    "series": [{"type": "bar", "data": [round(float(v), 2) for v in fr["Data"].tolist()], "itemStyle": {"color": "#1d4ed8"}}],
                }
                with race_src_placeholder:
                    render_echarts(option_sr, height=340, key=f"ech_src_race_static_{int(yr)}")
            else:
                fig_sr = px.bar(fr, x="Data", y="Source", orientation="h", title=f"Year {int(yr)}")
                race_src_placeholder.plotly_chart(fig_style(fig_sr, height=340, showlegend=False))

with tab2:
    st.markdown(
        "<div class='section-lead'><strong>Diagnosis goal:</strong> quantify transition momentum, volatility risk, and regional divergence through time.</div>",
        unsafe_allow_html=True,
    )
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Canada Renewable Share Trend")
        st.caption("A sustained upward slope indicates structural transition rather than a one-off spike.")
        rs = (
            flt[flt["Region"] == "Canada"]
            .assign(EnergyType=lambda d: np.where(d["Source"].isin(RENEWABLE), "Renewable", "Non-Renewable"))
            .groupby(["Year", "EnergyType"], as_index=False)["Data"]
            .sum()
        )
        piv = rs.pivot(index="Year", columns="EnergyType", values="Data").fillna(0.0)
        if "Renewable" not in piv.columns:
            piv["Renewable"] = 0.0
        if "Non-Renewable" not in piv.columns:
            piv["Non-Renewable"] = 0.0
        if not piv.empty:
            trend_df = pd.DataFrame({
                "Year": piv.index,
                "RenewableShare": np.where((piv["Renewable"] + piv["Non-Renewable"]) > 0, 100.0 * piv["Renewable"] / (piv["Renewable"] + piv["Non-Renewable"]), 0.0),
            })
            if ECHARTS_AVAILABLE:
                option_tr = {
                    "tooltip": {"trigger": "axis"},
                    "grid": {"left": 45, "right": 20, "top": 20, "bottom": 35, "containLabel": True},
                    "xAxis": {"type": "category", "data": [int(y) for y in trend_df["Year"].tolist()]},
                    "yAxis": {"type": "value", "name": "Renewable Share (%)", "max": 100},
                    "series": [
                        {
                            "name": "Renewable Share",
                            "type": "line",
                            "smooth": True,
                            "symbol": "circle",
                            "lineStyle": {"width": 3, "color": "#1d4ed8"},
                            "itemStyle": {"color": "#1d4ed8"},
                            "areaStyle": {"color": "rgba(37,99,235,0.15)"},
                            "data": [round(float(v), 2) for v in trend_df["RenewableShare"].tolist()],
                        }
                    ],
                }
                render_echarts(option_tr, height=330, key="ech_renew_trend")
            else:
                fig_tr = px.line(trend_df, x="Year", y="RenewableShare", markers=True, labels={"RenewableShare": "Renewable Share (%)"})
                st.plotly_chart(fig_style(fig_tr, height=330, showlegend=False))

    with c4:
        st.subheader("Volatility vs Capacity Map")
        st.caption("High-capacity and high-volatility sources represent the largest planning risk.")
        vol = flt.groupby(["Region", "Source"], as_index=False).agg(
            Mean=("Data", "mean"),
            Std=("Data", "std"),
        )
        vol["Std"] = vol["Std"].fillna(0.0)
        vol["CV"] = np.where(vol["Mean"] > 0, 100.0 * vol["Std"] / vol["Mean"], 0.0)
        if not vol.empty:
            vol["SizePx"] = np.clip(8.0 + (vol["Std"] / max(vol["Std"].max(), 1.0)) * 18.0, 8.0, 26.0)
            fig_vol = px.scatter(
                vol,
                x="CV",
                y="Mean",
                color="Source",
                size="SizePx",
                hover_name="Region",
                labels={"CV": "Volatility (CV %)", "Mean": "Average Capacity (MW)"},
            )
            fig_vol.update_traces(marker={"opacity": 0.85, "line": {"width": 0.5, "color": "white"}})
            st.plotly_chart(fig_style(fig_vol, height=330, showlegend=True))
        else:
            st.info("No data available for the current filters to build this panel.")

    st.subheader("Transition Momentum by Region")
    latest_by_region = region_piv[region_piv["Year"] == region_piv["Year"].max()][["Region", "RenewableShare"]].copy()
    start_by_region = region_piv[region_piv["Year"] == region_piv["Year"].min()][["Region", "RenewableShare"]].copy()
    latest_by_region = latest_by_region.rename(columns={"RenewableShare": "LatestRenewShare"})
    start_by_region = start_by_region.rename(columns={"RenewableShare": "StartRenewShare"})
    momentum = latest_by_region.merge(start_by_region, on="Region", how="left")
    momentum["DeltaPctPt"] = momentum["LatestRenewShare"] - momentum["StartRenewShare"]
    momentum = momentum.sort_values("DeltaPctPt", ascending=False).reset_index(drop=True)

    st.caption("Positive values indicate faster transition momentum since the start year.")
    if ECHARTS_AVAILABLE:
        option_momentum = {
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "grid": {"left": 50, "right": 20, "top": 20, "bottom": 40, "containLabel": True},
            "xAxis": {"type": "category", "data": momentum["Region"].tolist()},
            "yAxis": {"type": "value", "name": "Change in Renewable Share (pct-pt)"},
            "series": [
                {
                    "type": "bar",
                    "data": [round(float(v), 2) for v in momentum["DeltaPctPt"].tolist()],
                    "itemStyle": {
                        "color": "#0ea5a0"
                    },
                }
            ],
        }
        render_echarts(option_momentum, height=320, key="ech_momentum")
    else:
        fig_mom = px.bar(
            momentum,
            x="Region",
            y="DeltaPctPt",
            labels={"DeltaPctPt": "Change in Renewable Share (pct-pt)"},
        )
        st.plotly_chart(fig_style(fig_mom, height=320, showlegend=False))

    st.subheader("Transition Path")
    chart_mode = st.radio(
        "Line animation mode",
        ["Regional renewable share (%)", "Source capacity change (MW)"],
        index=1,
        horizontal=True,
    )

    if chart_mode == "Regional renewable share (%)":
        st.caption("Animated line view of renewable-share progression by region.")
        line_df = region_piv[["Year", "Region", "RenewableShare"]].rename(columns={"Region": "Series", "RenewableShare": "Value"})
        y_name = "Renewable Share (%)"
        y_min, y_max = 0, 100
        key_prefix = "ech_line_share"
    else:
        st.caption("Animated line view of source-capacity change over time (Canada aggregate).")
        src_df = (
            flt[flt["Region"] == "Canada"]
            .groupby(["Year", "Source"], as_index=False)["Data"]
            .sum()
            .rename(columns={"Source": "Series", "Data": "Value"})
        )
        line_df = src_df.copy()
        y_name = "Capacity (MW)"
        y_min, y_max = 0, float(line_df["Value"].max()) * 1.05 if not line_df.empty else 1
        key_prefix = "ech_line_source"

    years = sorted(line_df["Year"].unique()) if not line_df.empty else []
    line_placeholder = st.empty()

    if auto_play and years:
        for yr in years:
            frame = line_df[line_df["Year"] <= yr].sort_values(["Series", "Year"])
            if ECHARTS_AVAILABLE:
                names = sorted(frame["Series"].unique())
                series_line = []
                for nm in names:
                    nm_df = frame[frame["Series"] == nm]
                    series_line.append(
                        {
                            "name": nm,
                            "type": "line",
                            "smooth": True,
                            "showSymbol": False,
                            "lineStyle": {"width": 2},
                            "data": [[int(y), round(float(v), 2)] for y, v in zip(nm_df["Year"], nm_df["Value"])],
                        }
                    )

                option_line = {
                    "animationDuration": 420,
                    "animationDurationUpdate": 550,
                    "title": {"text": f"Up to Year {int(yr)}", "left": "center"},
                    "legend": {"type": "scroll", "top": 0},
                    "grid": {"left": 55, "right": 35, "top": 52, "bottom": 38, "containLabel": True},
                    "xAxis": {"type": "value", "name": "Year", "min": int(years[0]), "max": int(years[-1])},
                    "yAxis": {"type": "value", "name": y_name, "min": y_min, "max": y_max},
                    "series": series_line,
                }
                with line_placeholder:
                    render_echarts(option_line, height=450, key=f"{key_prefix}_{int(yr)}")
            else:
                fig_line = px.line(frame, x="Year", y="Value", color="Series", markers=False)
                fig_line.update_yaxes(range=[y_min, y_max], title=y_name)
                fig_line.update_xaxes(range=[years[0], years[-1]], title="Year")
                line_placeholder.plotly_chart(fig_style(fig_line, height=450, showlegend=True))
            time.sleep(float(anim_speed))
    elif years:
        frame = line_df.sort_values(["Series", "Year"])
        if ECHARTS_AVAILABLE:
            names = sorted(frame["Series"].unique())
            series_line = []
            for nm in names:
                nm_df = frame[frame["Series"] == nm]
                series_line.append(
                    {
                        "name": nm,
                        "type": "line",
                        "smooth": True,
                        "showSymbol": False,
                        "lineStyle": {"width": 2},
                        "data": [[int(y), round(float(v), 2)] for y, v in zip(nm_df["Year"], nm_df["Value"])],
                    }
                )
            option_line = {
                "title": {"text": "Full Period", "left": "center"},
                "legend": {"type": "scroll", "top": 0},
                "grid": {"left": 55, "right": 35, "top": 52, "bottom": 38, "containLabel": True},
                "xAxis": {"type": "value", "name": "Year", "min": int(years[0]), "max": int(years[-1])},
                "yAxis": {"type": "value", "name": y_name, "min": y_min, "max": y_max},
                "series": series_line,
            }
            with line_placeholder:
                render_echarts(option_line, height=450, key=f"{key_prefix}_static")
        else:
            fig_line = px.line(frame, x="Year", y="Value", color="Series", markers=False)
            fig_line.update_yaxes(range=[y_min, y_max], title=y_name)
            fig_line.update_xaxes(range=[years[0], years[-1]], title="Year")
            line_placeholder.plotly_chart(fig_style(fig_line, height=450, showlegend=True))

with tab3:
    st.markdown(
        "<div class='section-lead'><strong>Forecast goal:</strong> compare tuned models, justify best method per series, and expose horizon-sensitive predictions for decision use.</div>",
        unsafe_allow_html=True,
    )
    st.subheader("Per-Source Forecasting and Model Selection")
    st.caption("Model ranking uses leakage-safe walk-forward backtesting, not random splits.")
    series = base[(base["Region"] == lab_region) & (base["Source"] == lab_source)][["Year", "Data"]].sort_values("Year")
    if len(series) < 10:
        st.warning("Selected series has fewer than 10 observations; choose another source/region.")
    else:
        fc, met = micro_forecast(series, horizon=int(lab_horizon))
        signal = summarize_prediction_signal(series, fc, met, int(lab_horizon))

        st.markdown("**Prediction Intelligence**")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Predicted direction", str(signal.get("Signal", "n/a")))
        s2.metric(f"{lab_horizon}Y delta", f"{float(signal.get('DeltaMW', 0.0)):,.0f} MW")
        s3.metric("% change", f"{float(signal.get('DeltaPct', 0.0)):.1f}%")
        s4.metric("Annualized (CAGR)", f"{float(signal.get('CAGR', 0.0)):.2f}%")
        st.caption(
            f"Interpretation: using {signal.get('BestModel', 'best')} model, forecast uncertainty band width at horizon is "
            f"{float(signal.get('UncertaintyPct', 0.0)):.1f}% of point forecast."
        )

        left, right = st.columns([1.45, 1])

        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series["Year"], y=series["Data"], mode="lines+markers", name="Historical", line={"width": 3, "color": "#0f172a"}))
            last_year = int(series["Year"].iloc[-1])
            last_val = float(series["Data"].iloc[-1])
            for m in sorted(fc["Model"].unique()):
                part = fc[fc["Model"] == m].sort_values("Year")
                line = pd.concat(
                    [pd.DataFrame({"Year": [last_year], "Forecast_MW": [last_val]}), part[["Year", "Forecast_MW"]]],
                    ignore_index=True,
                )
                fig.add_trace(go.Scatter(x=line["Year"], y=line["Forecast_MW"], mode="lines+markers", name=m))
            st.plotly_chart(fig_style(fig, height=430, showlegend=True))

        with right:
            met = met.sort_values(["RMSE", "MAE"]).reset_index(drop=True)
            show_cols = [c for c in ["Model", "MAE", "RMSE", "sMAPE", "BacktestPoints", "BestParams"] if c in met.columns]
            render_aggrid(met[show_cols].round(3), key="grid_metrics", height=250)
            top = met.iloc[0]
            st.markdown(f"**Best model:** {top['Model']}  ")
            st.markdown(
                f"**RMSE:** {float(top['RMSE']):.2f} | **MAE:** {float(top['MAE']):.2f} | **sMAPE:** {float(top['sMAPE']):.2f}%"
            )
            st.caption(f"Backtest points used for ranking: {int(top['BacktestPoints'])}")
            st.caption(model_commentary(met, series, lab_region, lab_source))

        top = met.iloc[0]
        st.markdown("**Selected Model and Practical Usefulness**")
        st.caption(
            f"Selected model for this series: {top['Model']}. It is chosen by lowest walk-forward RMSE/MAE and used to generate decision-ready horizon forecasts with uncertainty ranges."
        )

        st.markdown("**Forecast Output Table**")
        if prediction_view == "Best model only":
            best = str(met.iloc[0]["Model"])
            out = fc[fc["Model"] == best][["Year", "Forecast_MW", "Lower80_MW", "Upper80_MW", "Model"]].copy()
            out = out.rename(columns={"Forecast_MW": "Predicted_MW"}).round(2)
            render_aggrid(out, key="grid_prediction_best", height=260)
        else:
            wide = fc.pivot_table(index="Year", columns="Model", values="Forecast_MW", aggfunc="mean").reset_index().round(2)
            render_aggrid(wide, key="grid_prediction_all", height=260)
        st.caption(f"Live update: changing forecast horizon recalculates this table instantly (current horizon: {lab_horizon} years).")

        st.subheader("Region Outlook by Source (Meaningful Forecast Ranking)")
        st.caption("Ranks where capacity is forecast to expand or contract most in the selected region.")
        run_full_outlook = st.toggle("Compute full region outlook (slower)", value=False)
        if run_full_outlook:
            outlook = build_region_source_outlook(base, lab_region, tuple(all_sources), int(lab_horizon))
            if outlook.empty:
                st.info("Not enough history to produce region-level source ranking for this selection.")
            else:
                render_aggrid(outlook.round(2), key="grid_region_outlook", height=310)
                top_growth = outlook.iloc[0]
                top_decline = outlook.sort_values("DeltaPct", ascending=True).iloc[0]
                st.caption(
                    f"Top growth: {top_growth['Source']} ({top_growth['DeltaPct']:.1f}% over {lab_horizon} years). "
                    f"Largest decline: {top_decline['Source']} ({top_decline['DeltaPct']:.1f}%)."
                )
        else:
            st.info("Enable full region outlook to compute and rank all source forecasts for this region.")

with tab4:
    st.markdown(
        "<div class='section-lead'><strong>Decision goal:</strong> validate reliability at portfolio level, then convert evidence into ranked policy action.</div>",
        unsafe_allow_html=True,
    )

    live_per_source, live_by_model = build_live_reliability(base, lab_region, tuple(selected_sources), int(lab_horizon))

    rc1, rc2, rc3 = st.columns(3)
    if not live_by_model.empty:
        rc1.metric("Best model (live scope)", str(live_by_model.iloc[0]["BestModel"]))
        rc2.metric("Series wins (best)", f"{int(live_by_model.iloc[0]['WinCount'])}")
        rc3.metric("Mean RMSE (best)", f"{float(live_by_model.iloc[0]['MeanRMSE']):.2f}")

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Cross-Series Model Performance")
        st.caption("Computed from live model selection over currently selected sources.")
        if not live_by_model.empty:
            show_live = live_by_model.rename(columns={"BestModel": "Model", "MeanRMSE": "RMSE", "MeanMAE": "MAE", "MeansMAPE": "sMAPE"})
            render_aggrid(show_live.round(3), key="grid_live_global_summary", height=260)
            if ECHARTS_AVAILABLE:
                option_rel = {
                    "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                    "grid": {"left": 50, "right": 20, "top": 20, "bottom": 40, "containLabel": True},
                    "xAxis": {"type": "category", "data": show_live["Model"].tolist()},
                    "yAxis": {"type": "value", "name": "Mean RMSE"},
                    "series": [
                        {
                            "name": "Mean RMSE",
                            "type": "bar",
                            "data": [round(float(v), 3) for v in show_live["RMSE"].tolist()],
                            "itemStyle": {"color": "#0ea5a0"},
                        }
                    ],
                }
                render_echarts(option_rel, height=260, key="ech_live_global_rmse")
    with c6:
        st.subheader("Best-Model Win Count")
        st.caption("Shows how frequently each model is selected as best in current scope.")
        if not live_by_model.empty:
            counts = live_by_model[["BestModel", "WinCount"]].rename(columns={"BestModel": "Model", "WinCount": "Count"})
            render_aggrid(counts, key="grid_live_best_frequency", height=260)
        else:
            counts = pd.DataFrame(columns=["Model", "Count"])

        if ECHARTS_AVAILABLE and not counts.empty:
            option_wins = {
                "tooltip": {"trigger": "item"},
                "series": [
                    {
                        "type": "pie",
                        "radius": ["35%", "70%"],
                        "label": {"formatter": "{b}: {c}"},
                        "data": [
                            {"name": str(r["Model"]), "value": int(r["Count"])}
                            for _, r in counts.iterrows()
                        ],
                    }
                ],
            }
            render_echarts(option_wins, height=260, key="ech_best_model_wins")

    st.subheader("Forecast Deliverables (Live Model Output)")
    st.caption("Produced forecast outputs that directly support planning and policy decisions.")
    series_deliv = base[(base["Region"] == lab_region) & (base["Source"] == lab_source)][["Year", "Data"]].sort_values("Year")
    if len(series_deliv) < 10:
        st.info("Not enough historical data points for live forecast deliverables in this region/source.")
    else:
        fc_deliv, met_deliv = micro_forecast(series_deliv, horizon=int(lab_horizon))
        if fc_deliv.empty or met_deliv.empty:
            st.info("Live forecast could not be produced for this selection.")
        else:
            best_deliv = met_deliv.sort_values(["RMSE", "MAE"]).iloc[0]
            best_model_deliv = str(best_deliv["Model"])
            fo = fc_deliv[fc_deliv["Model"] == best_model_deliv].sort_values("Year").copy()

            if ECHARTS_AVAILABLE and not fo.empty:
                year_axis = [int(y) for y in series_deliv["Year"].tolist()] + [int(y) for y in fo["Year"].tolist()]
                actual_vals = [round(float(v), 2) for v in series_deliv["Data"].tolist()] + [None] * len(fo)
                fc_vals = [None] * len(series_deliv) + [round(float(v), 2) for v in fo["Forecast_MW"].tolist()]
                lo_vals = [None] * len(series_deliv) + [round(float(v), 2) for v in fo["Lower80_MW"].tolist()]
                hi_vals = [None] * len(series_deliv) + [round(float(v), 2) for v in fo["Upper80_MW"].tolist()]
                option_fo = {
                    "tooltip": {"trigger": "axis"},
                    "legend": {"top": 0},
                    "grid": {"left": 50, "right": 20, "top": 40, "bottom": 35, "containLabel": True},
                    "xAxis": {"type": "category", "data": year_axis},
                    "yAxis": {"type": "value", "name": "MW"},
                    "series": [
                        {
                            "name": "Actual",
                            "type": "line",
                            "data": actual_vals,
                            "smooth": False,
                            "lineStyle": {"width": 2, "color": "#111827"},
                        },
                        {
                            "name": "Forecast",
                            "type": "line",
                            "data": fc_vals,
                            "smooth": True,
                            "lineStyle": {"width": 3, "color": "#1d4ed8"},
                        },
                        {
                            "name": "Lower80",
                            "type": "line",
                            "data": lo_vals,
                            "lineStyle": {"width": 1, "type": "dashed", "color": "#6b7280"},
                            "symbol": "none",
                        },
                        {
                            "name": "Upper80",
                            "type": "line",
                            "data": hi_vals,
                            "lineStyle": {"width": 1, "type": "dashed", "color": "#6b7280"},
                            "symbol": "none",
                        },
                    ],
                }
                render_echarts(option_fo, height=300, key="ech_forecast_live_deliverables")
            elif not fo.empty:
                hist_plot = series_deliv.rename(columns={"Data": "Actual_MW"}).copy()
                hist_plot["Series"] = "Actual"
                fc_plot = fo[["Year", "Forecast_MW"]].rename(columns={"Forecast_MW": "Actual_MW"}).copy()
                fc_plot["Series"] = "Forecast"
                combo_plot = pd.concat([hist_plot[["Year", "Actual_MW", "Series"]], fc_plot], ignore_index=True)
                fig_fo = px.line(combo_plot, x="Year", y="Actual_MW", color="Series", title="Live forecast deliverable")
                st.plotly_chart(fig_style(fig_fo, height=300))

            fo_table = fo[[c for c in ["Year", "Forecast_MW", "Lower80_MW", "Upper80_MW"] if c in fo.columns]].copy()
            fo_table["Model"] = best_model_deliv
            render_aggrid(fo_table.round(2), key="grid_forecast_live_deliverables", height=220)
            st.caption(
                f"Live deliverable uses current walk-forward ranking. Best model: {best_model_deliv} "
                f"(RMSE {float(best_deliv['RMSE']):.2f}, MAE {float(best_deliv['MAE']):.2f}, sMAPE {float(best_deliv['sMAPE']):.2f})."
            )

    st.markdown(
        "**Executive takeaway:** combine transition momentum, model reliability, and policy priority ranking to justify resource allocation decisions.")

    st.subheader("Model Choice Justification")
    st.caption("Models were chosen to cover persistence, trend, nonlinear effects, blended robustness, and dedicated time-series baselines.")
    model_just = pd.DataFrame(
        [
            {"Model": "NaiveLag1", "Why included": "Strong persistence baseline", "Evidence trigger": "High lag correlation", "Risk": "Misses trend shifts"},
            {"Model": "DriftTrend_TS", "Why included": "Time-series drift baseline", "Evidence trigger": "Long-run linear direction", "Risk": "Over-simplifies nonlinear regimes"},
            {"Model": "SES_TS", "Why included": "Level-focused time-series smoother", "Evidence trigger": "Noisy short-run fluctuations", "Risk": "Weak for structural breaks"},
            {"Model": "Blend_Naive_Drift_SES", "Why included": "Pure time-series ensemble", "Evidence trigger": "Persistence + drift + smoothing all present", "Risk": "Can underfit strong nonlinearity"},
            {"Model": "LinearTrend", "Why included": "Interpretable lag+trend regression", "Evidence trigger": "Trend with moderate noise", "Risk": "Cannot capture complex interactions"},
            {"Model": "Blend_Naive_Linear_RF", "Why included": "Diversified ensemble", "Evidence trigger": "Mixed persistence+trend+nonlinear patterns", "Risk": "More complex to explain"},
            {"Model": "RandomForest", "Why included": "Nonlinear interactions", "Evidence trigger": "Volatile or regime-dependent behavior", "Risk": "Can overfit small samples"},
        ]
    )
    render_aggrid(model_just, key="grid_model_justification", height=300)
