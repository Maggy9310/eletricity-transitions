from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Canada Electricity Intelligence Dashboard", page_icon="⚡", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "electricity-capacity-dataset.csv"
FORECAST_PATH = ROOT / "output" / "forecast_5y_region_source.csv"
OUTPUT_DIR = ROOT / "output"


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0) else 0.0


@st.cache_data
def load_dataset(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required_cols = ["Region", "Source", "Year", "Data", "Unit"]
    df = raw.dropna(subset=required_cols).copy()
    df["Region"] = df["Region"].astype(str).str.strip()
    df["Source"] = df["Source"].astype(str).str.strip()
    df["Year"] = df["Year"].astype(int)
    df["Data"] = df["Data"].astype(float)
    return df.sort_values(["Region", "Source", "Year"]).reset_index(drop=True)


@st.cache_data
def load_forecast(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    fc = pd.read_csv(path)
    keep = {"Region", "Source", "Year", "Forecast_MW", "Lower80_MW", "Upper80_MW", "Model"}
    if not keep.issubset(fc.columns):
        return None
    fc["Year"] = fc["Year"].astype(int)
    return fc


@st.cache_data
def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data
def build_region_year_features(df: pd.DataFrame, renew_sources: set[str]) -> pd.DataFrame:
    region_year = (
        df[df["Region"] != "Canada"]
        .groupby(["Region", "Year"], as_index=False)["Data"]
        .sum()
        .rename(columns={"Data": "total_capacity"})
    )
    renew = (
        df[(df["Region"] != "Canada") & (df["Source"].isin(renew_sources))]
        .groupby(["Region", "Year"], as_index=False)["Data"]
        .sum()
        .rename(columns={"Data": "renew_capacity"})
    )
    out = region_year.merge(renew, on=["Region", "Year"], how="left").fillna({"renew_capacity": 0.0})
    out["renew_share"] = out["renew_capacity"] / out["total_capacity"]
    out["fossil_share"] = 1.0 - out["renew_share"]
    return out


@st.cache_data
def build_region_features_fallback(region_mix: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    shares = df[df["Region"] != "Canada"][["Region", "Year", "Source", "Data"]].merge(
        region_mix[["Region", "Year", "total_capacity"]],
        on=["Region", "Year"],
        how="left",
    )
    shares["src_share"] = shares["Data"] / shares["total_capacity"]
    hhi_tmp = (
        shares.groupby(["Region", "Year"], as_index=False)
        .apply(lambda x: (x["src_share"] ** 2).sum(), include_groups=False)
        .rename(columns={None: "hhi"})
    )
    merged = region_mix.merge(hhi_tmp, on=["Region", "Year"], how="left")
    region_features = (
        merged.groupby("Region", as_index=False)
        .agg(
            renew_share_mean=("renew_share", "mean"),
            renew_share_std=("renew_share", "std"),
            fossil_share_mean=("fossil_share", "mean"),
            hhi_mean=("hhi", "mean"),
            capacity_mean=("total_capacity", "mean"),
        )
        .fillna(0.0)
    )
    region_features["Cluster"] = "N/A"
    region_features["ClusterLabel"] = "Cluster N/A"
    return region_features


def build_priority_table_like_notebook(region_mix: pd.DataFrame, region_features: pd.DataFrame) -> pd.DataFrame:
    by_region = region_mix.groupby("Region", as_index=False).agg(
        renew_start=("renew_share", "first"),
        renew_end=("renew_share", "last"),
        fossil_mean=("fossil_share", "mean"),
        capacity_mean=("total_capacity", "mean"),
    )
    by_region["renew_delta"] = by_region["renew_end"] - by_region["renew_start"]

    cut_end = by_region["renew_end"].median()
    cut_delta = by_region["renew_delta"].median()

    by_region["TransitionLabel"] = np.select(
        [
            (by_region["renew_end"] >= cut_end) & (by_region["renew_delta"] >= cut_delta),
            (by_region["renew_end"] < cut_end) & (by_region["renew_delta"] >= cut_delta),
            (by_region["renew_end"] >= cut_end) & (by_region["renew_delta"] < cut_delta),
        ],
        ["Leader", "Emerging", "Mature-Stable"],
        default="Lagging",
    )

    out = by_region.merge(
        region_features[["Region", "fossil_share_mean", "capacity_mean", "Cluster"]],
        on="Region",
        how="left",
    )
    out["TransitionScore"] = (
        0.55 * out["renew_end"].rank(pct=True)
        + 0.45 * out["renew_delta"].rank(pct=True)
    ) * 100.0

    priority_raw = (
        0.60 * (1.0 - out["TransitionScore"] / 100.0)
        + 0.25 * out["fossil_share_mean"].rank(pct=True)
        + 0.15 * out["capacity_mean"].rank(pct=True)
    )
    out["PriorityScore"] = (priority_raw * 100.0).round(1)

    q_high = float(out["PriorityScore"].quantile(2 / 3))
    q_mid = float(out["PriorityScore"].quantile(1 / 3))
    out["Priority"] = np.select(
        [out["PriorityScore"] >= q_high, out["PriorityScore"] >= q_mid],
        ["High", "Medium"],
        default="Low",
    )
    return out.sort_values(["PriorityScore", "capacity_mean"], ascending=[False, False])


def normalize_priority_table(priority_table: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    out = priority_table.copy()
    q_high = float(out["PriorityScore"].quantile(2 / 3))
    q_mid = float(out["PriorityScore"].quantile(1 / 3))

    if "Priority" not in out.columns:
        out["Priority"] = np.select(
            [out["PriorityScore"] >= q_high, out["PriorityScore"] >= q_mid],
            ["High", "Medium"],
            default="Low",
        )

    if "TransitionScore" not in out.columns and {"renew_end", "renew_delta"}.issubset(out.columns):
        out["TransitionScore"] = (
            0.55 * out["renew_end"].rank(pct=True)
            + 0.45 * out["renew_delta"].rank(pct=True)
        ) * 100.0

    return out, q_high, q_mid


def ensure_cluster_labels(region_features: pd.DataFrame) -> pd.DataFrame:
    out = region_features.copy()
    if "Cluster" not in out.columns:
        out["Cluster"] = "N/A"
    out["Cluster"] = out["Cluster"].astype(str)
    if "ClusterLabel" not in out.columns:
        out["ClusterLabel"] = "Cluster " + out["Cluster"]
    return out


if not DATA_PATH.exists():
    st.error("Dataset file electricity-capacity-dataset.csv is missing. Place it in the project root.")
    st.stop()

df = load_dataset(DATA_PATH)
renew_sources = {"Hydro", "Wind", "Solar", "Biomass"}
region_mix_full = build_region_year_features(df, renew_sources)

region_features_csv = load_optional_csv(OUTPUT_DIR / "nb_region_features_clusters.csv")
priority_csv = load_optional_csv(OUTPUT_DIR / "nb_priority_table.csv")
shift_df = load_optional_csv(OUTPUT_DIR / "nb_transition_stresstest_shifts.csv")

if isinstance(region_features_csv, pd.DataFrame) and {"Region", "renew_share_mean", "hhi_mean", "capacity_mean"}.issubset(region_features_csv.columns):
    region_features = ensure_cluster_labels(region_features_csv)
else:
    region_features = ensure_cluster_labels(build_region_features_fallback(region_mix_full, df))

if isinstance(priority_csv, pd.DataFrame) and {"Region", "PriorityScore"}.issubset(priority_csv.columns):
    priority_table, q_high, q_mid = normalize_priority_table(priority_csv)
else:
    priority_table = build_priority_table_like_notebook(region_mix_full, region_features)
    priority_table, q_high, q_mid = normalize_priority_table(priority_table)

forecast_df = load_forecast(FORECAST_PATH)

st.title("⚡ Canada Electricity Transition Intelligence Dashboard")
st.caption("Integrated 3×2 dashboard aligned with notebook outputs: trajectory, composition, clustered structure, priority, transition score, and stress-test impact.")

all_years = sorted(df["Year"].unique())
all_regions = sorted([r for r in df["Region"].unique() if r != "Canada"])

with st.sidebar:
    st.subheader("Filters")
    year_min, year_max = st.slider("Year range", min_value=min(all_years), max_value=max(all_years), value=(min(all_years), max(all_years)))
    selected_regions = st.multiselect("Regions", options=all_regions, default=all_regions)

if not selected_regions:
    selected_regions = all_regions

flt = df[df["Year"].between(year_min, year_max)]
flt_can = flt[flt["Region"] == "Canada"]

region_mix_flt = region_mix_full[
    (region_mix_full["Year"].between(year_min, year_max))
    & (region_mix_full["Region"].isin(selected_regions))
]

latest_year = int(region_mix_flt["Year"].max()) if not region_mix_flt.empty else max(all_years)
latest = region_mix_flt[region_mix_flt["Year"] == latest_year]

total_capacity_latest = float(latest["total_capacity"].sum()) if not latest.empty else 0.0
renew_capacity_latest = float(latest["renew_capacity"].sum()) if not latest.empty else 0.0
renew_share_latest = _safe_div(renew_capacity_latest, total_capacity_latest)

base_year = int(region_mix_flt["Year"].min()) if not region_mix_flt.empty else min(all_years)
base_total = float(region_mix_flt[region_mix_flt["Year"] == base_year]["total_capacity"].sum()) if not region_mix_flt.empty else 0.0
years_span = max(1, latest_year - base_year)
cagr = ((total_capacity_latest / base_total) ** (1 / years_span) - 1) if base_total > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Capacity (Latest Year)", f"{total_capacity_latest:,.0f} MW")
col2.metric("Renewable Share (Latest Year)", f"{renew_share_latest:.1%}")
col3.metric("Capacity CAGR", f"{cagr:.2%}")
col4.metric("Priority Thresholds", f"H≥{q_high:.1f} | M≥{q_mid:.1f}")

if shift_df is not None and {"Region", "Changed"}.issubset(shift_df.columns):
    shifted_regions = int(shift_df[(shift_df["Region"] != "Canada") & (shift_df["Changed"] == True)].shape[0])
else:
    shifted_regions = None

ml1, ml2, ml3 = st.columns(3)
ml1.metric("Transition Groups", str(int(priority_table["TransitionLabel"].nunique())) if "TransitionLabel" in priority_table.columns else "N/A")
ml2.metric("High Priority Regions", str(int((priority_table["Priority"] == "High").sum())))
ml3.metric("Regions Shifted in Stress-Test", "N/A" if shifted_regions is None else str(shifted_regions))

score_legend = pd.DataFrame(
    [
        ["TransitionScore", "0.55*rank(renew_end) + 0.45*rank(renew_delta)", "Higher = stronger transition progress", "Displayed in panel 5"],
        ["PriorityScore", "0.60*(1-TransitionScore) + 0.25*rank(fossil) + 0.15*rank(capacity)", "Higher = more urgent policy priority", "Displayed in panel 4"],
        ["Color code", "PriorityScore terciles", "High=Red, Medium=Orange, Low=Green", f"Cutoffs: High ≥ {q_high:.1f}, Medium ≥ {q_mid:.1f}, Low < {q_mid:.1f}"],
    ],
    columns=["Item", "How computed", "Interpretation", "Visual meaning"],
)
with st.expander("Score legend and interpretation", expanded=False):
    st.dataframe(score_legend, width="stretch", hide_index=True)

row1_col1, row1_col2, row1_col3 = st.columns(3)
row2_col1, row2_col2, row2_col3 = st.columns(3)

with row1_col1:
    can_tot = flt_can.groupby("Year", as_index=False)["Data"].sum()
    fig_national = px.line(can_tot, x="Year", y="Data", markers=True, title="1) National Capacity Trend", labels={"Data": "Capacity (MW)"})
    fig_national.update_layout(hovermode="x unified")
    st.plotly_chart(fig_national, width="stretch")

with row1_col2:
    latest_rank_panel = latest.sort_values("renew_share", ascending=False).copy()
    latest_rank_panel["non_renew_share"] = 1.0 - latest_rank_panel["renew_share"]
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(x=latest_rank_panel["Region"], y=latest_rank_panel["renew_share"], name="Renewable Share"))
    fig_comp.add_trace(go.Bar(x=latest_rank_panel["Region"], y=latest_rank_panel["non_renew_share"], name="Non-renewable Share"))
    fig_comp.update_layout(title=f"2) Renewable vs Non-renewable Composition ({latest_year})", barmode="group")
    fig_comp.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_comp, width="stretch")

with row1_col3:
    cluster_panel = region_features[region_features["Region"].isin(selected_regions)].copy()
    if {"renew_share_mean", "hhi_mean", "capacity_mean", "ClusterLabel"}.issubset(cluster_panel.columns):
        fig_cluster = px.scatter(
            cluster_panel,
            x="renew_share_mean",
            y="hhi_mean",
            color="ClusterLabel",
            size="capacity_mean",
            text="Region",
            title="3) Clustered Regional Structure",
            labels={"renew_share_mean": "Renewable Share (Mean)", "hhi_mean": "Concentration (HHI)"},
        )
        fig_cluster.update_traces(textposition="top center")
        st.plotly_chart(fig_cluster, width="stretch")
    else:
        st.info("Cluster structure data not available.")

priority_color_map = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}

with row2_col1:
    ps = priority_table[priority_table["Region"].isin(selected_regions)].copy().sort_values("PriorityScore", ascending=False)
    if not ps.empty:
        fig_priority = px.bar(ps, x="Region", y="PriorityScore", color="Priority", title="4) Priority Score by Region", color_discrete_map=priority_color_map)
        st.plotly_chart(fig_priority, width="stretch")
    else:
        st.info("Priority data not available.")

with row2_col2:
    transition_view = priority_table[priority_table["Region"].isin(selected_regions)].copy()
    if not transition_view.empty and "TransitionScore" in transition_view.columns:
        transition_view = transition_view.sort_values("TransitionScore", ascending=False)
        fig_transition = px.bar(
            transition_view,
            x="Region",
            y="TransitionScore",
            color="Priority",
            title="5) Transition Score (Priority Colors)",
            labels={"TransitionScore": "Transition Score"},
            color_discrete_map=priority_color_map,
        )
        st.plotly_chart(fig_transition, width="stretch")
    else:
        st.info("Transition score data not available.")

with row2_col3:
    if shift_df is not None and {"Region", "Changed"}.issubset(shift_df.columns):
        impact = shift_df[shift_df["Region"].isin(selected_regions)].copy()
        impact = impact[impact["Region"] != "Canada"]
        impact["Changed"] = impact["Changed"].astype(str).str.lower().isin(["true", "1", "yes"])
        impact = impact.groupby("Changed", as_index=False).size()
        impact["Outcome"] = impact["Changed"].map({True: "Shifted Class", False: "No Change"})
        fig_impact = px.bar(
            impact,
            x="Outcome",
            y="size",
            color="Outcome",
            title="6) Stress-Test Impact",
            labels={"size": "Number of Regions"},
            color_discrete_map={"Shifted Class": "#d62728", "No Change": "#2ca02c"},
        )
        st.plotly_chart(fig_impact, width="stretch")
    else:
        st.info("Stress-test artifact not found in output/.")

st.markdown("---")
year_span = f"{min(all_years)}-{max(all_years)}"
st.caption(
    f"Data years: {year_span} | Latest dashboard year: {latest_year} | "
    "Decision notes: PriorityScore guides intervention urgency; TransitionScore reflects progress; results are associative, not causal."
)

if forecast_df is not None:
    st.caption("Forecast file detected (output/forecast_5y_region_source.csv).")
