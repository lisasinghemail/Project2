import os
from typing import Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Isla Coralina Relief Operations Dashboard",
    page_icon="🛰️",
    layout="wide",
)

RELIEF_FILE = "isla_coralina_relief_operations.csv"
INFRA_FILE = "isla_coralina_infrastructure.csv"


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_csv(uploaded_file, default_path: str, label: str) -> Optional[pd.DataFrame]:
    """
    Load a CSV from the local folder first, then from Streamlit uploader.
    This lets the dashboard work both for assignment submission and ad hoc testing.
    """
    if os.path.exists(default_path):
        return load_csv_from_path(default_path)

    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    st.warning(f"{label} was not found. Place `{default_path}` beside `app.py` or upload it in the sidebar.")
    return None


def to_numeric_if_exists(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_relief(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "quantity_requested",
        "quantity_delivered",
        "population_at_center",
        "delivery_delay_hours",
        "number_of_aid_workers",
        "center_capacity",
    ]
    df = to_numeric_if_exists(df, numeric_cols)

    df["fulfillment_rate"] = np.where(
        df["quantity_requested"] > 0,
        df["quantity_delivered"] / df["quantity_requested"],
        np.nan,
    )

    df["fulfillment_pct"] = df["fulfillment_rate"] * 100
    df["unmet_quantity"] = df["quantity_requested"] - df["quantity_delivered"]
    df["unmet_quantity"] = df["unmet_quantity"].clip(lower=0)

    df["center_utilization"] = np.where(
        df["center_capacity"] > 0,
        df["population_at_center"] / df["center_capacity"],
        np.nan,
    )
    df["center_utilization_pct"] = df["center_utilization"] * 100

    df["below_80_pct"] = df["fulfillment_rate"] < 0.80

    return df


def clean_infrastructure(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None

    df = df.copy()
    df = to_numeric_if_exists(df, ["latitude", "longitude", "damage_severity", "population_served"])

    if "date_last_update" in df.columns:
        df["date_last_update"] = pd.to_datetime(df["date_last_update"], errors="coerce")

    if "facility_type" in df.columns:
        critical_terms = [
            "hospital",
            "clinic",
            "health",
            "shelter",
            "water",
            "power",
            "substation",
            "bridge",
            "distribution",
            "fire",
            "communications",
            "tower",
        ]
        pattern = "|".join(critical_terms)
        df["is_critical_facility"] = df["facility_type"].astype(str).str.lower().str.contains(pattern, na=False)
    else:
        df["is_critical_facility"] = True

    if "operational_status" in df.columns:
        operational_words = ["operational", "open", "functional", "active", "available"]
        operational_pattern = "|".join(operational_words)
        df["is_operational"] = (
            df["operational_status"]
            .astype(str)
            .str.lower()
            .str.contains(operational_pattern, na=False)
        )
    else:
        df["is_operational"] = np.nan

    df["non_operational_critical"] = df["is_critical_facility"] & (df["is_operational"] == False)

    return df


def filter_dataframe(
    relief: pd.DataFrame,
    infra: Optional[pd.DataFrame],
    selected_municipalities: list[str],
    selected_supply_types: list[str],
    selected_transport_modes: list[str],
    selected_road_conditions: list[str],
    date_range,
):
    start_date, end_date = date_range

    filtered_relief = relief[
        (relief["municipality"].isin(selected_municipalities))
        & (relief["supply_type"].isin(selected_supply_types))
        & (relief["transport_mode"].isin(selected_transport_modes))
        & (relief["road_condition"].isin(selected_road_conditions))
        & (relief["date"].dt.date >= start_date)
        & (relief["date"].dt.date <= end_date)
    ].copy()

    filtered_infra = None
    if infra is not None and "municipality" in infra.columns:
        filtered_infra = infra[infra["municipality"].isin(selected_municipalities)].copy()
    elif infra is not None:
        filtered_infra = infra.copy()

    return filtered_relief, filtered_infra


def metric_number(value, digits=0, suffix=""):
    if pd.isna(value):
        return "N/A"
    if digits == 0:
        return f"{value:,.0f}{suffix}"
    return f"{value:,.{digits}f}{suffix}"


def safe_mean(series: pd.Series):
    series = pd.to_numeric(series, errors="coerce")
    if series.dropna().empty:
        return np.nan
    return series.mean()


def build_daily_summary(relief: pd.DataFrame) -> pd.DataFrame:
    daily = (
        relief.groupby("date", as_index=False)
        .agg(
            deliveries=("delivery_id", "count"),
            requested=("quantity_requested", "sum"),
            delivered=("quantity_delivered", "sum"),
            avg_delay=("delivery_delay_hours", "mean"),
            under_80=("below_80_pct", "mean"),
        )
    )
    daily["fulfillment_pct"] = np.where(daily["requested"] > 0, daily["delivered"] / daily["requested"] * 100, np.nan)
    daily["under_80_pct"] = daily["under_80"] * 100
    return daily


def get_top_operational_text(relief: pd.DataFrame, infra: Optional[pd.DataFrame]) -> str:
    if relief.empty:
        return "No records match the selected filters. Broaden the filters before making an operational decision."

    avg_fulfillment = safe_mean(relief["fulfillment_pct"])
    avg_delay = safe_mean(relief["delivery_delay_hours"])
    under_80 = relief["below_80_pct"].mean() * 100

    by_municipality = (
        relief.groupby("municipality", as_index=False)
        .agg(
            requested=("quantity_requested", "sum"),
            delivered=("quantity_delivered", "sum"),
            avg_delay=("delivery_delay_hours", "mean"),
        )
    )
    by_municipality["fulfillment_pct"] = by_municipality["delivered"] / by_municipality["requested"] * 100
    weakest_muni = by_municipality.sort_values(["fulfillment_pct", "avg_delay"], ascending=[True, False]).iloc[0]

    by_supply = (
        relief.groupby("supply_type", as_index=False)
        .agg(
            requested=("quantity_requested", "sum"),
            delivered=("quantity_delivered", "sum"),
        )
    )
    by_supply["fulfillment_pct"] = by_supply["delivered"] / by_supply["requested"] * 100
    weakest_supply = by_supply.sort_values("fulfillment_pct").iloc[0]

    if infra is not None and not infra.empty and "non_operational_critical" in infra.columns:
        non_op = int(infra["non_operational_critical"].sum())
        infra_sentence = (
            f"The filtered infrastructure view shows **{non_op:,} non-operational critical facilities**. "
            "Treat these as constraints when assigning convoy routes and staging backup services."
        )
    else:
        infra_sentence = (
            "Infrastructure records are not currently loaded, so the dashboard cannot yet confirm which "
            "critical facilities are non-operational."
        )

    return f"""
    **Operational interpretation.** In the current filtered view, the average fulfillment rate is
    **{avg_fulfillment:,.1f}%**, average delay is **{avg_delay:,.1f} hours**, and
    **{under_80:,.1f}%** of deliveries fulfill less than 80% of the request. The weakest municipality
    by fulfillment is **{weakest_muni['municipality']}** at **{weakest_muni['fulfillment_pct']:,.1f}%**.
    The weakest supply category is **{weakest_supply['supply_type']}** at
    **{weakest_supply['fulfillment_pct']:,.1f}%**.

    **Suggested action.** Prioritize additional loads for the lowest-fulfillment municipality and supply
    category first. Use trucks where road access is available, but shift delayed or inaccessible routes to
    boats or helicopters only where the urgency justifies the lower fulfillment efficiency. {infra_sentence}
    """


# ------------------------------------------------------------
# Sidebar data loading
# ------------------------------------------------------------
st.title("Isla Coralina Relief Operations Dashboard")
st.caption("Operational view for infrastructure status, supply fulfillment, delivery delays, and priority response areas.")

with st.sidebar:
    st.header("Data files")
    relief_upload = st.file_uploader("Relief operations CSV", type=["csv"])
    infra_upload = st.file_uploader("Infrastructure CSV", type=["csv"])

relief_raw = load_csv(relief_upload, RELIEF_FILE, "Relief operations CSV")
infra_raw = load_csv(infra_upload, INFRA_FILE, "Infrastructure CSV")

if relief_raw is None:
    st.stop()

relief = clean_relief(relief_raw)
infra = clean_infrastructure(infra_raw)

# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------
municipality_values = sorted(relief["municipality"].dropna().unique().tolist())
if infra is not None and "municipality" in infra.columns:
    municipality_values = sorted(set(municipality_values).union(set(infra["municipality"].dropna().unique().tolist())))

supply_values = sorted(relief["supply_type"].dropna().unique().tolist())
transport_values = sorted(relief["transport_mode"].dropna().unique().tolist())
road_values = sorted(relief["road_condition"].dropna().unique().tolist())

min_date = relief["date"].min().date()
max_date = relief["date"].max().date()

with st.sidebar:
    st.header("Interactive filters")

    selected_municipalities = st.multiselect(
        "Municipality",
        municipality_values,
        default=municipality_values,
    )

    date_range = st.slider(
        "Delivery date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
    )

    selected_supply_types = st.multiselect(
        "Supply type",
        supply_values,
        default=supply_values,
    )

    selected_transport_modes = st.multiselect(
        "Transport mode",
        transport_values,
        default=transport_values,
    )

    selected_road_conditions = st.multiselect(
        "Road access / condition",
        road_values,
        default=road_values,
    )

if not selected_municipalities:
    st.error("Select at least one municipality.")
    st.stop()
if not selected_supply_types:
    st.error("Select at least one supply type.")
    st.stop()
if not selected_transport_modes:
    st.error("Select at least one transport mode.")
    st.stop()
if not selected_road_conditions:
    st.error("Select at least one road condition.")
    st.stop()

relief_f, infra_f = filter_dataframe(
    relief,
    infra,
    selected_municipalities,
    selected_supply_types,
    selected_transport_modes,
    selected_road_conditions,
    date_range,
)

# ------------------------------------------------------------
# KPI summary
# ------------------------------------------------------------
if infra_f is not None and "population_served" in infra_f.columns:
    total_population_served = infra_f["population_served"].sum()
    population_label = "Population served by infrastructure"
else:
    total_population_served = (
        relief_f.groupby("distribution_center_id")["population_at_center"].max().sum()
        if not relief_f.empty else np.nan
    )
    population_label = "Population at active centers"

average_delay = safe_mean(relief_f["delivery_delay_hours"]) if not relief_f.empty else np.nan
under_80_pct = relief_f["below_80_pct"].mean() * 100 if not relief_f.empty else np.nan

if infra_f is not None and "non_operational_critical" in infra_f.columns:
    non_operational_critical = infra_f["non_operational_critical"].sum()
else:
    non_operational_critical = np.nan

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(population_label, metric_number(total_population_served))
kpi2.metric("Average delivery delay", metric_number(average_delay, digits=1, suffix=" hrs"))
kpi3.metric("Deliveries below 80% fulfillment", metric_number(under_80_pct, digits=1, suffix="%"))
kpi4.metric("Non-operational critical facilities", metric_number(non_operational_critical))

st.markdown(get_top_operational_text(relief_f, infra_f))

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab_infra, tab_relief, tab_priority = st.tabs(
    ["Infrastructure Status", "Relief Distribution Performance", "Operational Priorities"]
)

# ------------------------------------------------------------
# Tab 1: Infrastructure
# ------------------------------------------------------------
with tab_infra:
    st.subheader("Critical infrastructure status")

    if infra_f is None or infra_f.empty:
        st.info(
            "Upload or place `isla_coralina_infrastructure.csv` beside `app.py` to activate "
            "the infrastructure maps and facility status charts."
        )
    else:
        map_cols = {"latitude", "longitude"}
        if map_cols.issubset(infra_f.columns):
            hover_cols = [
                c for c in [
                    "facility_id",
                    "facility_type",
                    "municipality",
                    "operational_status",
                    "damage_severity",
                    "population_served",
                    "road_access",
                    "has_generator",
                ] if c in infra_f.columns
            ]

            fig_map = px.scatter_mapbox(
                infra_f.dropna(subset=["latitude", "longitude"]),
                lat="latitude",
                lon="longitude",
                color="operational_status" if "operational_status" in infra_f.columns else "municipality",
                size="damage_severity" if "damage_severity" in infra_f.columns else None,
                hover_name="facility_name" if "facility_name" in infra_f.columns else None,
                hover_data=hover_cols,
                zoom=9,
                height=560,
                title="Infrastructure map by operational status and damage severity",
            )
            fig_map.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 45, "l": 0, "b": 0})
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("The infrastructure file must include `latitude` and `longitude` columns to draw the map.")

        col1, col2 = st.columns(2)

        with col1:
            if {"municipality", "operational_status"}.issubset(infra_f.columns):
                status_counts = (
                    infra_f.groupby(["municipality", "operational_status"], as_index=False)
                    .size()
                    .rename(columns={"size": "facility_count"})
                )
                fig_status = px.bar(
                    status_counts,
                    x="municipality",
                    y="facility_count",
                    color="operational_status",
                    barmode="stack",
                    title="Facility status by municipality",
                    labels={
                        "municipality": "Municipality",
                        "facility_count": "Number of facilities",
                        "operational_status": "Operational status",
                    },
                )
                st.plotly_chart(fig_status, use_container_width=True)

        with col2:
            if {"municipality", "damage_severity"}.issubset(infra_f.columns):
                damage_summary = (
                    infra_f.groupby("municipality", as_index=False)
                    .agg(avg_damage=("damage_severity", "mean"))
                    .sort_values("avg_damage", ascending=False)
                )
                fig_damage = px.bar(
                    damage_summary,
                    x="municipality",
                    y="avg_damage",
                    title="Average damage severity by municipality",
                    labels={
                        "municipality": "Municipality",
                        "avg_damage": "Average damage severity, 1 low to 5 high",
                    },
                )
                st.plotly_chart(fig_damage, use_container_width=True)

        if {"municipality", "non_operational_critical"}.issubset(infra_f.columns):
            non_op_by_muni = (
                infra_f.groupby("municipality", as_index=False)
                .agg(non_operational_critical=("non_operational_critical", "sum"))
                .sort_values("non_operational_critical", ascending=False)
            )
            fig_nonop = px.bar(
                non_op_by_muni,
                x="municipality",
                y="non_operational_critical",
                title="Non-operational critical facilities by municipality",
                labels={
                    "municipality": "Municipality",
                    "non_operational_critical": "Non-operational critical facilities",
                },
            )
            st.plotly_chart(fig_nonop, use_container_width=True)

# ------------------------------------------------------------
# Tab 2: Relief distribution
# ------------------------------------------------------------
with tab_relief:
    st.subheader("Relief distribution performance")

    if relief_f.empty:
        st.info("No relief delivery records match the current filters.")
    else:
        daily = build_daily_summary(relief_f)

        fig_daily = px.line(
            daily,
            x="date",
            y=["deliveries", "fulfillment_pct", "avg_delay"],
            markers=True,
            title="Daily operational trend: volume, fulfillment, and delay",
            labels={
                "date": "Date",
                "value": "Value",
                "variable": "Metric",
            },
        )
        st.plotly_chart(fig_daily, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            supply_summary = (
                relief_f.groupby("supply_type", as_index=False)
                .agg(
                    requested=("quantity_requested", "sum"),
                    delivered=("quantity_delivered", "sum"),
                    avg_delay=("delivery_delay_hours", "mean"),
                    under_80=("below_80_pct", "mean"),
                )
            )
            supply_summary["fulfillment_pct"] = supply_summary["delivered"] / supply_summary["requested"] * 100
            supply_summary = supply_summary.sort_values("fulfillment_pct")

            fig_supply = px.bar(
                supply_summary,
                x="fulfillment_pct",
                y="supply_type",
                orientation="h",
                title="Fulfillment rate by supply type",
                labels={
                    "fulfillment_pct": "Fulfillment rate (%)",
                    "supply_type": "Supply type",
                },
                hover_data=["requested", "delivered", "avg_delay"],
            )
            fig_supply.add_vline(x=80, line_dash="dash", annotation_text="80% target")
            st.plotly_chart(fig_supply, use_container_width=True)

        with col2:
            fig_box = px.box(
                relief_f,
                x="transport_mode",
                y="delivery_delay_hours",
                points="outliers",
                title="Delivery delay distribution by transport mode",
                labels={
                    "transport_mode": "Transport mode",
                    "delivery_delay_hours": "Delay hours",
                },
            )
            st.plotly_chart(fig_box, use_container_width=True)

        muni_supply = (
            relief_f.groupby(["municipality", "supply_type"], as_index=False)
            .agg(
                requested=("quantity_requested", "sum"),
                delivered=("quantity_delivered", "sum"),
            )
        )
        muni_supply["fulfillment_pct"] = muni_supply["delivered"] / muni_supply["requested"] * 100
        heatmap_data = muni_supply.pivot(index="municipality", columns="supply_type", values="fulfillment_pct")

        fig_heatmap = px.imshow(
            heatmap_data,
            text_auto=".1f",
            aspect="auto",
            title="Fulfillment heatmap by municipality and supply type",
            labels={
                "x": "Supply type",
                "y": "Municipality",
                "color": "Fulfillment (%)",
            },
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        road_summary = (
            relief_f.groupby("road_condition", as_index=False)
            .agg(
                deliveries=("delivery_id", "count"),
                avg_delay=("delivery_delay_hours", "mean"),
                avg_fulfillment=("fulfillment_pct", "mean"),
            )
        )

        fig_road = px.scatter(
            road_summary,
            x="avg_delay",
            y="avg_fulfillment",
            size="deliveries",
            hover_name="road_condition",
            title="Road condition effect on delay and fulfillment",
            labels={
                "avg_delay": "Average delay hours",
                "avg_fulfillment": "Average fulfillment (%)",
                "deliveries": "Deliveries",
            },
        )
        st.plotly_chart(fig_road, use_container_width=True)

# ------------------------------------------------------------
# Tab 3: Operational priorities
# ------------------------------------------------------------
with tab_priority:
    st.subheader("Priority centers and recommended action")

    if relief_f.empty:
        st.info("No relief delivery records match the current filters.")
    else:
        center_priority = (
            relief_f.groupby(["distribution_center_id", "municipality"], as_index=False)
            .agg(
                delivery_count=("delivery_id", "count"),
                requested=("quantity_requested", "sum"),
                delivered=("quantity_delivered", "sum"),
                unmet_quantity=("unmet_quantity", "sum"),
                avg_delay_hours=("delivery_delay_hours", "mean"),
                population_at_center=("population_at_center", "max"),
                center_capacity=("center_capacity", "max"),
                aid_workers=("number_of_aid_workers", "mean"),
            )
        )

        center_priority["fulfillment_pct"] = np.where(
            center_priority["requested"] > 0,
            center_priority["delivered"] / center_priority["requested"] * 100,
            np.nan,
        )
        center_priority["center_utilization_pct"] = np.where(
            center_priority["center_capacity"] > 0,
            center_priority["population_at_center"] / center_priority["center_capacity"] * 100,
            np.nan,
        )

        # Normalize selected factors to produce an easy-to-rank priority score.
        for col in ["unmet_quantity", "avg_delay_hours", "center_utilization_pct"]:
            max_val = center_priority[col].max()
            if pd.isna(max_val) or max_val == 0:
                center_priority[f"{col}_score"] = 0
            else:
                center_priority[f"{col}_score"] = center_priority[col] / max_val

        center_priority["priority_score"] = (
            0.50 * center_priority["unmet_quantity_score"]
            + 0.30 * center_priority["avg_delay_hours_score"]
            + 0.20 * center_priority["center_utilization_pct_score"]
        ) * 100

        # Add infrastructure details if the two files share facility IDs.
        if infra_f is not None and "facility_id" in infra_f.columns:
            merge_cols = [
                c for c in [
                    "facility_id",
                    "facility_name",
                    "facility_type",
                    "operational_status",
                    "damage_severity",
                    "latitude",
                    "longitude",
                    "road_access",
                    "has_generator",
                ] if c in infra_f.columns
            ]
            infra_lookup = infra_f[merge_cols].drop_duplicates(subset=["facility_id"])
            center_priority = center_priority.merge(
                infra_lookup,
                left_on="distribution_center_id",
                right_on="facility_id",
                how="left",
            )

        center_priority = center_priority.sort_values(
            ["priority_score", "unmet_quantity", "avg_delay_hours"],
            ascending=False,
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            fig_priority = px.scatter(
                center_priority,
                x="fulfillment_pct",
                y="avg_delay_hours",
                size="unmet_quantity",
                color="municipality",
                hover_name="distribution_center_id",
                hover_data=[
                    "delivery_count",
                    "requested",
                    "delivered",
                    "population_at_center",
                    "center_utilization_pct",
                    "priority_score",
                ],
                title="Priority centers: low fulfillment, high delay, high unmet quantity",
                labels={
                    "fulfillment_pct": "Fulfillment rate (%)",
                    "avg_delay_hours": "Average delay hours",
                    "unmet_quantity": "Unmet quantity",
                },
            )
            fig_priority.add_vline(x=80, line_dash="dash", annotation_text="80% target")
            st.plotly_chart(fig_priority, use_container_width=True)

        with col2:
            if {"latitude", "longitude"}.issubset(center_priority.columns):
                center_map = center_priority.dropna(subset=["latitude", "longitude"])
                if not center_map.empty:
                    fig_center_map = px.scatter_mapbox(
                        center_map,
                        lat="latitude",
                        lon="longitude",
                        size="priority_score",
                        color="municipality",
                        hover_name="facility_name" if "facility_name" in center_map.columns else "distribution_center_id",
                        hover_data=[
                            c for c in [
                                "distribution_center_id",
                                "operational_status",
                                "damage_severity",
                                "fulfillment_pct",
                                "avg_delay_hours",
                                "unmet_quantity",
                                "priority_score",
                            ] if c in center_map.columns
                        ],
                        zoom=9,
                        height=520,
                        title="Priority distribution centers on map",
                    )
                    fig_center_map.update_layout(
                        mapbox_style="open-street-map",
                        margin={"r": 0, "t": 45, "l": 0, "b": 0},
                    )
                    st.plotly_chart(fig_center_map, use_container_width=True)
                else:
                    st.info("No matching distribution centers have latitude and longitude values.")
            else:
                st.info(
                    "Upload the infrastructure file to map priority distribution centers. "
                    "The dashboard can still rank centers from relief operations data."
                )

        st.markdown("### Top priority centers")
        display_cols = [
            "distribution_center_id",
            "facility_name",
            "municipality",
            "facility_type",
            "operational_status",
            "damage_severity",
            "delivery_count",
            "requested",
            "delivered",
            "unmet_quantity",
            "fulfillment_pct",
            "avg_delay_hours",
            "population_at_center",
            "center_capacity",
            "center_utilization_pct",
            "aid_workers",
            "priority_score",
        ]
        display_cols = [c for c in display_cols if c in center_priority.columns]

        st.dataframe(
            center_priority[display_cols].head(15).style.format(
                {
                    "fulfillment_pct": "{:.1f}%",
                    "avg_delay_hours": "{:.1f}",
                    "center_utilization_pct": "{:.1f}%",
                    "aid_workers": "{:.1f}",
                    "priority_score": "{:.1f}",
                }
            ),
            use_container_width=True,
        )

       
