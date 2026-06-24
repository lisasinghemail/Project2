import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Isla Coralina Relief Dashboard",
    page_icon="🌴",
    layout="wide"
)

st.title("Isla Coralina Relief Operations Dashboard")


def prepare_relief_data(df):
    """Create the same derived columns used in Problem 2."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["supply_gap"] = df["quantity_requested"] - df["quantity_delivered"]
    df["fulfillment_rate"] = df["quantity_delivered"] / df["quantity_requested"]
    df["shortfall_pct"] = df["supply_gap"] / df["quantity_requested"]
    df["capacity_utilization"] = df["population_at_center"] / df["center_capacity"]
    df["day_after_hurricane"] = (df["date"] - df["date"].min()).dt.days + 1
    return df


def show_kpi(label, value):
    st.metric(label, value)


relief_df = pd.read_csv("isla_coralina_relief_operations.csv")
infra_df = pd.read_csv("isla_coralina_infrastructure.csv")


#interactive filters
st.sidebar.header("Filters")

municipalities = sorted(relief_df["municipality"].dropna().unique())
selected_municipalities = st.sidebar.multiselect(
    "Municipality",
    options=municipalities,
    default=municipalities
)

date= pd.to_datetime(relief_df["date"], errors="coerce")
min_date = date.min().date()
max_date = date.max().date()
selected_dates = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date, end_date = min_date, max_date

filtered_relief = relief_df[
    (relief_df["municipality"].isin(selected_municipalities))
    & (date.dt.date >= start_date)
    & (date.dt.date <= end_date)
].copy()

if infra_df is not None:
    filtered_infra = infra_df[infra_df["municipality"].isin(selected_municipalities)].copy()
else:
    filtered_infra = None

#summary/ KPI
st.subheader("Operational Summary")

if filtered_relief.empty:
    st.warning("No relief delivery records match the selected filters.")
    st.stop()

under_80_pct = (filtered_relief["fulfillment_rate"] < 0.80).mean() * 100
weighted_fulfillment = filtered_relief["quantity_delivered"].sum() / filtered_relief["quantity_requested"].sum()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    show_kpi("Total population at centers", f"{filtered_relief['population_at_center'].sum():,.0f}")
with kpi2:
    show_kpi("Average delivery delay", f"{filtered_relief['delivery_delay_hours'].mean():.1f} hrs")
with kpi3:
    show_kpi("Deliveries below 80% fulfilled", f"{under_80_pct:.1f}%")
with kpi4:
    show_kpi("Weighted fulfillment rate", f"{weighted_fulfillment:.1%}")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["Infrastructure Status", "Relief Distribution"])

with tab1:
    st.header("Infrastructure Status")

    if filtered_infra is None:
        st.warning(
            "Infrastructure CSV was not found. Upload isla_coralina_infrastructure.csv in the sidebar "
            "or place it in the same folder as this app."
        )
    elif filtered_infra.empty:
        st.warning("No infrastructure records match the selected municipality filter.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            show_kpi("Facilities", f"{len(filtered_infra):,.0f}")
        with c2:
            non_op = (filtered_infra["operational_status"] == "Non-Operational").sum()
            show_kpi("Non-operational facilities", f"{non_op:,.0f}")
        with c3:
            show_kpi("Average damage severity", f"{filtered_infra['damage_severity'].mean():.2f}")

        st.markdown("### Facility Map")
        map_fig = px.scatter_map(
            filtered_infra,
            lat="latitude",
            lon="longitude",
            color="operational_status",
            size="damage_severity",
            hover_name="facility_name",
            hover_data=["municipality", "facility_type", "road_access", "population_served"],
            zoom=8,
            height=520,
            title="Infrastructure Status and Damage Severity"
        )
        map_fig.update_layout(map_style="open-street-map", margin={"r": 0, "t": 45, "l": 0, "b": 0})
        st.plotly_chart(map_fig, use_container_width=True)

        st.markdown("### Operational Status by Municipality")
        status_summary = (
            filtered_infra.groupby(["municipality", "operational_status"])
            .size()
            .reset_index(name="facility_count")
        )
        status_fig = px.bar(
            status_summary,
            x="municipality",
            y="facility_count",
            color="operational_status",
            title="Facility Operational Status by Municipality",
            labels={"facility_count": "Number of facilities", "municipality": "Municipality"},
            barmode="stack"
        )
        st.plotly_chart(status_fig, use_container_width=True)

        st.info(
            "Action: use the map to identify damaged and non-operational facilities. "
            "Municipalities with many non-operational facilities should receive repair crews, backup power, "
            "and alternate service coverage first."
        )

with tab2:
    st.header("Relief Distribution")

    # Chart from Problem 2: total supply gap by municipality
    municipality_summary = (
        filtered_relief.groupby("municipality")
        .agg(
            requested=("quantity_requested", "sum"),
            delivered=("quantity_delivered", "sum"),
            gap=("supply_gap", "sum"),
            avg_delay_hours=("delivery_delay_hours", "mean")
        )
        .reset_index()
    )
    municipality_summary["weighted_fulfillment"] = (
        municipality_summary["delivered"] / municipality_summary["requested"]
    )
    municipality_summary = municipality_summary.sort_values("gap", ascending=False)

    st.markdown("### Total Supply Gap by Municipality")
    gap_fig = px.bar(
        municipality_summary,
        x="municipality",
        y="gap",
        title="Total Supply Gap by Municipality",
        labels={"municipality": "Municipality", "gap": "Requested Minus Delivered Quantity"}
    )
    st.plotly_chart(gap_fig, use_container_width=True)

    st.markdown("### Weighted Fulfillment Rate by Municipality")
    fulfill_fig = px.bar(
        municipality_summary,
        x="municipality",
        y="weighted_fulfillment",
        title="Weighted Fulfillment Rate by Municipality",
        labels={"municipality": "Municipality", "weighted_fulfillment": "Delivered / Requested"}
    )
    fulfill_fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fulfill_fig, use_container_width=True)

    st.markdown("### Daily Fulfillment Trend")
    daily_summary = (
        filtered_relief.groupby("date")
        .agg(requested=("quantity_requested", "sum"), delivered=("quantity_delivered", "sum"))
        .reset_index()
    )
    daily_summary["weighted_fulfillment"] = daily_summary["delivered"] / daily_summary["requested"]
    daily_fig = px.line(
        daily_summary,
        x="date",
        y="weighted_fulfillment",
        markers=True,
        title="Weighted Fulfillment Rate Over Time",
        labels={"date": "Date", "weighted_fulfillment": "Delivered / Requested"}
    )
    daily_fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(daily_fig, use_container_width=True)

    st.markdown("### Top 10 Distribution Centers by Supply Gap")
    center_summary = (
        filtered_relief.groupby(["distribution_center_id", "municipality"])
        .agg(gap=("supply_gap", "sum"))
        .reset_index()
        .sort_values("gap", ascending=False)
        .head(10)
    )
    center_summary["center_label"] = (
        center_summary["distribution_center_id"].astype(str)
        + " ("
        + center_summary["municipality"]
        + ")"
    )
    center_fig = px.bar(
        center_summary.sort_values("gap"),
        x="gap",
        y="center_label",
        orientation="h",
        title="Top 10 Distribution Centers by Supply Gap",
        labels={"gap": "Requested Minus Delivered Quantity", "center_label": "Distribution Center"}
    )
    st.plotly_chart(center_fig, use_container_width=True)

    st.markdown("### Delivery Delay Box Plot")
    delay_fig = px.box(
        filtered_relief,
        x="municipality",
        y="delivery_delay_hours",
        title="Delivery Delay by Municipality",
        labels={"municipality": "Municipality", "delivery_delay_hours": "Delay Hours"}
    )
    st.plotly_chart(delay_fig, use_container_width=True)

    worst_gap_row = municipality_summary.iloc[0]
    st.info(
        f"Action: prioritize {worst_gap_row['municipality']} because it has the largest filtered supply gap "
        f"({worst_gap_row['gap']:,.0f} units). Watch municipalities with low fulfillment rates and long delay "
        "distributions because they indicate access or routing problems."
    )
