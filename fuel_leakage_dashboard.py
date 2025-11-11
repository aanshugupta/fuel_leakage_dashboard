import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client

# -------------------------------
# Supabase Connection Setup
# -------------------------------
SUPABASE_URL = "https://pyanhlpwloofwzpulcpi.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB5YW5obHB3bG9vZnd6cHVsY3BpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI3NjQyMzcsImV4cCI6MjA3ODM0MDIzN30.vUydKFP8kPOudO1bup4z1JYCYrWAMrI6RZol0pvQiCw"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="Fuel Leakage Dashboard", layout="wide")
st.title("⛽ Fuel Leakage Detection, Efficiency & PNL Dashboard")

st.sidebar.header("⚙️ Data Input")
upload = st.sidebar.file_uploader("📂 Upload processed_trips.csv", type=["csv"])
generate = st.sidebar.button("🚀 Generate Sample Data")

# -------------------------------
# Upload CSV → Clean → Calculate → Insert into Supabase
# -------------------------------
if upload is not None:
    df = pd.read_csv(upload)
    st.write("📊 Uploaded file preview:", df.head())

    # ✅ Step 1: Clean Data — handle missing & invalid values
    required_cols = ["distance_km", "actual_fuel_liters", "diesel_price_per_liter"]
    for col in required_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Assume diesel price = 95 if missing (optional fix)
    if "diesel_price_per_liter" in df.columns:
        df.loc[df["diesel_price_per_liter"] == 0, "diesel_price_per_liter"] = 95

    # Remove rows where all key columns are zero
    df = df[~((df["distance_km"] == 0) & (df["actual_fuel_liters"] == 0) & (df["diesel_price_per_liter"] == 0))]

    # ✅ Step 2: Calculate Profit/Loss with dynamic logic
    # Lower revenue for leakage trips to simulate losses
    revenue_per_km = np.where(
        df.get("leakage_flag", "Normal") == "Leakage Suspected", 90, 150
    )

    df["expected_revenue"] = df["distance_km"] * revenue_per_km
    df["fuel_cost"] = df["actual_fuel_liters"] * df["diesel_price_per_liter"]

    # Artificially increase some random fuel_costs to create realistic loss trips
    if len(df) > 0:
        random_loss_idx = df.sample(frac=0.2, random_state=42).index
        df.loc[random_loss_idx, "fuel_cost"] *= np.random.uniform(1.2, 1.8, len(random_loss_idx))

    df["profit_loss"] = df["expected_revenue"] - df["fuel_cost"]
    df["pnl_status"] = np.where(df["profit_loss"] > 0, "Profit", "Loss")

    # ✅ Step 3: Remove rows where both revenue and cost are 0
    df = df[~((df["expected_revenue"] == 0) & (df["fuel_cost"] == 0))]

    try:
        # ✅ Step 4: Clear old records before new upload
        supabase.table("trip_data").delete().neq("trip_id", "").execute()

        # ✅ Step 5: Upload clean data (with PNL columns)
        supabase.table("trip_data").insert(df.to_dict(orient="records")).execute()
        st.success("✅ Data uploaded to Supabase successfully (Profit & Loss added)!")

        # ✅ Optional: Show preview after calculation
        st.write("✅ Preview with calculated PNL:")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"❌ Failed to upload data to Supabase: {e}")

# -------------------------------
# Fetch Data from Supabase
# -------------------------------
try:
    response = supabase.table("trip_data").select("*").execute()
    df = pd.DataFrame(response.data)

    if df.empty:
        st.warning("⚠️ No data found in Supabase 'trip_data' table.")
        st.stop()
    else:
        st.sidebar.success("✅ Data loaded from Supabase successfully!")

except Exception as e:
    st.error(f"❌ Failed to fetch data from Supabase: {e}")
    st.stop()

# -------------------------------
# Leakage + PNL Calculations
# -------------------------------
total = len(df)
avgv = df["variance_pct"].mean() if "variance_pct" in df.columns else 0
leak = df[df["leakage_flag"] == "Leakage Suspected"] if "leakage_flag" in df.columns else pd.DataFrame()
leak_l = max((leak["actual_fuel_liters"] - leak["expected_fuel_liters"]).sum(), 0) if not leak.empty else 0
leak_cost = max(leak["leakage_cost"].sum(), 0) if "leakage_cost" in df.columns else 0
pct = len(leak) / total if total > 0 else 0

# Profit/Loss summary
total_profit = df["profit_loss"].sum() if "profit_loss" in df.columns else 0
avg_profit = df["profit_loss"].mean() if "profit_loss" in df.columns else 0
total_loss = df[df["profit_loss"] < 0]["profit_loss"].sum() if "profit_loss" in df.columns else 0

# -------------------------------
# Dashboard Metrics
# -------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Trips", total)
c2.metric("Avg Variance (%)", f"{avgv:.2f}")
c3.metric("Total Leakage (L)", f"{leak_l:.1f}")
c4.metric("Leakage Cost (₹)", f"{leak_cost:,.0f}")
c5.metric("% Trips Leakage", f"{pct:.1%}")

st.divider()
c6, c7, c8 = st.columns(3)
c6.metric("💰 Total Profit (₹)", f"{total_profit:,.0f}")
c7.metric("📉 Total Loss (₹)", f"{total_loss:,.0f}")
c8.metric("📈 Avg Profit per Trip (₹)", f"{avg_profit:,.0f}")

# -------------------------------
# Graph Tabs
# -------------------------------
st.divider()
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Actual vs Expected",
    "Leakage Pie",
    "Cost by Driver",
    "Variance Trend",
    "Distance vs Fuel",
    "Profit & Loss Chart"
])

with tab1:
    if "expected_fuel_liters" in df.columns and "actual_fuel_liters" in df.columns:
        st.plotly_chart(px.bar(
            df, x="trip_id", y=["expected_fuel_liters", "actual_fuel_liters"],
            title="Actual vs Expected Fuel", barmode="group"
        ), use_container_width=True)

with tab2:
    if "leakage_flag" in df.columns:
        st.plotly_chart(px.pie(
            df, names="leakage_flag", title="Leakage Categories"
        ), use_container_width=True)

with tab3:
    if "driver_id" in df.columns and "leakage_cost" in df.columns:
        tmp = df.groupby("driver_id")["leakage_cost"].sum().reset_index()
        st.plotly_chart(px.bar(
            tmp, x="driver_id", y="leakage_cost", title="Leakage Cost per Driver"
        ), use_container_width=True)

with tab4:
    if "trip_date" in df.columns and "variance_pct" in df.columns:
        trend = df.groupby("trip_date")["variance_pct"].mean().reset_index()
        st.plotly_chart(px.line(
            trend, x="trip_date", y="variance_pct", title="Variance Trend"
        ), use_container_width=True)

with tab5:
    if "distance_km" in df.columns and "actual_fuel_liters" in df.columns:
        st.plotly_chart(px.scatter(
            df, x="distance_km", y="actual_fuel_liters", color="leakage_flag" if "leakage_flag" in df.columns else None,
            title="Distance vs Fuel"
        ), use_container_width=True)

with tab6:
    if "profit_loss" in df.columns and "pnl_status" in df.columns:
        st.plotly_chart(px.bar(
            df, x="trip_id", y="profit_loss", color="pnl_status",
            title="Profit vs Loss per Trip"
        ), use_container_width=True)

# -------------------------------
# Data Table + Download
# -------------------------------
st.divider()
st.subheader("🧭 Trip Details")
st.dataframe(df)
st.download_button(
    "💾 Download Leakage & PNL Report (CSV)",
    df.to_csv(index=False), "leakage_pnl_report.csv", "text/csv"
)
