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

    # ✅ Step 1: Clean Data — Remove incomplete rows
    required_cols = ["distance_km", "actual_fuel_liters", "diesel_price_per_liter"]
    df = df.dropna(subset=required_cols)
    df[required_cols] = df[required_cols].replace("", 0)

    # ✅ Step 2: Calculate Profit/Loss before uploading
    revenue_per_km = 150
    df["expected_revenue"] = df["distance_km"] * revenue_per_km
    df["fuel_cost"] = df["actual_fuel_liters"] * df["diesel_price_per_liter"]
    df["profit_loss"] = df["expected_revenue"] - df["fuel_cost"]
    df["pnl_status"] = np.where(df["profit_loss"] > 0, "Profit", "Loss")

    try:
        # ✅ Step 3: Clear old records before new upload
        supabase.table("trip_data").delete().neq("trip_id", "").execute()

        # ✅ Step 4: Upload clean data (with PNL columns)
        supabase.table("trip_data").insert(df.to_dict(orient="records")).execute()
        st.success("✅ Data uploaded to Supabase successfully (with Profit/Loss)!")
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
avgv = df["variance_pct"].mean()
leak = df[df.leakage_flag == "Leakage Suspected"]
leak_l = max((leak.actual_fuel_liters - leak.expected_fuel_liters).sum(), 0)
leak_cost = max(leak["leakage_cost"].sum(), 0)
pct = len(leak) / total if total > 0 else 0

# Profit/Loss summary
total_profit = df["profit_loss"].sum()
avg_profit = df["profit_loss"].mean()

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
c6, c7 = st.columns(2)
c6.metric("💰 Total Profit/Loss (₹)", f"{total_profit:,.0f}")
c7.metric("📈 Avg Profit per Trip (₹)", f"{avg_profit:,.0f}")

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
    st.plotly_chart(px.bar(
        df, x="trip_id", y=["expected_fuel_liters", "actual_fuel_liters"],
        title="Actual vs Expected Fuel", barmode="group"
    ), use_container_width=True)

with tab2:
    st.plotly_chart(px.pie(
        df, names="leakage_flag", title="Leakage Categories"
    ), use_container_width=True)

with tab3:
    tmp = df.groupby("driver_id")["leakage_cost"].sum().reset_index()
    st.plotly_chart(px.bar(
        tmp, x="driver_id", y="leakage_cost", title="Leakage Cost per Driver"
    ), use_container_width=True)

with tab4:
    trend = df.groupby("trip_date")["variance_pct"].mean().reset_index()
    st.plotly_chart(px.line(
        trend, x="trip_date", y="variance_pct", title="Variance Trend"
    ), use_container_width=True)

with tab5:
    st.plotly_chart(px.scatter(
        df, x="distance_km", y="actual_fuel_liters", color="leakage_flag",
        title="Distance vs Fuel"
    ), use_container_width=True)

with tab6:
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
