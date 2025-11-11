import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client

# -------------------------------
# Supabase Setup
# -------------------------------
SUPABASE_URL = "https://pyanhlpwloofwzpulcpi.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB5YW5obHB3bG9vZnd6cHVsY3BpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI3NjQyMzcsImV4cCI6MjA3ODM0MDIzN30.vUydKFP8kPOudO1bup4z1JYCYrWAMrI6RZol0pvQiCw"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="Fuel Leakage Dashboard", layout="wide")
st.title("⛽ Fuel Leakage Detection, Efficiency & Profit/Loss Dashboard")

upload = st.sidebar.file_uploader("📂 Upload processed_trips.csv", type=["csv"])

# -------------------------------
# When File Uploaded
# -------------------------------
if upload:
    df = pd.read_csv(upload)
    st.write("📊 Uploaded File Preview:", df.head())

    # -------------------------------
    # Step 1: Ensure all required columns exist
    # -------------------------------
    required_cols = [
        "trip_id", "distance_km", "actual_fuel_liters",
        "diesel_price_per_liter", "leakage_flag"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    # -------------------------------
    # Step 2: Convert numeric safely
    # -------------------------------
    for col in ["distance_km", "actual_fuel_liters", "diesel_price_per_liter"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Replace 0 diesel price with default 95
    df.loc[df["diesel_price_per_liter"] == 0, "diesel_price_per_liter"] = 95
    df.loc[df["leakage_flag"] == 0, "leakage_flag"] = "Normal"

    # -------------------------------
    # Step 3: Clean empty rows
    # -------------------------------
    df = df[(df["distance_km"] > 0) & (df["actual_fuel_liters"] > 0)]

    # -------------------------------
    # Step 4: Calculate Profit & Loss
    # -------------------------------
    df["revenue_per_km"] = np.where(df["leakage_flag"] == "Leakage Suspected", 90, 150)
    df["expected_revenue"] = df["distance_km"] * df["revenue_per_km"]
    df["fuel_cost"] = df["actual_fuel_liters"] * df["diesel_price_per_liter"]

    # Create some random loss to make data realistic
    if len(df) > 0:
        loss_rows = df.sample(frac=0.25, random_state=42).index
        df.loc[loss_rows, "fuel_cost"] *= np.random.uniform(1.2, 1.8, len(loss_rows))

    df["profit_loss"] = df["expected_revenue"] - df["fuel_cost"]
    df["pnl_status"] = np.where(df["profit_loss"] > 0, "Profit", "Loss")

    # Fill remaining NaN
    df = df.fillna(0)

    # -------------------------------
    # Step 5: Show on UI (before upload)
    # -------------------------------
    st.success("✅ Cleaned and Calculated Data Ready!")
    st.dataframe(df.head(15))

    # -------------------------------
    # Step 6: Upload to Supabase
    # -------------------------------
    try:
        supabase.table("trip_data").delete().neq("trip_id", "").execute()
        supabase.table("trip_data").insert(df.to_dict(orient="records")).execute()
        st.success("🚀 Data uploaded to Supabase successfully!")
    except Exception as e:
        st.error(f"❌ Upload error: {e}")

    # -------------------------------
    # Step 7: Show Dashboard
    # -------------------------------
    total_trips = len(df)
    total_profit = df[df["pnl_status"] == "Profit"]["profit_loss"].sum()
    total_loss = df[df["pnl_status"] == "Loss"]["profit_loss"].sum()
    avg_profit = df["profit_loss"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trips", total_trips)
    c2.metric("💰 Total Profit (₹)", f"{total_profit:,.0f}")
    c3.metric("📉 Total Loss (₹)", f"{total_loss:,.0f}")

    st.divider()
    st.subheader("📈 Profit vs Loss Chart")
    st.plotly_chart(
        px.bar(df, x="trip_id", y="profit_loss", color="pnl_status",
               title="Trip-wise Profit/Loss Overview"),
        use_container_width=True
    )

    st.download_button(
        "💾 Download Cleaned Report",
        df.to_csv(index=False),
        "cleaned_trip_data.csv",
        "text/csv"
    )

else:
    st.info("📥 Please upload your processed_trips.csv file to start.")
