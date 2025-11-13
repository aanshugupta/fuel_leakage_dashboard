import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client

# -----------------------------------------------------------------
# SUPABASE SETTINGS
# -----------------------------------------------------------------
SUPABASE_URL = "https://pyanhlpwloofwzpulcpi.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB5YW5obHB3bG9vZnd6cHVsY3BpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI3NjQyMzcsImV4cCI6MjA3ODM0MDIzN30.vUydKFP8kPOudO1bup4z1JYCYrWAMrI6RZol0pvQiCw"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------------------------------------------
# STREAMLIT UI SETTINGS
# -----------------------------------------------------------------
st.set_page_config(page_title="Fuel Leakage Dashboard", layout="wide")
st.title("⛽ Fuel Leakage Detection, Efficiency & Profit/Loss Dashboard")

upload = st.sidebar.file_uploader("Upload processed_trips CSV or Excel", type=["csv", "xlsx"])

# -----------------------------------------------------------------
# WHEN USER UPLOADS FILE
# -----------------------------------------------------------------
if upload:
    if upload.name.endswith(".xlsx"):
        df = pd.read_excel(upload)
    else:
        df = pd.read_csv(upload)

    st.write("Uploaded File Preview:", df.head())

    # Required columns
    req = ["trip_id", "distance_km", "actual_fuel_liters", "diesel_price_per_liter"]
    for c in req:
        if c not in df.columns:
            df[c] = 0

    df[req] = df[req].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Default diesel price
    df.loc[df["diesel_price_per_liter"] == 0, "diesel_price_per_liter"] = 95

    # Revenue logic
    df["expected_revenue"] = df["distance_km"] * 150
    df["fuel_cost"] = df["actual_fuel_liters"] * df["diesel_price_per_liter"]
    df["profit_loss"] = df["expected_revenue"] - df["fuel_cost"]
    df["pnl_status"] = np.where(df["profit_loss"] > 0, "Profit", "Loss")

    st.success("Data processed successfully!")
    st.dataframe(df.head())

    # Upload to Supabase
    try:
        supabase.table("trip_data").delete().neq("trip_id", "").execute()
        supabase.table("trip_data").insert(df.to_dict(orient="records")).execute()
        st.success("Data uploaded to Supabase successfully!")
    except Exception as e:
        st.error(f"Upload error: {e}")

    # Dashboard
    st.subheader("Dashboard Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trips", len(df))
    c2.metric("Total Profit", f"{df[df.pnl_status=='Profit']['profit_loss'].sum():,.0f}")
    c3.metric("Total Loss", f"{df[df.pnl_status=='Loss']['profit_loss'].sum():,.0f}")

    st.plotly_chart(px.bar(df, x="trip_id", y="profit_loss", color="pnl_status",
                           title="Trip Profit/Loss Chart"),
                    use_container_width=True)

else:
    st.info("Please upload processed_trips file to continue.")
