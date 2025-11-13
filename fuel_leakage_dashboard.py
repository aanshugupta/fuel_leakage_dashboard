 import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai   # GEMINI AI

# -------------------------------------------------
# GEMINI AI CONFIG
# -------------------------------------------------
GEN_AI_KEY = "api key"   # <--- PUT YOUR API KEY
genai.configure(api_key=GEN_AI_KEY)
model = genai.GenerativeModel("gemini-pro")

# -------------------------------------------------
# Supabase Setup
# -------------------------------------------------
SUPABASE_URL = "https://pyanhlpwloofwzpulcpi.supabase.co"
SUPABASE_KEY = "YOUR_SUPABASE_ANON_KEY"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------------------------
# Streamlit Page Setup
# -------------------------------------------------
st.set_page_config(page_title="Fuel Leakage Dashboard + AI", layout="wide")
st.title("⛽ Fuel Leakage Detection + AI Assistant Dashboard")

upload = st.sidebar.file_uploader("📂 Upload processed_trips file", type=["csv", "xlsx"])

# -------------------------------------------------
# AI CHATBOT (SIDEBAR)
# -------------------------------------------------
st.sidebar.subheader("🤖 Gemini AI Assistant")
user_q = st.sidebar.text_input("Ask anything…")

if st.sidebar.button("Ask AI"):
    if user_q.strip() != "":
        ai_answer = model.generate_content(user_q)
        st.sidebar.success(ai_answer.text)
    else:
        st.sidebar.warning("Please enter a question")

# -------------------------------------------------
# When File Uploaded
# -------------------------------------------------
if upload:

    # AUTO Detect CSV / XLSX
    if upload.name.endswith(".csv"):
        df = pd.read_csv(upload)
    else:
        df = pd.read_excel(upload)

    st.write("📊 Uploaded File Preview:", df.head())

    # Ensure required columns
    required_cols = [
        "trip_id", "distance_km", "actual_fuel_liters",
        "diesel_price_per_liter", "leakage_flag"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    # Safe numeric conversion
    for col in ["distance_km", "actual_fuel_liters", "diesel_price_per_liter"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df.loc[df["diesel_price_per_liter"] == 0, "diesel_price_per_liter"] = 95
    df.loc[df["leakage_flag"] == 0, "leakage_flag"] = "Normal"

    # Remove empty rows
    df = df[(df["distance_km"] > 0) & (df["actual_fuel_liters"] > 0)]

    # ----- Route + Mileage -----
    if "avg_mileage" not in df.columns:
        df["avg_mileage"] = 3

    df["route_km"] = df["distance_km"]
    df["total_liters"] = df["route_km"] / df["avg_mileage"]

    # ----- Profit Loss -----
    df["revenue_per_km"] = np.where(df["leakage_flag"] == "Leakage Suspected", 90, 150)
    df["expected_revenue"] = df["distance_km"] * df["revenue_per_km"]
    df["fuel_cost"] = df["actual_fuel_liters"] * df["diesel_price_per_liter"]

    # Add random loss
    if len(df) > 0:
        loss_rows = df.sample(frac=0.25, random_state=42).index
        df.loc[loss_rows, "fuel_cost"] *= np.random.uniform(1.2, 1.8, len(loss_rows))

    df["profit_loss"] = df["expected_revenue"] - df["fuel_cost"]
    df["pnl_status"] = np.where(df["profit_loss"] > 0, "Profit", "Loss")

    df = df.fillna(0)

    # ----- Show Clean Data -----
    st.success("✅ Data cleaned & calculated!")
    st.dataframe(df.head(20))

    # ----- Upload to Supabase -----
    try:
        supabase.table("trip_data").delete().neq("trip_id", "").execute()
        supabase.table("trip_data").insert(df.to_dict(orient="records")).execute()
        st.success("🚀 Uploaded to Supabase!")
    except Exception as e:
        st.error(f"❌ Upload error: {e}")

    # ----- Metrics -----
    total_trips = len(df)
    total_profit = df[df["pnl_status"] == "Profit"]["profit_loss"].sum()
    total_loss = abs(df[df["pnl_status"] == "Loss"]["profit_loss"].sum())
    avg_profit = df["profit_loss"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trips", total_trips)
    c2.metric("Profit (₹)", f"{total_profit:,.0f}")
    c3.metric("Loss (₹)", f"{total_loss:,.0f}")

    # ----- Charts -----
    st.subheader("📈 Profit vs Loss")
    st.plotly_chart(
        px.bar(df, x="trip_id", y="profit_loss", color="pnl_status"),
        use_container_width=True
    )

    st.subheader("⛽ Fuel vs Distance")
    st.plotly_chart(
        px.scatter(df, x="distance_km", y="actual_fuel_liters", color="leakage_flag"),
        use_container_width=True
    )

    # ----- Download -----
    st.download_button(
        "💾 Download Cleaned File",
        df.to_csv(index=False),
        "cleaned_trip_data.csv",
        "text/csv"
    )

else:
    st.info("📥 Upload CSV or XLSX to start.")
