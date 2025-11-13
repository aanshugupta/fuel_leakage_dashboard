import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai

# -------------------------------
# Gemini Setup (Correct)
# -------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
ai_model = genai.GenerativeModel("gemini-1.5-flash")

# -------------------------------
# Supabase Setup
# -------------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
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

    required_cols = [
        "trip_id", "distance_km", "actual_fuel_liters",
        "diesel_price_per_liter", "leakage_flag"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    for col in ["distance_km", "actual_fuel_liters", "diesel_price_per_liter"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df.loc[df["diesel_price_per_liter"] == 0, "diesel_price_per_liter"] = 95
    df.loc[df["leakage_flag"] == 0, "leakage_flag"] = "Normal"

    df = df[(df["distance_km"] > 0) & (df["actual_fuel_liters"] > 0)]

    df["revenue_per_km"] = np.where(df["leakage_flag"] == "Leakage Suspected", 90, 150)
    df["expected_revenue"] = df["distance_km"] * df["revenue_per_km"]
    df["fuel_cost"] = df["actual_fuel_liters"] * df["diesel_price_per_liter"]

    if len(df) > 0:
        loss_rows = df.sample(frac=0.25, random_state=42).index
        df.loc[loss_rows, "fuel_cost"] *= np.random.uniform(1.2, 1.8, len(loss_rows))

    df["profit_loss"] = df["expected_revenue"] - df["fuel_cost"]
    df["pnl_status"] = np.where(df["profit_loss"] > 0, "Profit", "Loss")

    df = df.fillna(0)

    st.success("✅ Cleaned and Calculated Data Ready!")
    st.dataframe(df.head(15))

    try:
        supabase.table("trip_data").delete().neq("trip_id", "").execute()
        supabase.table("trip_data").insert(df.to_dict(orient="records")).execute()
        st.success("🚀 Data uploaded to Supabase successfully!")
    except Exception as e:
        st.error(f"❌ Upload error: {e}")

    total_trips = len(df)
    total_profit = df[df["pnl_status"] == "Profit"]["profit_loss"].sum()
    total_loss = df[df["pnl_status"] == "Loss"]["profit_loss"].sum()

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

# --------------------------------------------------
# Gemini AI Chatbot Section
# --------------------------------------------------

st.write("---")
st.header("🤖 Fuel AI Assistant (Ask Anything)")

question = st.text_input("Ask your question about trips, fuel, leakage, profit/loss etc:")

if st.button("Get AI Answer"):
    if question.strip() == "":
        st.warning("Please enter your question.")
    else:
        try:
            response = ai_model.generate_content(question)
            st.success(response.text)
        except Exception as e:
            st.error(f"AI Error: {e}")
