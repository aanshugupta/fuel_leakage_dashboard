import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai
import json

# ------------------------------------------------
# CONFIG (use Streamlit Secrets)
# ------------------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_model = genai.GenerativeModel("gemini-1.5-flash")

# ------------------------------------------------
# STREAMLIT SETUP
# ------------------------------------------------
st.set_page_config(page_title="Fuel & Transaction Dashboard", layout="wide")
st.title("⛽ Fuel Leakage + Transaction Analysis Dashboard")

# ------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------
uploaded = st.sidebar.file_uploader("📂 Upload CSV or Excel File", type=["csv", "xlsx"])

# CHATBOT (Sidebar)
st.sidebar.write("---")
st.sidebar.subheader("🤖 Ask AI about data")

user_q = st.sidebar.text_input("Your Question:")
if st.sidebar.button("Get Answer"):
    if user_q.strip():
        try:
            ans = ai_model.generate_content(user_q)
            st.sidebar.success(ans.text)
        except Exception as e:
            st.sidebar.error(f"AI Error: {e}")
    else:
        st.sidebar.warning("Type a question first.")

# ------------------------------------------------
# WHEN FILE IS UPLOADED
# ------------------------------------------------
if uploaded:

    # Auto-detect file type
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head(20))

    # ------------------------------
    # CLEANING : VERY SAFE (DO NOT REMOVE DATA)
    # ------------------------------

    df = df.dropna(how="all")          # remove fully empty rows
    df.columns = [str(c).strip() for c in df.columns]   # clean column names

    # -----------------------------------------
    # STORE IN SUPABASE (Same format as upload)
    # -----------------------------------------
    if st.button("Upload to Supabase"):
        try:
            data = df.where(pd.notnull(df), None).to_dict(orient="records")
            res = supabase.table("trip_data").insert(data).execute()

            if res.status_code in (200, 201):
                st.success("🚀 File saved to Supabase successfully!")
            else:
                st.error(f"Error: {res.data}")
        except Exception as e:
            st.error(f"Upload Failed: {e}")

    st.write("---")

    # ------------------------------------------------
    # DROPDOWN → Transaction ID
    # ------------------------------------------------
    st.header("🔍 Search Truck / Transaction Details")

    # Detect best column for dropdown
    possible_keys = ["Transaction ID", "transaction_id", "txn_id", "TXN", "TXN ID"]

    txn_col = None
    for c in df.columns:
        if "txn" in c.lower() or "transaction" in c.lower():
            txn_col = c
            break

    if txn_col:
        txn_list = df[txn_col].dropna().astype(str).unique().tolist()
        selected_txn = st.selectbox("Select Transaction ID:", txn_list)

        # Show selected record
        if selected_txn:
            record = df[df[txn_col].astype(str) == selected_txn]
            st.subheader("📄 Transaction Details")
            st.dataframe(record)
    else:
        st.warning("No Transaction ID column detected.")

    st.write("---")

    # ------------------------------------------------
    # OPTIONAL: Profit/Loss calculation if required columns exist
    # ------------------------------------------------
    required_for_pnl = {"distance_km", "actual_fuel_liters", "diesel_price_per_liter"}

    if required_for_pnl.issubset(set(df.columns)):

        st.header("💰 PNL Calculation (Auto-detected)")

        df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce").fillna(0)
        df["actual_fuel_liters"] = pd.to_numeric(df["actual_fuel_liters"], errors="coerce").fillna(0)
        df["diesel_price_per_liter"] = pd.to_numeric(df["diesel_price_per_liter"], errors="coerce").fillna(95)

        df["expected_revenue"] = df["distance_km"] * 150
        df["fuel_cost"] = df["actual_fuel_liters"] * df["diesel_price_per_liter"]
        df["profit_loss"] = df["expected_revenue"] - df["fuel_cost"]
        df["pnl_status"] = np.where(df["profit_loss"] > 0, "Profit", "Loss")

        # KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Trips", len(df))
        c2.metric("Total Profit", f"{df[df.pnl_status=='Profit']['profit_loss'].sum():,.0f}")
        c3.metric("Total Loss", f"{df[df.pnl_status=='Loss']['profit_loss'].sum():,.0f}")

        # Chart
        st.subheader("📈 Profit/Loss Chart")
        st.plotly_chart(
            px.bar(df, x=df.index, y="profit_loss", color="pnl_status"),
            use_container_width=True
        )
    else:
        st.info("PNL calculation skipped because required columns not found.")

else:
    st.info("📥 Upload CSV or Excel file to begin.")
