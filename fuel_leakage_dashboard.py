import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai
import json

# ------------------------------------------
# Load Secrets (Gemini + Supabase)
# ------------------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

if not GEMINI_API_KEY:
    st.error("Gemini API key missing! Add it in Streamlit Secrets.")
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase credentials missing!")

# ------------------------------------------
# Gemini Setup (FINAL)
# ------------------------------------------
genai.configure(api_key=GEMINI_API_KEY)
gmodel = genai.GenerativeModel("gemini-2.5-flash")  # requested model

# ------------------------------------------
# Supabase Setup
# ------------------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------------------------
# UI Page Config
# ------------------------------------------
st.set_page_config(page_title="Sales & Fuel Dashboard", layout="wide")
st.title("📊 Sales Transaction + Fuel Leakage Dashboard")

# ------------------------------------------
# Sidebar: File Upload + Chatbot
# ------------------------------------------
st.sidebar.header("📂 Upload Data File")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.write("---")
st.sidebar.header("🤖 Ask AI Anything")
side_question = st.sidebar.text_input("Ask here")

if st.sidebar.button("Ask AI"):
    try:
        out = gmodel.generate_content(side_question)
        st.sidebar.success(out.text)
    except Exception as e:
        st.sidebar.error(str(e))

# ------------------------------------------
# If file uploaded → Process
# ------------------------------------------
if uploaded:

    # 1️⃣ READ FILE & SKIP FIRST 9 USELESS ROWS
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded, skiprows=9)
        else:
            df = pd.read_excel(uploaded, skiprows=9)
    except Exception as e:
        st.error(f"File error: {e}")
        st.stop()

    # Show Cleaned Data
    st.subheader("📌 Cleaned Data (after skipping 9 rows)")
    st.dataframe(df.head(20))

    # 2️⃣ NORMALIZE COLUMN NAMES
    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

    rename_map = {
        "sno": "s_no",
        "s.no": "s_no",
        "transaction_id": "transaction_id",
        "txn_id": "transaction_id",
        "fuel_liters": "actual_fuel_liters",
        "distance": "distance_km",
    }
    df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)

    # 3️⃣ CHECK TRANSACTION ID
    if "transaction_id" not in df.columns:
        st.error("❌ 'Transaction ID' column missing! Cannot proceed.")
        st.stop()
    else:
        st.success("✔ Transaction ID detected")

    # 4️⃣ Upload to Supabase
    if st.button("⬆ Upload Cleaned Data to Supabase"):
        try:
            clean_records = df.replace({np.nan: None}).to_dict(orient="records")
            supabase.table("trip_data").insert(clean_records).execute()
            st.success("🎉 Successfully uploaded to Supabase")
        except Exception as e:
            st.error(str(e))

    st.write("---")

    # ------------------------------------------
    # SEARCH BY TRANSACTION ID
    # ------------------------------------------
    st.header("🔍 Search Transaction Details")

    txn_list = df["transaction_id"].dropna().astype(str).unique().tolist()
    selected_txn = st.selectbox("Choose Transaction ID", txn_list)

    if selected_txn:
        card = df[df["transaction_id"] == selected_txn]
        st.subheader("🧾 Transaction Detail")
        st.dataframe(card)

    st.write("---")

    # ------------------------------------------
    # CHARTS SECTION
    # ------------------------------------------
    st.header("📈 Data Visualization")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if num_cols:
        chosen = st.selectbox("Select numeric column for Pie Chart", num_cols)
        pie_df = df[chosen].value_counts().reset_index()
        pie_df.columns = ["Value", "Count"]

        fig = px.pie(pie_df, names="Value", values="Count", title=f"{chosen} Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numeric columns found!")

    st.write("---")

    # ------------------------------------------
    # AI INSIGHTS MAIN AREA
    # ------------------------------------------
    st.header("🤖 AI Insights for Your Data")

    user_q = st.text_input("Ask something like: 'Which transaction has highest amount?'")

    if st.button("Get AI Insights"):

        small_preview = df.head(10).to_dict(orient="records")
        prompt = f"""
        You are a data analyst AI.
        Here is sample of the uploaded data:
        {small_preview}

        User question: {user_q}

        Give a clear, short, helpful answer.
        """

        try:
            ans = gmodel.generate_content(prompt)
            st.success(ans.text)
        except Exception as e:
            st.error(str(e))

else:
    st.info("Please upload a CSV or Excel file to begin.")
