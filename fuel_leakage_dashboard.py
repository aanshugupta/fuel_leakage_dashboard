import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai
import json

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
ai_model = genai.GenerativeModel("gemini-1.5-flash")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------------------------------
# PAGE SETTINGS
# ------------------------------------------------
st.set_page_config(page_title="Fuel & Transaction Dashboard", layout="wide")
st.title("⛽ Fuel Leakage + Transaction Analysis Dashboard")

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
uploaded = st.sidebar.file_uploader("📂 Upload CSV / Excel", type=["csv", "xlsx"])

# Chatbot
st.sidebar.write("---")
st.sidebar.subheader("🤖 Ask AI")
q = st.sidebar.text_input("Your Question")
if st.sidebar.button("Ask AI"):
    if q.strip():
        try:
            ans = ai_model.generate_content(q)
            st.sidebar.success(ans.text)
        except Exception as e:
            st.sidebar.error(f"AI Error: {e}")
    else:
        st.sidebar.warning("Type a question first")

# ------------------------------------------------
# PROCESS FILE
# ------------------------------------------------
if uploaded:

    # Detect format
    if uploaded.name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded, header=None)
    else:
        raw_df = pd.read_excel(uploaded, header=None)

    # ---- REMOVE FIRST 9 LINES ----
    df = raw_df.iloc[9:].reset_index(drop=True)

    # ---- SET ROW 10 AS HEADER ----
    df.columns = df.iloc[0]          # row 10 → header
    df = df[1:].reset_index(drop=True)

    st.subheader("📊 Cleaned Data Preview (after fixing row issue)")
    st.dataframe(df.head(20))

    # ------------------------------------------------
    # SAVE TO SUPABASE EXACT FORMAT
    # ------------------------------------------------
    if st.button("Upload to Supabase"):
        try:
            data = df.where(pd.notnull(df), None).to_dict(orient="records")
            res = supabase.table("trip_data").insert(data).execute()
            if res.status_code in (200, 201):
                st.success("🚀 Data uploaded to Supabase successfully!")
            else:
                st.error(f"Supabase Error: {res.data}")
        except Exception as e:
            st.error(f"Upload Failed: {e}")

    st.write("---")

    # ------------------------------------------------
    # TRANSACTION ID DROPDOWN
    # ------------------------------------------------
    st.header("🔍 Search by Transaction ID")

    # Detect best column name
    txn_col = None
    for c in df.columns:
        if "txn" in c.lower() or "transaction" in c.lower():
            txn_col = c
            break

    if txn_col:

        txn_values = (
            df[txn_col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_txn = st.selectbox("Select Transaction ID:", txn_values)

        if selected_txn:
            result = df[df[txn_col].astype(str) == selected_txn]
            st.subheader("📄 Transaction Details")
            st.dataframe(result)

    else:
        st.warning("Transaction ID column not found.")

    st.write("---")

    # ------------------------------------------------
    # FULL DATA GRAPHS
    # ------------------------------------------------
    st.header("📈 Charts & Visualization")

    # PIE CHART → count by any categorical
    cat_cols = [c for c in df.columns if df[c].nunique() < 20]

    if len(cat_cols) > 0:
        col = st.selectbox("Select column for Pie Chart:", cat_cols)
        pie_df = df[col].value_counts().reset_index()
        pie_df.columns = ['value', 'count']

        fig_pie = px.pie(pie_df, names="value", values="count",
                         title=f"Distribution by {col}")
        st.plotly_chart(fig_pie, use_container_width=True)

    # BAR CHART → numerical column
    num_cols = []
    for c in df.columns:
        try:
            if pd.to_numeric(df[c], errors='coerce').notnull().sum() > 0:
                num_cols.append(c)
        except:
            pass

    if len(num_cols) >= 2:
        x = st.selectbox("X-axis (Numeric):", num_cols)
        y = st.selectbox("Y-axis (Numeric):", num_cols, index=1)

        df[x] = pd.to_numeric(df[x], errors="coerce")
        df[y] = pd.to_numeric(df[y], errors="coerce")

        fig_bar = px.bar(df, x=x, y=y, title=f"{x} vs {y}")
        st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.info("📥 Upload your CSV or Excel file to start.")
