import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai
import json

# ------------------------------------------------
# API KEYS
# ------------------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ------------------------------------------------
# UI SETTINGS
# ------------------------------------------------
st.set_page_config(page_title="Fuel + Transaction Dashboard", layout="wide")
st.title("⛽ Fuel Leakage + Transaction Analysis Dashboard")

uploaded = st.sidebar.file_uploader("📂 Upload CSV / Excel", type=["csv", "xlsx"])

st.sidebar.write("---")
st.sidebar.subheader("🤖 Ask AI")
q = st.sidebar.text_input("Ask your question:")

if st.sidebar.button("Ask"):
    try:
        reply = model.generate_content(q)
        st.sidebar.success(reply.text)
    except Exception as e:
        st.sidebar.error(str(e))


# ------------------------------------------------
# PROCESS FILE
# ------------------------------------------------
if uploaded:

    # Read without header
    if uploaded.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded, header=None)
    else:
        df_raw = pd.read_excel(uploaded, header=None)

    # Remove 1–9 rows → row-10 becomes header
    df = df_raw.iloc[9:].reset_index(drop=True)

    header = df.iloc[0].astype(str).tolist()
    df = df[1:].reset_index(drop=True)

    # CLEAN DUPLICATE COLUMN NAMES
    clean_cols = []
    seen = {}

    for col in header:
        col = col.strip()
        if col == "nan" or col == "":
            col = "Blank"

        if col not in seen:
            seen[col] = 1
            clean_cols.append(col)
        else:
            seen[col] += 1
            clean_cols.append(f"{col}_{seen[col]}")

    df.columns = clean_cols

    st.subheader("📊 Cleaned Data Preview")
    st.dataframe(df.head(20))

    # ------------------------------------------------
    # FIND TRANSACTION COLUMN
    # ------------------------------------------------
    txn_col = None
    patterns = ["txn", "transaction", "tx id", "transaction id"]

    for col in df.columns:
        lower = col.lower()
        if any(key in lower for key in patterns):
            txn_col = col
            break

    if not txn_col:
        st.error("No Transaction ID column detected. Ensure 'Transaction ID' exists in row 10.")
        st.stop()

    # ------------------------------------------------
    # TRANSACTION SEARCH BOX
    # ------------------------------------------------
    st.write("---")
    st.header("🔍 Search by Transaction ID")

    txn_list = df[txn_col].astype(str).unique().tolist()

    selected_txn = st.selectbox("Select Transaction ID:", txn_list)

    if selected_txn:
        result = df[df[txn_col].astype(str) == selected_txn]
        st.subheader("📄 Transaction Details")
        st.dataframe(result)

    # ------------------------------------------------
    # PIE CHART AND BAR CHART
    # ------------------------------------------------
    st.write("---")
    st.header("📈 Charts")

    # PIE CHART
    cat_cols = [c for c in df.columns if df[c].nunique() < 20]

    if len(cat_cols) > 0:
        pie_col = st.selectbox("Select column for Pie Chart:", cat_cols)
        pc = df[pie_col].value_counts().reset_index()
        pc.columns = ["value", "count"]
        fig1 = px.pie(pc, names="value", values="count")
        st.plotly_chart(fig1, use_container_width=True)

    # BAR CHART
    num_cols = []
    for c in df.columns:
        try:
            if pd.to_numeric(df[c], errors="coerce").notnull().sum() > 0:
                num_cols.append(c)
        except:
            pass

    if len(num_cols) > 1:
        c1, c2 = st.columns(2)
        x = c1.selectbox("X axis:", num_cols)
        y = c2.selectbox("Y axis:", num_cols, index=1)

        df[x] = pd.to_numeric(df[x], errors="coerce")
        df[y] = pd.to_numeric(df[y], errors="coerce")

        fig2 = px.bar(df, x=x, y=y)
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("📥 Upload your file to begin analysis.")
