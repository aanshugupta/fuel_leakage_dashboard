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

# Sidebar
uploaded = st.sidebar.file_uploader("📂 Upload CSV / Excel", type=["csv", "xlsx"])

# Chatbot
st.sidebar.write("---")
st.sidebar.subheader("🤖 Ask AI")
q = st.sidebar.text_input("Your Question")
if st.sidebar.button("Ask AI"):
    if q:
        try:
            ans = ai_model.generate_content(q)
            st.sidebar.success(ans.text)
        except Exception as e:
            st.sidebar.error(f"AI Error: {e}")
    else:
        st.sidebar.warning("Type a question")


# ------------------------------------------------
# PROCESS FILE
# ------------------------------------------------
if uploaded:

    # Read file without header
    if uploaded.name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded, header=None)
    else:
        raw_df = pd.read_excel(uploaded, header=None)

    # ---- Remove first 9 rows ----
    df = raw_df.iloc[9:].reset_index(drop=True)

    # ---- Set row 10 as header ----
    header_row = df.iloc[0].astype(str).tolist()

    # FIX: Make duplicate column names unique
    new_columns = []
    seen = {}

    for col in header_row:
        if col not in seen:
            seen[col] = 1
            new_columns.append(col)
        else:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")

    df = df[1:].reset_index(drop=True)
    df.columns = new_columns

    st.subheader("📊 Cleaned Data Preview")
    st.dataframe(df.head(20))

    # Save to Supabase
    if st.button("Upload to Supabase"):
        try:
            data = df.where(pd.notnull(df), None).to_dict(orient="records")
            res = supabase.table("trip_data").insert(data).execute()
            st.success("Uploaded to Supabase!")
        except Exception as e:
            st.error(f"Upload Failed: {e}")

    st.write("---")

    # ------------------------------
    # TRANSACTION ID DROPDOWN
    # ------------------------------
    st.header("🔍 Search by Transaction ID")

    txn_col = None
    for c in df.columns:
        if "txn" in c.lower() or "transaction" in c.lower():
            txn_col = c
            break

    if txn_col:
        txn_values = df[txn_col].dropna().astype(str).unique().tolist()

        selected = st.selectbox("Select Transaction ID", txn_values)

        if selected:
            st.subheader("📄 Details:")
            st.dataframe(df[df[txn_col].astype(str) == selected])

    else:
        st.warning("No Transaction ID column found!")

    st.write("---")

    # ------------------------------
    # Charts
    # ------------------------------
    st.header("📈 Charts")

    # Pie Chart
    cat_cols = [c for c in df.columns if df[c].nunique() < 20]

    if len(cat_cols) > 0:
        col = st.selectbox("Pie chart column:", cat_cols)
        pie_df = df[col].value_counts().reset_index()
        pie_df.columns = ["value", "count"]
        fig = px.pie(pie_df, names="value", values="count", title=f"{col} Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # Bar Chart
    num_cols = []
    for c in df.columns:
        try:
            if pd.to_numeric(df[c], errors="coerce").notnull().sum() > 0:
                num_cols.append(c)
        except:
            pass

    if len(num_cols) >= 2:
        x = st.selectbox("X-axis:", num_cols)
        y = st.selectbox("Y-axis:", num_cols, index=1)

        df[x] = pd.to_numeric(df[x], errors="coerce")
        df[y] = pd.to_numeric(df[y], errors="coerce")

        fig2 = px.bar(df, x=x, y=y, title=f"{x} vs {y}")
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("📥 Upload your file to start.")
