import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from supabase import create_client, Client
import os, json, re

# ------------------------
# GEMINI SETUP (NEW SDK)
# ------------------------
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") if st.secrets else os.environ.get("GEMINI_API_KEY")
GEMINI_AVAILABLE = False
genai_client = None

try:
    from google import genai
    if GEMINI_KEY:
        genai_client = genai.Client(api_key=GEMINI_KEY)
        GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False


def ask_gemini(prompt: str) -> str:
    if not GEMINI_AVAILABLE:
        return "Gemini 2.5 Flash not configured."

    try:
        resp = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return resp.text
    except Exception as e:
        return f"Gemini error: {e}"


# ------------------------
# SUPABASE SETUP
# ------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None


# ------------------------
# READ FILE FROM ROW 10
# ------------------------
def read_uploaded_file(file):
    name = file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(file, header=9)
    else:
        df = pd.read_excel(file, engine="openpyxl", header=9)

    return df


# ------------------------
# CLEAN COLUMN NAMES
# ------------------------
REQUIRED_COLS = {
    "S.No.": "s_no",
    "Transaction ID": "transaction_id",
    "Transaction Date": "transaction_date",
    "Transaction Time": "transaction_time",
    "Transaction Type": "transaction_type",
    "Name": "customer_name",
}

def clean_dataframe(df):
    df = df.rename(columns=REQUIRED_COLS)
    df = df[list(REQUIRED_COLS.values())]
    return df


# ------------------------
# STREAMLIT UI
# ------------------------
st.set_page_config(page_title="Fuel Leakage + Transaction Dashboard", layout="wide")
st.title("⛽ Fuel Leakage + Transaction Dashboard")

uploaded = st.file_uploader("Upload CSV/XLSX file", type=["csv", "xlsx"])

if uploaded:
    df = read_uploaded_file(uploaded)
    df = clean_dataframe(df)

    st.subheader("📊 Cleaned Data Preview")
    st.dataframe(df)

    # Dropdown Search
    st.subheader("🔍 Search by Transaction ID")
    txn = st.selectbox("Select Transaction:", df["transaction_id"].astype(str).unique().tolist())

    selected_row = df[df["transaction_id"] == txn]

    st.write("### Selected Transaction Details")
    st.dataframe(selected_row.T)

    # Pie chart
    fig = px.pie(df, names="transaction_type", title="Transaction Type Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # Upload to Supabase
    if st.button("Upload to Supabase"):
        data = df.to_dict(orient="records")
        supabase.table("trip_data").insert(data).execute()
        st.success("Uploaded successfully!")

    # AI Assistant
    st.subheader("🤖 Ask AI About Your Data")
    q = st.text_input("Your Question")

    if st.button("Ask AI"):
        preview = df.head(10).to_dict(orient="records")
        prompt = f"Here is sample data: {json.dumps(preview)}\n\nQuestion: {q}"
        st.write(ask_gemini(prompt))
