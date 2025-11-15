import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai

# ============================================================
# CONFIGURATION
# ============================================================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL_NAME)


# ============================================================
# LOAD DATA FROM SUPABASE
# ============================================================
def load_table(table_name):
    try:
        data = supabase.table(table_name).select("*").execute()
        return pd.DataFrame(data.data)
    except:
        return pd.DataFrame()


# ============================================================
# AI FUNCTIONS
# ============================================================
def ai_text(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "AI Error. Please retry."


def ai_leakage(df):
    required = ["Product Volume", "Rate (Rs./ Ltr)", "Purchase Amount"]

    if not all(col in df.columns for col in required):
        return None, "Required fields missing"

    df["Expected"] = df["Product Volume"] * df["Rate (Rs./ Ltr)"]
    df["Variance"] = df["Purchase Amount"] - df["Expected"]
    df["Leak Percent"] = (df["Variance"] / df["Expected"]) * 100

    text = ai_text(
        f"Analyze fuel leakage patterns and give summary. Data: {df.to_dict()}"
    )

    return df, text


def ai_transaction_summary(row):
    return ai_text(
        f"Explain this transaction in simple language: {row.to_dict()}"
    )


def ai_fraud(df):
    return ai_text(
        f"Detect fraud, duplicate entries, abnormal fuel usage, high prices. Data: {df.to_dict()}"
    )


def ai_monthly(df):
    return ai_text(
        f"Give a clean monthly summary of fuel usage, cost, station performance, vehicle efficiency. Data: {df.to_dict()}"
    )


def ai_chat(question, df):
    return ai_text(
        f"You are a fuel analytics expert. Use only this dataset to answer: {df.to_dict()}. Question: {question}"
    )


# ============================================================
# UI START
# ============================================================

st.set_page_config(page_title="Fuel Intelligence Dashboard", layout="wide")

st.title("Fuel Intelligence Dashboard (Premium Edition)")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Data Controls")

table_name = st.sidebar.selectbox(
    "Select Supabase Table",
    ["sales_data", "drive_summary", "trip_data"]
)

df = load_table(table_name)

if df.empty:
    st.error("No data found in selected table.")
    st.stop()

st.sidebar.success(f"Loaded {len(df)} rows")


# ============================================================
# SECTION 1 — TRANSACTION LOOKUP
# ============================================================

st.header("Transaction Lookup")

txn_field = None
for c in df.columns:
    if "transaction" in c.lower():
        txn_field = c

if txn_field:
    txn_id = st.selectbox("Select Transaction ID", df[txn_field].astype(str).unique())

    if txn_id:
        row = df[df[txn_field].astype(str) == txn_id].iloc[0]
        st.subheader("Transaction Details")
        st.dataframe(pd.DataFrame([row]), use_container_width=True)

        st.subheader("AI Summary")
        st.info(ai_transaction_summary(row))


# ============================================================
# SECTION 2 — QUICK STATS
# ============================================================

st.header("Quick Statistics")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", len(df))
col2.metric("Numeric Columns", len(df.select_dtypes(include=["int64", "float64"]).columns))
col3.metric("Unique Vehicles", df["Vehicle Number"].nunique() if "Vehicle Number" in df.columns else 0)


# ============================================================
# SECTION 3 — FUEL LEAKAGE ANALYSIS
# ============================================================

st.header("Fuel Leakage Detection")

leak_df, leak_text = ai_leakage(df)

if leak_df is not None:
    st.dataframe(leak_df[["Product Volume", "Rate (Rs./ Ltr)", "Purchase Amount", "Expected", "Leak Percent"]])
    st.subheader("AI Leakage Result")
    st.warning(leak_text)
else:
    st.error(leak_text)


# ============================================================
# SECTION 4 — MONTHLY SUMMARY
# ============================================================

st.header("Monthly AI Summary")

st.info(ai_monthly(df))


# ============================================================
# SECTION 5 — FRAUD DETECTION
# ============================================================

st.header("Fraud & Anomaly Detection")

st.warning(ai_fraud(df))


# ============================================================
# SECTION 6 — VEHICLE LEVEL ANALYTICS
# ============================================================

st.header("Vehicle Level Analysis")

if "Vehicle Number" in df.columns:

    vehicle = st.selectbox("Select Vehicle", ["All"] + list(df["Vehicle Number"].unique()))

    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    metric = st.multiselect("Select Metrics", num_cols)

    if metric:

        if vehicle == "All":
            fig = px.line(df, x=df.columns[0], y=metric)
        else:
            fig = px.line(df[df["Vehicle Number"] == vehicle], x=df.columns[0], y=metric)

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# SECTION 7 — CHART BUILDER
# ============================================================

st.header("Chart Builder")

num_cols = df.select_dtypes(include=["int64", "float64"]).columns

if len(num_cols) > 0:
    c = st.selectbox("Select Column", num_cols)
    t = st.radio("Chart Type", ["Line", "Bar", "Area"])

    if t == "Line":
        fig = px.line(df, y=c)
    elif t == "Bar":
        fig = px.bar(df, y=c)
    else:
        fig = px.area(df, y=c)

    st.plotly_chart(fig)
else:
    st.warning("No numeric fields available")


# ============================================================
# SECTION 8 — AI CHATBOT
# ============================================================

st.header("AI Assistant")

q = st.text_input("Ask something about your data")

if q:
    st.success(ai_chat(q, df))
