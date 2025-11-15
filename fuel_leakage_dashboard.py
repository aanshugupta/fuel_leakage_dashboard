import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai

# ---------------------------------------------------------
# 🔐 CONFIGURE KEYS
# ---------------------------------------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Create supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configure Gemini model
genai.configure(api_key=GEMINI_API_KEY)

# ⭐ BEST MODEL SELECTED FROM YOUR SCREENSHOT
MODEL_NAME = "gemini-1.5-pro"
model = genai.GenerativeModel(MODEL_NAME)

# ---------------------------------------------------------
# 📥 GET DATA FROM SUPABASE TABLE
# --------------------------------------------------------
def load_table(table_name):
    try:
        data = supabase.table(table_name).select("*").execute()
        df = pd.DataFrame(data.data)
        return df
    except:
        return pd.DataFrame()


# ---------------------------------------------------------
# 🧠 1. AI LEAKAGE DETECTION ENGINE
# ---------------------------------------------------------
def detect_leakage(df):
    required = ["Product Volume", "Rate (Rs./ Ltr)", "Purchase Amount"]

    if not all(col in df.columns for col in required):
        return None, "Required columns missing"

    df["Expected"] = df["Product Volume"] * df["Rate (Rs./ Ltr)"]
    df["Diff"] = df["Purchase Amount"] - df["Expected"]
    df["Leak %"] = (df["Diff"] / df["Expected"]) * 100

    ai_summary = model.generate_content(
        f"""
        Analyze this leakage dataset and describe:
        - leakage probability
        - suspicious transactions
        - overcharging patterns
        Data:
        {df.to_dict()}
        """
    ).text

    return df, ai_summary


# ---------------------------------------------------------
# 🧠 2. AI TRANSACTION EXPLAINER
# ---------------------------------------------------------
def explain_transaction(row):
    prompt = f"""
    Give a clear human-style summary of this fuel transaction:

    {row.to_dict()}
    """

    return model.generate_content(prompt).text


# ---------------------------------------------------------
# 🧠 3. AI FRAUD PATTERN DETECTION
# ---------------------------------------------------------
def detect_fraud(df):
    prompt = f"""
    Detect fuel fraud, misuse, station overcharging, duplicate
    transactions and abnormal amounts in this dataset.
    Give a clean summary.
    Data:
    {df.to_dict()}
    """
    return model.generate_content(prompt).text


# ---------------------------------------------------------
# 🧠 4. AI MONTHLY SUMMARY
# ---------------------------------------------------------
def monthly_summary(df):
    prompt = f"""
    Generate a smart monthly summary from this dataset:
    - Month wise totals
    - Most expensive station
    - Truck efficiency
    - Alerts

    Data:
    {df.to_dict()}
    """

    return model.generate_content(prompt).text


# ---------------------------------------------------------
# 🧠 5. AI CHATBOT
# ---------------------------------------------------------
def ask_ai_about_data(question, df):
    prompt = f"""
    You are a fuel analytics expert AI.
    Use ONLY the given dataset to answer.

    Dataset:
    {df.to_dict()}

    Question: {question}
    """

    return model.generate_content(prompt).text


# =========================================================
# 🚀 UI STARTS HERE (PREMIUM VERSION)
# =========================================================

st.title("⛽ Premium Fuel Intelligence Dashboard — Pack 3 (AI Engine)")

# ---------------------------------------------------------
# SIDEBAR: SELECT SUPABASE TABLE
# ---------------------------------------------------------
st.sidebar.header("📦 Select Supabase Table")
table = st.sidebar.selectbox(
    "Choose table",
    ["sales_data", "drive_summary", "trip_data"]
)

df = load_table(table)

if df.empty:
    st.error("⚠ No data found in this Supabase table.")
    st.stop()

st.success(f"Loaded {len(df)} records from: {table}")

# ---------------------------------------------------------
# TRANSACTION LOOKUP
# ---------------------------------------------------------
st.subheader("🔍 Transaction Lookup")

txn_field = None
for c in df.columns:
    if "transaction" in c.lower():
        txn_field = c
        break

if txn_field:
    txn_id = st.selectbox("Select Transaction ID", df[txn_field].astype(str).unique())

    if txn_id:
        row = df[df[txn_field].astype(str) == txn_id].iloc[0]
        st.write("### Transaction Details")
        st.dataframe(pd.DataFrame([row]))

        st.write("### 🧠 AI Explanation")
        st.info(explain_transaction(row))


# ---------------------------------------------------------
# QUICK STATS
# ---------------------------------------------------------
st.subheader("📊 Quick Stats")
st.metric("Total Rows", len(df))
st.metric("Numeric Columns", len(df.select_dtypes(include=['int64', 'float64']).columns))


# ---------------------------------------------------------
# LEAKAGE DETECTION
# ---------------------------------------------------------
st.subheader("🚨 Fuel Leakage Detection (AI)")

leak_df, leak_text = detect_leakage(df)

if leak_df is not None:
    st.dataframe(leak_df[["Product Volume", "Rate (Rs./ Ltr)", "Purchase Amount", "Leak %"]])
    st.write("### 🧠 AI Leakage Summary")
    st.error(leak_text)
else:
    st.warning(leak_text)


# ---------------------------------------------------------
# MONTHLY SUMMARY AI
# ---------------------------------------------------------
st.subheader("🗓 Monthly Summary (AI)")
st.info(monthly_summary(df))


# ---------------------------------------------------------
# FRAUD ANALYSIS
# ---------------------------------------------------------
st.subheader("🔍 Fraud Detection (AI)")
st.warning(detect_fraud(df))


# ---------------------------------------------------------
# CHART BUILDER
# ---------------------------------------------------------
st.subheader("📈 Chart Builder (Premium)")

num_cols = df.select_dtypes(include=["int64", "float64"]).columns

if len(num_cols) > 0:
    col = st.selectbox("Select numeric column", num_cols)
    chart = st.radio("Chart Type", ["Line", "Bar", "Area"])

    if chart == "Line":
        fig = px.line(df, y=col)
    elif chart == "Bar":
        fig = px.bar(df, y=col)
    else:
        fig = px.area(df, y=col)

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No numeric columns available for charts.")


# ---------------------------------------------------------
# AI CHATBOT
# ---------------------------------------------------------
st.subheader("🤖 Fuel Assistant — Gemini Pro")

q = st.text_input("Ask anything about your dataset...")

if q:
    st.success(ask_ai_about_data(q, df))
