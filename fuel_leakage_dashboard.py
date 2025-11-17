import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai

st.set_page_config(
    page_title="Fuel Intelligence Dashboard",
    layout="wide"
)

# --------------------------------------------
# KEYS
# --------------------------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL_NAME)

# --------------------------------------------
# SMART COLUMN MAPPER (semantic)
# --------------------------------------------
def smart_find(df, keywords):
    for col in df.columns:
        name = col.lower().replace(" ", "")
        if any(k in name for k in keywords):
            return col
    return None

# --------------------------------------------
# LOAD DATA FROM SUPABASE
# --------------------------------------------
def load_supabase(table):
    try:
        data = supabase.table(table).select("*").execute()
        return pd.DataFrame(data.data)
    except:
        return pd.DataFrame()# --------------------------------------------
# AI: Transaction Explainer
# --------------------------------------------
def ai_explain_transaction(row):
    prompt = f"""
    Explain this fuel transaction clearly:

    {row.to_dict()}
    """
    return model.generate_content(prompt).text


# --------------------------------------------
# AI: Leakage Detection
# --------------------------------------------
def ai_leakage(df):
    volume = smart_find(df, ["volume", "qty", "litre"])
    rate = smart_find(df, ["rate", "price"])
    amount = smart_find(df, ["amount", "purchase", "total"])

    if not all([volume, rate, amount]):
        return None, "Required columns missing."

    df["expected_amount"] = df[volume] * df[rate]
    df["variance"] = df[amount] - df["expected_amount"]
    df["leak_pct"] = (df["variance"] / df["expected_amount"]) * 100

    summary = model.generate_content(
        f"Analyze fuel leakage patterns:\n{df.to_dict()}"
    ).text

    return df, summary


# --------------------------------------------
# AI: Fraud Detection
# --------------------------------------------
def ai_fraud(df):
    prompt = f"""
    Analyze the dataset for:
    - Duplicate transactions
    - Fake litres
    - Overcharging
    - Card misuse
    - Station fraud

    Data:
    {df.to_dict()}
    """

    return model.generate_content(prompt).text


# --------------------------------------------
# AI: Monthly Summary
# --------------------------------------------
def ai_monthly_summary(df):
    prompt = f"""
    Give a month-wise smart summary:
    - Total fuel
    - Total spend
    - Efficiency
    - Vehicle highlights
    - Station patterns
    - Red flags

    Dataset:
    {df.to_dict()}
    """

    return model.generate_content(prompt).text


# --------------------------------------------
# AI Chatbot
# --------------------------------------------
def ai_chat(question, df):
    prompt = f"""
    You are a fuel analytics AI.
    Use only this dataset to answer clearly:

    Dataset: {df.to_dict()}
    Question: {question}
    """

    return model.generate_content(prompt).textst.title("⛽ Premium Fuel Intelligence Dashboard")
st.write("Single-page AI dashboard with Supabase + CSV upload + Gemini 2.0 Flash")

# ------------------------------------------------------
# DATA SOURCE SECTION
# ------------------------------------------------------
st.header("📥 Load Your Data")

col1, col2 = st.columns(2)

with col1:
    table = st.selectbox("Select Supabase Table", 
                         ["sales_data", "trip_data", "drive_summary"])
    df = load_supabase(table)

    if not df.empty:
        st.success(f"Loaded {len(df)} rows from {table}")

with col2:
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if uploaded:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.success(f"Uploaded file: {uploaded.name} — {len(df)} rows loaded")

# No data? stop
if df.empty:
    st.error("No data available.")
    st.stop()

# ------------------------------------------------------
# TABLE PREVIEW
# ------------------------------------------------------
st.subheader("🔎 Data Preview")
st.dataframe(df.head(50), use_container_width=True)# ------------------------------------------------------
# TRANSACTION LOOKUP
# ------------------------------------------------------
st.header("🔍 Transaction Lookup")

txn_col = smart_find(df, ["transactionid", "txn"])

if txn_col:
    txn_id = st.selectbox("Select Transaction ID", df[txn_col].astype(str).unique())

    if txn_id:
        row = df[df[txn_col].astype(str) == txn_id].iloc[0]
        st.write("### Transaction Details")
        st.dataframe(pd.DataFrame([row]))

        st.write("### 🧠 AI Explanation")
        st.info(ai_explain_transaction(row))


# ------------------------------------------------------
# QUICK STATS
# ------------------------------------------------------
st.header("📊 Quick Stats")

st.metric("Total Rows", len(df))
st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))


# ------------------------------------------------------
# LEAKAGE DETECTION
# ------------------------------------------------------
st.header("🚨 Fuel Leakage Detection (AI)")

leak_df, leak_text = ai_leakage(df)

if leak_df is not None:
    st.dataframe(leak_df[["expected_amount", "variance", "leak_pct"]].head(50))
    st.info(leak_text)
else:
    st.warning(leak_text)


# ------------------------------------------------------
# FRAUD DETECTION
# ------------------------------------------------------
st.header("🛑 Fraud Detection (AI)")
st.warning(ai_fraud(df))


# ------------------------------------------------------
# MONTHLY SUMMARY
# ------------------------------------------------------
st.header("🗓 Monthly Summary (AI)")
st.info(ai_monthly_summary(df))# ------------------------------------------------------
# ADVANCED CHART BUILDER
# ------------------------------------------------------
st.header("📈 Chart Builder")

num_cols = df.select_dtypes(include=[np.number]).columns

if len(num_cols) > 0:
    col = st.selectbox("Select Numeric Column", num_cols)
    method = st.radio("Chart Type", ["Line", "Bar", "Area"])

    if method == "Line":
        fig = px.line(df, y=col)
    elif method == "Bar":
        fig = px.bar(df, y=col)
    else:
        fig = px.area(df, y=col)

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No numeric columns found.")


# ------------------------------------------------------
# AI CHATBOT
# ------------------------------------------------------
st.header("🤖 AI Assistant — Gemini 2.0 Flash")

question = st.text_input("Ask anything about your dataset...")

if question:
    st.success(ai_chat(question, df))
