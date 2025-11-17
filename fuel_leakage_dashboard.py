import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai
import numpy as np

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)

MODEL = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL)


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def find_column(df, keywords):
    keywords = [k.lower() for k in keywords]
    for col in df.columns:
        c = col.lower().replace(" ", "").replace("_", "")
        for k in keywords:
            if k in c:
                return col
    return None


def load_table(table):
    try:
        res = supabase.table(table).select("*").execute()
        return pd.DataFrame(res.data)
    except:
        return pd.DataFrame()


# ------------------------------------------------------------
# AI SAFE PROMPT SYSTEM (No JSON, No Dict Output)
# ------------------------------------------------------------
def ai_answer(prompt):
    return model.generate_content(prompt).text


def ai_summary(df):
    sample = df.head(30).to_dict()

    prompt = f"""
    You are a fuel analytics expert.
    Write a very simple monthly summary in clean English.
    No JSON, no lists, no bullet points.

    Explain:
    - overall spending
    - total fuel usage
    - any unusual days
    - general performance
    - trends (up/down)

    Make it friendly and very easy to understand.

    Use this small data sample:
    {sample}
    """
    return ai_answer(prompt)


def ai_fraud(df):
    sample = df.head(40).to_dict()

    prompt = f"""
    You are a fraud detection expert.
    Explain in simple English if the dataset shows:

    - repeated unusual transactions
    - unexpectedly high amounts
    - duplicate entries
    - repeated misuse at same fuel station

    Keep it friendly and readable.

    Use this sample:
    {sample}
    """
    return ai_answer(prompt)


def ai_explain_transaction(row):
    prompt = f"""
    You are helping a non-technical fleet owner.
    Explain this fuel transaction in simple English:
    {row.to_dict()}

    No bullet points, just a clean paragraph.
    """
    return ai_answer(prompt)


def ai_leakage(df):
    qty = find_column(df, ["volume", "qty", "litre"])
    rate = find_column(df, ["rate", "price"])
    amt = find_column(df, ["amount", "purchase", "total"])

    if not qty or not rate or not amt:
        return None, "Required columns not found."

    df["expected"] = df[qty] * df[rate]
    df["diff"] = df[amt] - df["expected"]
    df["leak_pct"] = (df["diff"] / df["expected"]) * 100

    sample = df.head(30).to_dict()

    prompt = f"""
    Explain possible fuel leakage or mismatches in simple English.
    No JSON or code. Keep it human-friendly.

    Sample Data:
    {sample}
    """
    return df, ai_answer(prompt)


def ai_chart_explain(df, col):
    vals = df[col].dropna().head(40).tolist()

    prompt = f"""
    Explain this chart to a non-technical user.
    Column: {col}
    Values: {vals}

    Explain:
    - What the trend means
    - Is it increasing, decreasing or irregular?
    - What a fleet owner should understand
    """
    return ai_answer(prompt)


def ai_chat(question, df):
    sample = df.head(40).to_dict()

    prompt = f"""
    You are a friendly fleet-data expert.
    Answer in simple English only.
    No code. No JSON.

    Question: {question}

    Use this sample dataset only:
    {sample}
    """
    return ai_answer(prompt)


# ------------------------------------------------------------
# UI SETTINGS (Light, clean)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Fuel Dashboard",
    page_icon="⛽",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #f8f9fa;
}
h2, h3, h1 {
    color: #0b2239;
    font-family: 'Segoe UI', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.header("Data Controls")

table = st.sidebar.selectbox("Select Supabase Table", ["sales_data", "trip_data"])
df_supabase = load_table(table)
st.sidebar.success(f"Loaded {len(df_supabase)} rows")

# Upload
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
df_uploaded = None

if uploaded:
    df_uploaded = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
    st.sidebar.success(f"Uploaded {len(df_uploaded)} rows")

df = df_uploaded if df_uploaded is not None else df_supabase
source = "Uploaded File" if df_uploaded is not None else "Supabase"


st.write(f"### Using Data Source: **{source}**")

# ------------------------------------------------------------
# PREVIEW
# ------------------------------------------------------------
st.header("📋 Data Preview")
st.dataframe(df.head(), use_container_width=True)


# ------------------------------------------------------------
# TRANSACTION LOOKUP
# ------------------------------------------------------------
st.header("🔍 Transaction Lookup")

txn_col = find_column(df, ["transactionid", "txn"])
if txn_col:
    txn = st.selectbox("Select Transaction ID", df[txn_col].astype(str).unique())
    row = df[df[txn_col].astype(str) == txn].iloc[0]

    st.subheader("Selected Transaction")
    st.dataframe(pd.DataFrame([row]))

    st.subheader("🤖 AI Explanation")
    st.info(ai_explain_transaction(row))
else:
    st.warning("Transaction ID column not found.")


# ------------------------------------------------------------
# QUICK STATS
# ------------------------------------------------------------
st.header("📊 Quick Stats")

c1, c2 = st.columns(2)
c1.metric("Total Rows", len(df))
c2.metric("Numeric Columns", len(df.select_dtypes(include='number').columns))


# ------------------------------------------------------------
# LEAKAGE
# ------------------------------------------------------------
st.header("🚨 Leakage Detection (AI)")

leak_df, leak_text = ai_leakage(df)
if leak_df is not None:
    st.dataframe(leak_df[["expected", "diff", "leak_pct"]].head())
    st.info(leak_text)
else:
    st.warning(leak_text)


# ------------------------------------------------------------
# FRAUD
# ------------------------------------------------------------
st.header("🕵️ Fraud Detection")
st.info(ai_fraud(df))


# ------------------------------------------------------------
# MONTHLY SUMMARY
# ------------------------------------------------------------
st.header("📅 Monthly Summary (AI)")
st.success(ai_summary(df))


# ------------------------------------------------------------
# CHART + AI EXPLAIN
# ------------------------------------------------------------
st.header("📈 Chart + AI Explanation")

num_cols = df.select_dtypes(include="number").columns
if len(num_cols) > 0:
    col = st.selectbox("Choose Numeric Column", num_cols)
    fig = px.line(df, y=col, title=f"{col} Trend")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🤖 AI Chart Explanation")
    st.info(ai_chart_explain(df, col))
else:
    st.warning("No numeric columns found.")


# ------------------------------------------------------------
# AI ASSISTANT
# ------------------------------------------------------------
st.header("🤖 AI Assistant — Ask Anything")
q = st.text_input("Ask a question about your data...")
if q:
    st.success(ai_chat(q, df))
