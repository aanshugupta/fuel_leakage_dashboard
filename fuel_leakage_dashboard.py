import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client
import google.generativeai as genai

# =========================
# CONFIG
# =========================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)

MODEL = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL)


# =========================
# LOAD DATA FROM SUPABASE
# =========================
def load_table(table_name):
    try:
        res = supabase.table(table_name).select("*").execute()
        return pd.DataFrame(res.data)
    except Exception as e:
        st.error(f"Supabase error: {e}")
        return pd.DataFrame()


# =========================
# SAFE AI PROMPT HANDLER
# =========================
def safe_ai_request(prompt):
    try:
        return model.generate_content(prompt).text
    except Exception:
        return "⚠ The AI model could not process full data. Try asking a shorter question."


# =========================
# AI LEAKAGE DETECTION
# =========================
def ai_leakage(df):
    req = ["product_volume", "rate", "purchase_amount"]

    if not all(col in df.columns for col in req):
        return None, "Required columns missing."

    df["expected"] = df["product_volume"] * df["rate"]
    df["diff"] = df["purchase_amount"] - df["expected"]
    df["leak_pct"] = np.where(df["expected"] == 0, 0,
                              (df["diff"] / df["expected"]) * 100)

    sample = df.head(50).to_json()

    prompt = f"""
    Analyze leakage patterns from fuel transaction sample:
    {sample}

    Give:
    • Any leakage alerts  
    • Overcharging patterns  
    • Suspicious transactions  
    """
    ans = safe_ai_request(prompt)
    return df, ans


# =========================
# AI TRANSACTION EXPLAINER
# =========================
def ai_explain(row):
    prompt = f"""
    Explain this fuel transaction in simple language:
    {row.to_dict()}
    """
    return safe_ai_request(prompt)


# =========================
# AI FRAUD DETECTION
# =========================
def ai_fraud(df):
    sample = df.head(80).to_json()

    prompt = f"""
    Detect fraud, anomalies, duplicate records, abnormal amounts.
    Data sample:
    {sample}
    """
    return safe_ai_request(prompt)


# =========================
# AI MONTHLY SUMMARY
# =========================
def ai_summary(df):
    sample = df.head(100).to_json()

    prompt = f"""
    Create a clear monthly summary for non-technical users.
    Include:
    • Total fuel usage  
    • High/low spending  
    • Best performing vehicles  
    • Any red flags  

    Data:
    {sample}
    """
    return safe_ai_request(prompt)


# =========================
# AI CHATBOT
# =========================
def ai_chat(q, df):
    sample = df.head(120).to_json()

    prompt = f"""
    You are an expert fuel data analyst.  
    Answer the question using ONLY this dataset sample:

    {sample}

    Question: {q}
    """
    return safe_ai_request(prompt)


# =========================
# UI STARTS
# =========================
st.title("⛽ Fuel Intelligence Dashboard — Premium Pack")


# --------------------
# DATA CONTROLS
# --------------------
st.sidebar.header("📦 Data Controls")

table_sel = st.sidebar.selectbox("Select Supabase Table", ["sales_data", "trip_data"])
df_supabase = load_table(table_sel)

st.sidebar.success(f"Loaded {len(df_supabase)} rows")


# --------------------
# FILE UPLOAD
# --------------------
st.sidebar.header("📤 Upload CSV / Excel (NEW)")

upload = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

df_file = None
if upload:
    if upload.name.endswith(".csv"):
        df_file = pd.read_csv(upload)
    else:
        df_file = pd.read_excel(upload)
    st.sidebar.success(f"Uploaded {upload.name} — {len(df_file)} rows")


# --------------------
# SELECT DATA SOURCE
# --------------------
if df_file is not None:
    df = df_file.copy()
    st.info("Using **Uploaded File**")
else:
    df = df_supabase.copy()
    st.info("Using **Supabase Table**")


# CONVERT numeric columns
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    except:
        pass


# --------------------
# DATA PREVIEW
# --------------------
st.subheader("📋 Data Preview")
st.dataframe(df.head(50))


# --------------------
# TRANSACTION LOOKUP
# --------------------
st.subheader("🔍 Transaction Lookup")

txn_col = None
for c in df.columns:
    if "transaction" in c.lower() and "id" in c.lower():
        txn_col = c
        break

if txn_col:
    txn = st.selectbox("Select Transaction ID", df[txn_col].astype(str).unique())
    row = df[df[txn_col].astype(str) == txn].iloc[0]

    st.write("### Transaction Details")
    st.dataframe(pd.DataFrame([row]))

    st.write("### 🧠 AI Explanation")
    st.info(ai_explain(row))


# --------------------
# QUICK STATS
# --------------------
st.subheader("📊 Quick Stats")

st.metric("Rows", len(df))
st.metric("Numeric Columns", len(df.select_dtypes(include=["int64", "float64"]).columns))


# --------------------
# LEAKAGE DETECTION
# --------------------
st.subheader("🚨 AI Leakage Detection")

leak_df, text = ai_leakage(df)

if leak_df is not None:
    st.dataframe(leak_df[["product_volume", "rate", "purchase_amount", "leak_pct"]])
    st.error(text)
else:
    st.warning(text)


# --------------------
# FRAUD ANALYSIS
# --------------------
st.subheader("🕵️ Fraud Detection")
st.warning(ai_fraud(df))


# --------------------
# MONTHLY SUMMARY
# --------------------
st.subheader("🗓 Monthly Summary (AI)")
st.info(ai_summary(df))


# --------------------
# CHART BUILDER
# --------------------
st.subheader("📈 Simple Charts (Easy to Understand)")

num_cols = df.select_dtypes(include=["int64", "float64"]).columns

if len(num_cols) > 0:
    col_sel = st.selectbox("Choose Numeric Column", num_cols)
    chart_t = st.radio("Chart Type", ["Line", "Bar", "Area"])

    if chart_t == "Line":
        fig = px.line(df, y=col_sel)
    elif chart_t == "Bar":
        fig = px.bar(df, y=col_sel)
    else:
        fig = px.area(df, y=col_sel)

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No numeric columns available.")


# --------------------
# AI CHATBOT
# --------------------
st.subheader("🤖 AI Assistant (Ask Anything)")

q = st.text_input("Ask something about your data…")

if q:
    st.success(ai_chat(q, df))
