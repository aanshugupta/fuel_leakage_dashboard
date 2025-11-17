import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai

# -----------------------------
# CONFIG
# -----------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)

MODEL = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL)


# -----------------------------
# LOAD FROM SUPABASE
# -----------------------------
def load_table(table_name):
    try:
        response = supabase.table(table_name).select("*").execute()
        return pd.DataFrame(response.data)
    except:
        return pd.DataFrame([])


# -----------------------------
# AI FUNCTIONS
# -----------------------------
def ai_leakage(df):
    if "product_volume" not in df.columns or "rate" not in df.columns or "purchase_amount" not in df.columns:
        return None, "Required columns missing."

    df["expected"] = df["product_volume"] * df["rate"]
    df["diff"] = df["purchase_amount"] - df["expected"]
    df["leak_pct"] = (df["diff"] / df["expected"]) * 100

    prompt = f"Analyze leakage & patterns:\n{df.to_dict()}"
    ans = model.generate_content(prompt).text
    return df, ans


def ai_explain(row):
    prompt = f"Explain this transaction clearly:\n{row.to_dict()}"
    return model.generate_content(prompt).text


def ai_fraud(df):
    prompt = f"Detect fraud & anomalies:\n{df.to_dict()}"
    return model.generate_content(prompt).text


def ai_summary(df):
    prompt = f"Monthly summary insight:\n{df.to_dict()}"
    return model.generate_content(prompt).text


def ai_chat(question, df):
    prompt = f"""
    You are a fuel dataset expert.
    Use ONLY this dataset when replying.

    Dataset:
    {df.to_dict()}

    Question: {question}
    """
    return model.generate_content(prompt).text


# -----------------------------
# UI LAYOUT
# -----------------------------
st.title("⛽ Fuel Intelligence Dashboard — Premium Pack")

# -----------------------------
# DATA CONTROLS
# -----------------------------
st.sidebar.header("📦 Data Controls")

table_choice = st.sidebar.selectbox(
    "Select Supabase Table",
    ["sales_data", "trip_data"]
)

df_supabase = load_table(table_choice)
st.sidebar.success(f"Loaded {len(df_supabase)} rows")


# -----------------------------
# NEW FEATURE 1:
# UPLOAD CSV / EXCEL (FULL FUNCTIONALITY)
# -----------------------------
st.sidebar.header("📤 Upload CSV / Excel (NEW)")

uploaded = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

df_uploaded = None

if uploaded:
    if uploaded.name.endswith(".csv"):
        df_uploaded = pd.read_csv(uploaded)
    else:
        df_uploaded = pd.read_excel(uploaded)

    st.sidebar.success(f"Uploaded file loaded: {len(df_uploaded)} rows")


# -----------------------------
# PRIORITY LOGIC:
# If upload exists → use upload
# Else → use Supabase
# -----------------------------
if df_uploaded is not None:
    df = df_uploaded.copy()
    data_source = "Uploaded File"
else:
    df = df_supabase.copy()
    data_source = "Supabase Table"

st.info(f"Using Data Source: **{data_source}**")

# Show data preview
st.subheader("📋 Data Preview")
st.dataframe(df.head(50))


# -----------------------------
# TRANSACTION LOOKUP
# -----------------------------
st.subheader("🔍 Transaction Lookup")

txn_col = None
for c in df.columns:
    if "transaction" in c.lower() and "id" in c.lower():
        txn_col = c
        break

if txn_col:
    txn_id = st.selectbox("Select Transaction ID", df[txn_col].astype(str).unique())

    row = df[df[txn_col].astype(str) == txn_id].iloc[0]
    st.write("### Transaction Details")
    st.dataframe(pd.DataFrame([row]))

    st.write("### 🧠 AI Explanation")
    st.info(ai_explain(row))


# -----------------------------
# QUICK STATS
# -----------------------------
st.subheader("📊 Quick Stats")
st.metric("Rows", len(df))
st.metric("Numeric Columns", len(df.select_dtypes(include=['int64','float64']).columns))


# -----------------------------
# LEAKAGE AI
# -----------------------------
st.subheader("🚨 AI Leakage Detection")

leak_df, leak_text = ai_leakage(df)

if leak_df is not None:
    st.dataframe(leak_df[["product_volume", "rate", "purchase_amount", "leak_pct"]])
    st.error(leak_text)
else:
    st.warning(leak_text)


# -----------------------------
# FRAUD ANALYSIS
# -----------------------------
st.subheader("🕵️ Fraud Detection")
st.warning(ai_fraud(df))


# -----------------------------
# MONTHLY SUMMARY
# -----------------------------
st.subheader("🗓 Monthly Summary AI")
st.info(ai_summary(df))


# -----------------------------
# CHART BUILDER
# -----------------------------
st.subheader("📈 Chart Builder")

num_cols = df.select_dtypes(include=['int64','float64']).columns

if len(num_cols) == 0:
    st.warning("No numeric columns found!")
else:
    selected = st.selectbox("Select column", num_cols)
    chart_type = st.radio("Chart Type", ["Line", "Bar", "Area"])

    if chart_type == "Line":
        fig = px.line(df, y=selected)
    elif chart_type == "Bar":
        fig = px.bar(df, y=selected)
    else:
        fig = px.area(df, y=selected)

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# AI CHATBOT (WORKS FOR SUPABASE + UPLOAD BOTH)
# -----------------------------
st.subheader("🤖 AI Assistant")

q = st.text_input("Ask something about your data")

if q:
    st.success(ai_chat(q, df))
