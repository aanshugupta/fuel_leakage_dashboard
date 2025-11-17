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
# HELPER → semantic column finder
# ------------------------------------------------------------
def find_column(df, keywords):
    keywords = [k.lower() for k in keywords]
    for col in df.columns:
        name = col.lower().replace(" ", "").replace("_", "")
        for k in keywords:
            if k in name:
                return col
    return None


# ------------------------------------------------------------
# LOAD FROM SUPABASE
# ------------------------------------------------------------
def load_table(table):
    try:
        res = supabase.table(table).select("*").execute()
        return pd.DataFrame(res.data)
    except:
        return pd.DataFrame()


# ------------------------------------------------------------
# AI FUNCTIONS
# ------------------------------------------------------------
def ai_answer(prompt):
    return model.generate_content(prompt).text


def ai_explain_transaction(row):
    prompt = f"Explain this fuel transaction for a non-technical user:\n{row.to_dict()}"
    return ai_answer(prompt)


def ai_fraud(df):
    prompt = f"Detect fraud patterns, duplicate transactions, misuse, pricing anomalies:\n{df.to_dict()}"
    return ai_answer(prompt)


def ai_summary(df):
    prompt = f"Give a clean monthly summary:\n{df.to_dict()}"
    return ai_answer(prompt)


def ai_leakage(df):
    qty = find_column(df, ["volume", "qty", "litre", "quantity"])
    rate = find_column(df, ["rate", "price"])
    amt = find_column(df, ["amount", "purchase", "total_transaction"])

    if not qty or not rate or not amt:
        return None, "Required columns not found."

    df["expected"] = df[qty] * df[rate]
    df["diff"] = df[amt] - df["expected"]
    df["leak_pct"] = (df["diff"] / df["expected"]) * 100

    prompt = f"Analyze leakage and abnormalities:\n{df.to_dict()}"
    return df, ai_answer(prompt)


def ai_chart_explain(df, column):
    prompt = f"""
    Explain this chart in very simple words for a non-technical person.
    Column: {column}
    Data sample: {df[column].head().to_list()}
    """
    return ai_answer(prompt)


def ai_chat(question, df):
    prompt = f"Dataset:\n{df.to_dict()}\nAnswer this: {question}"
    return ai_answer(prompt)


# ------------------------------------------------------------
# BEAUTIFUL UI THEME
# ------------------------------------------------------------
st.set_page_config(
    page_title="Fuel Dashboard",
    layout="wide",
    page_icon="⛽"
)

st.markdown("""
<style>
/* Sidebar beautify */
[data-testid="stSidebar"] {
    background-color: #082F49;
    padding: 20px;
}

h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
}

/* Card style */
.card {
    padding: 15px;
    background: white;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.title("📦 Data Controls")

table = st.sidebar.selectbox(
    "Choose Supabase Table",
    ["sales_data", "trip_data"]
)

df_supabase = load_table(table)
st.sidebar.success(f"Loaded {len(df_supabase)} rows")


# Upload section
st.sidebar.markdown("### 📤 Upload CSV / Excel (Optional)")
uploaded = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx"])
df_uploaded = None

if uploaded:
    if uploaded.name.endswith(".csv"):
        df_uploaded = pd.read_csv(uploaded)
    else:
        df_uploaded = pd.read_excel(uploaded)
    st.sidebar.success(f"Uploaded {len(df_uploaded)} rows")


# ------------------------------------------------------------
# PRIORITY: Uploaded → Supabase
# ------------------------------------------------------------
df = df_uploaded if df_uploaded is not None else df_supabase
source = "Uploaded File" if df_uploaded is not None else "Supabase"

st.markdown(f"### 📌 Using Data Source: **{source}**")


# ------------------------------------------------------------
# SECTION: Data Preview
# ------------------------------------------------------------
st.markdown("## 📋 Data Preview")
st.dataframe(df.head(), use_container_width=True)


# ------------------------------------------------------------
# TRANSACTION LOOKUP
# ------------------------------------------------------------
st.markdown("## 🔍 Transaction Lookup")

txn_col = find_column(df, ["transactionid", "txn"])
if txn_col:
    txn_id = st.selectbox("Select Transaction ID", df[txn_col].astype(str).unique())
    row = df[df[txn_col].astype(str) == txn_id].iloc[0]

    st.markdown("#### Selected Transaction Details")
    st.dataframe(pd.DataFrame([row]), use_container_width=True)

    st.markdown("#### 🤖 AI Explanation")
    st.info(ai_explain_transaction(row))
else:
    st.warning("No Transaction ID column found.")


# ------------------------------------------------------------
# QUICK STATS
# ------------------------------------------------------------
st.markdown("## 📊 Quick Stats")

col1, col2 = st.columns(2)
col1.metric("Total Rows", len(df))
col2.metric("Numeric Columns", len(df.select_dtypes(include='number').columns))


# ------------------------------------------------------------
# LEAKAGE DETECTION
# ------------------------------------------------------------
st.markdown("## 🚨 Leakage Detection (AI)")

leak_df, leak_text = ai_leakage(df)

if leak_df is not None:
    st.dataframe(leak_df[["expected", "diff", "leak_pct"]].head())
    st.error(leak_text)
else:
    st.warning(leak_text)


# ------------------------------------------------------------
# FRAUD DETECTION
# ------------------------------------------------------------
st.markdown("## 🕵️ Fraud Detection")
st.warning(ai_fraud(df))


# ------------------------------------------------------------
# MONTHLY SUMMARY
# ------------------------------------------------------------
st.markdown("## 🗓 Monthly Summary (AI)")
st.info(ai_summary(df))


# ------------------------------------------------------------
# CHART + AI EXPLANATION
# ------------------------------------------------------------
st.markdown("## 📈 Chart Builder + AI Explanation")

num_cols = df.select_dtypes(include="number").columns

if len(num_cols) > 0:
    selected = st.selectbox("Choose Column", num_cols)
    fig = px.line(df, y=selected, title=f"Chart of {selected}")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🤖 AI Explanation of Chart")
    st.success(ai_chart_explain(df, selected))
else:
    st.warning("No numeric columns for charts.")


# ------------------------------------------------------------
# AI ASSISTANT (Placed BELOW upload section)
# ------------------------------------------------------------
st.markdown("## 🤖 AI Assistant (Ask Anything)")

q = st.text_input("Ask something about your data…")

if q:
    st.success(ai_chat(q, df))
