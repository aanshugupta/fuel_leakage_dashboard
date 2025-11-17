import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai

# ---------------------------------------
# LIGHT BEAUTIFUL UI (Fix for Sidebar Text)
# ---------------------------------------
st.markdown("""
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #F5F7FA !important;
    color: #000 !important;
    padding-top: 20px;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: #000 !important;
    font-family: 'Segoe UI', sans-serif;
}

/* Main title */
h1, h2, h3 {
    color: #0A2A43 !important;
    font-family: 'Segoe UI', sans-serif;
}

/* Cards */
.card {
    padding: 15px;
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e5e5e5;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------
# CONFIG
# ------------------------------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

genai.configure(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL)


# ------------------------------------------------
# COMMON AI CALL
# ------------------------------------------------
def ai_answer(prompt):
    try:
        ans = model.generate_content(prompt).text
        return ans
    except Exception as e:
        return f"AI Error: {str(e)}"


# ------------------------------------------------
# LOAD FROM SUPABASE
# ------------------------------------------------
def load_table(table):
    try:
        resp = supabase.table(table).select("*").execute()
        return pd.DataFrame(resp.data)
    except:
        return pd.DataFrame([])


# ------------------------------------------------
# AI FUNCTIONS
# ------------------------------------------------
def ai_explain(row):
    prompt = f"""
    Explain this transaction in simple language.
    No JSON, no dictionary.
    Just a clean human summary.

    Transaction:
    {row.to_dict()}
    """
    return ai_answer(prompt)


def ai_leakage(df):
    needed = ["product_volume", "rate", "purchase_amount"]
    if not all(col in df.columns for col in needed):
        return None, "Required columns missing (volume, rate, amount)"

    df["expected"] = df["product_volume"] * df["rate"]
    df["diff"] = df["purchase_amount"] - df["expected"]
    df["leak_pct"] = (df["diff"] / df["expected"]) * 100

    prompt = f"""
    Analyse fuel leakage % and explain in simple text:
    DO NOT return JSON.

    Data:
    {df[['product_volume','rate','purchase_amount','leak_pct']].to_dict()}
    """
    return df, ai_answer(prompt)


def ai_fraud(df):
    prompt = f"""
    Detect fraud or suspicious fuel data.
    Give result in simple natural language only.
    Do NOT return JSON.

    Dataset:
    {df.to_dict()}
    """
    return ai_answer(prompt)


def ai_summary(df):
    prompt = f"""
    Create a clean monthly summary.
    No JSON output.
    Use bullet points + short paragraphs.
    Very easy for a normal non-technical person.

    Data:
    {df.to_dict()}
    """
    return ai_answer(prompt)


def ai_chat(question, df):
    prompt = f"""
    You are an expert fuel analytics AI.
    Answer ONLY using this dataset.
    No JSON. Clean simple explanation.

    Dataset:
    {df.to_dict()}
    Question: {question}
    """
    return ai_answer(prompt)


# ------------------------------------------------
# MAIN UI
# ------------------------------------------------
st.title("⛽ Fuel Intelligence Dashboard — Premium Edition")


# ------------------------------------------------
# SIDEBAR DATA CONTROLS
# ------------------------------------------------
st.sidebar.header("🗂️ Data Controls")

table_name = st.sidebar.selectbox(
    "Select Supabase Table",
    ["sales_data", "trip_data"]
)

df_supabase = load_table(table_name)
st.sidebar.success(f"Loaded {len(df_supabase)} rows")


# ------------------------------------------------
# SIDEBAR — FILE UPLOAD
# ------------------------------------------------
st.sidebar.header("📤 Upload CSV / Excel")

uploaded = st.sidebar.file_uploader(
    "Upload file",
    type=["csv", "xlsx"]
)

df_uploaded = None
if uploaded:
    if uploaded.name.endswith(".csv"):
        df_uploaded = pd.read_csv(uploaded)
    else:
        df_uploaded = pd.read_excel(uploaded)
    st.sidebar.success(f"Uploaded rows: {len(df_uploaded)}")


# Priority: Uploaded > Supabase
df = df_uploaded.copy() if df_uploaded is not None else df_supabase.copy()

st.info(f"Using data source: **{'Uploaded File' if df_uploaded is not None else 'Supabase Table'}**")


# ------------------------------------------------
# DATA PREVIEW
# ------------------------------------------------
st.subheader("📋 Data Preview")
st.dataframe(df.head(50))


# ------------------------------------------------
# TRANSACTION LOOKUP FIXED
# ------------------------------------------------
st.subheader("🔍 Transaction Lookup")

txn_col = None
for c in df.columns:
    if ("transaction" in c.lower()) and ("id" in c.lower()):
        txn_col = c
        break

if txn_col:
    txn_id = st.selectbox("Choose Transaction ID", df[txn_col].astype(str).unique())
    row = df[df[txn_col].astype(str) == txn_id].iloc[0]

    st.write("### Details")
    st.dataframe(pd.DataFrame([row]))

    st.write("### 🧠 AI Explanation")
    st.info(ai_explain(row))


# ------------------------------------------------
# QUICK STATS
# ------------------------------------------------
st.subheader("📊 Quick Stats")

num_cols = df.select_dtypes(include=['float64','int64']).columns

st.metric("Rows", len(df))
st.metric("Numeric Columns", len(num_cols))


# ------------------------------------------------
# Leakage Detection
# ------------------------------------------------
st.subheader("🚨 Fuel Leakage Detection (AI)")
leak_df, leak_text = ai_leakage(df)

if leak_df is not None:
    st.dataframe(leak_df[["product_volume","rate","purchase_amount","leak_pct"]])
    st.error(leak_text)
else:
    st.warning(leak_text)


# ------------------------------------------------
# Fraud Analysis
# ------------------------------------------------
st.subheader("🕵️ Fraud Detection (AI)")
st.warning(ai_fraud(df))


# ------------------------------------------------
# Monthly Summary
# ------------------------------------------------
st.subheader("📅 Monthly Summary (AI)")
st.info(ai_summary(df))


# ------------------------------------------------
# Charts
# ------------------------------------------------
st.subheader("📈 Chart Builder (Simple & Clean)")

if len(num_cols) > 0:
    sel_col = st.selectbox("Choose numeric column", num_cols)
    chart_type = st.radio("Chart Type", ["Line","Bar","Area"])

    if chart_type == "Line":
        fig = px.line(df, y=sel_col)
    elif chart_type == "Bar":
        fig = px.bar(df, y=sel_col)
    else:
        fig = px.area(df, y=sel_col)

    st.plotly_chart(fig, use_container_width=True)

    # AI Explanation of Chart
    st.write("### 🧠 AI Explanation of Chart")
    st.info(
        ai_answer(
            f"Explain this chart in simple words for a non-technical person. Column: {sel_col}. Data: {df[sel_col].tolist()}"
        )
    )
else:
    st.warning("No numeric columns available.")


# ------------------------------------------------
# AI CHATBOT
# ------------------------------------------------
st.subheader("🤖 AI Assistant — Ask Anything")

user_q = st.text_input("Your question about the data:")

if user_q:
    st.success(ai_chat(user_q, df))
