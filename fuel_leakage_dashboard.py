import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai

# --------------------------
# Load Secrets
# --------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

genai.configure(api_key=GEMINI_API_KEY)
gmodel = genai.GenerativeModel("gemini-1.5-flash")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="Fuel / Sales Dashboard", layout="wide")
st.title("⛽ Fuel / Sales Transaction Dashboard")

# --------------------------
# Sidebar – Data Source
# --------------------------
st.sidebar.header("Select Data Source")

tables = ["sales_data", "trip_data", "driver_summary"]

selected_table = st.sidebar.selectbox("Choose Supabase Table", tables)

uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.write("---")
st.sidebar.header("🤖 ChatbotGemini")
ai_q = st.sidebar.text_input("Ask anything about the data:")

if st.sidebar.button("Ask (ChatbotGemini)"):
    st.sidebar.write("Thinking...")

# --------------------------
# Load Data from Supabase
# --------------------------
def load_supabase_table(table):
    try:
        response = supabase.table(table).select("*").execute()
        df = pd.DataFrame(response.data)
        return df
    except Exception as e:
        st.error(f"Supabase load error: {e}")
        return pd.DataFrame()

# --------------------------
# Normalize column names
# --------------------------
def clean_columns(df):
    rename_map = {
        "S.No": "s_no",
        "Transaction ID": "transaction_id",
        "Transaction Date": "transaction_date",
        "Transaction Time": "transaction_time"
    }
    df.rename(columns=rename_map, inplace=True)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

# --------------------------
# Choose File OR Supabase
# --------------------------
if uploaded:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded, skiprows=9)
        else:
            df = pd.read_excel(uploaded, skiprows=9)
    except:
        st.error("File read error")
        st.stop()
else:
    df = load_supabase_table(selected_table)

if df.empty:
    st.error("No data found. Check table or file.")
    st.stop()

df = clean_columns(df)

# --------------------------
# Show Data
# --------------------------
st.subheader("📌 Data Preview")
st.dataframe(df)

# --------------------------
# Search by Transaction ID
# --------------------------
if "transaction_id" in df.columns:
    st.subheader("🔍 Search by Transaction ID")
    txns = df["transaction_id"].dropna().unique()
    selected_txn = st.selectbox("Choose Transaction ID", txns)

    if selected_txn:
        st.dataframe(df[df["transaction_id"] == selected_txn])

# --------------------------
# Charts
# --------------------------
st.subheader("📊 Charts")

num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

if num_cols:
    selected_col = st.selectbox("Select column for pie chart", num_cols)
    pie_data = df[selected_col].value_counts().reset_index()
    fig = px.pie(pie_data, names="index", values=selected_col)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Chatbot Gemini
# --------------------------
st.subheader("🤖 AI Insights – ChatbotGemini")

user_q = st.text_input("Ask something:")

if st.button("Ask AI"):
    sample = df.head(20).to_dict(orient="records")

    prompt = f"""
    You are a data analysis expert.
    Here is the sample data:
    {sample}

    Answer the user clearly:
    {user_q}
    """

    out = gmodel.generate_content(prompt)
    st.write(out.text)
