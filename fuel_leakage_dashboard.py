import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client
import google.generativeai as genai

# -----------------------
# Load Secrets
# -----------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# -----------------------
# Configure Gemini
# -----------------------
genai.configure(api_key=GEMINI_API_KEY)
gmodel = genai.GenerativeModel("gemini-2.0-flash")

# -----------------------
# Configure Supabase
# -----------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Fuel | Sales Dashboard", layout="wide")
st.title("⛽ Fuel / Sales Transaction Dashboard")

# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("📂 Select Data Source")

# --- FETCH TABLE LIST FROM SUPABASE ---
tables = ["sales_data", "trip_data", "driver_summary"]

selected_table = st.sidebar.selectbox("Choose Supabase Table", tables)

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.write("---")
st.sidebar.header("🤖 ChatbotGemini")

chat_query = st.sidebar.text_input("Ask anything about the data:")

# -----------------------
# LOAD DATA
# -----------------------
df = pd.DataFrame()

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, skiprows=9)
        else:
            df = pd.read_excel(uploaded_file, skiprows=9)
        st.success("File loaded successfully!")
    except Exception as e:
        st.error(f"Error reading upload: {e}")

else:
    # Load from Supabase table
    try:
        response = supabase.table(selected_table).select("*").execute()
        df = pd.DataFrame(response.data)
        st.success(f"Loaded data from Supabase: {selected_table}")
    except Exception as e:
        st.error(f"Supabase loading error: {e}")

# -----------------------
# DATA CLEANING
# -----------------------
if not df.empty:

    # Fix column names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    st.subheader("📊 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # -----------------------
    # SUMMARY STATS
    # -----------------------
    st.subheader("📌 Quick Summary")
    st.write(df.describe(include="all"))

    # -----------------------
    # TRANSACTION ID SEARCH
    # -----------------------
    if "transaction_id" in df.columns:
        st.subheader("🔍 Search Transaction")
        ids = df["transaction_id"].dropna().astype(str).unique().tolist()
        selected_txn = st.selectbox("Choose Transaction ID:", ids)
        txn_data = df[df["transaction_id"] == selected_txn]
        st.dataframe(txn_data)

    # -----------------------
    # CHARTS SECTION
    # -----------------------
    st.subheader("📈 Charts")

    numeric_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()

    if numeric_cols:
        col = st.selectbox("Choose column for Pie Chart:", numeric_cols)
        pie_data = df[col].value_counts().reset_index()
        pie_data.columns = ["Value", "Count"]
        fig_pie = px.pie(pie_data, names="Value", values="Count", title=f"{col} Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    # -----------------------
    # PROFIT / LOSS DETECTION
    # -----------------------
    if "amount" in df.columns and "actual_fuel_liters" in df.columns:
        st.subheader("💹 Profit / Loss Analysis")

        df["profit_loss"] = df["amount"] - (df["actual_fuel_liters"] * 85)

        fig = px.histogram(df, x="profit_loss", nbins=50,
                           title="Profit / Loss Distribution")
        st.plotly_chart(fig)

# -----------------------
# GEMINI AI CHATBOT
# -----------------------
if st.sidebar.button("Ask (ChatbotGemini)"):

    if df.empty:
        st.sidebar.error("No data available!")
    else:
        sample = df.head(20).to_dict(orient="records")

        prompt = f"""
        You are ChatbotGemini.
        You are an expert analyst. Use ONLY this data:

        SAMPLE DATA:
        {sample}

        User question:
        {chat_query}

        Answer clearly and only about THIS dataset.
        """

        reply = gmodel.generate_content(prompt)
        st.sidebar.success(reply.text)
