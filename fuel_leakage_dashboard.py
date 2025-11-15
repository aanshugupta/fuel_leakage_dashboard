import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai

# ----------------------------------------------------
# Load Secrets
# ----------------------------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

if not GEMINI_API_KEY:
    st.error("Gemini API key missing!")
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase credentials missing!")

# ----------------------------------------------------
# Gemini Setup
# ----------------------------------------------------
genai.configure(api_key=GEMINI_API_KEY)
gmodel = genai.GenerativeModel("gemini-2.0-flash")

# ----------------------------------------------------
# Supabase Setup
# ----------------------------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------------------------------------
# App Page Setup
# ----------------------------------------------------
st.set_page_config(page_title="Fuel / Sales Transaction Dashboard", layout="wide")
st.title("⛽ Fuel / Sales Transaction Dashboard")

# ----------------------------------------------------
# Sidebar Upload Section
# ----------------------------------------------------
st.sidebar.header("Upload File")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

# -------------------------------
# Sidebar Chatbot Gemini
# -------------------------------
st.sidebar.write("---")
st.sidebar.header("🤖 Chatbot Gemini")

side_question = st.sidebar.text_input("Ask anything about the data:")

if st.sidebar.button("Ask (Chatbot Gemini)"):

    if uploaded:
        sample = df.head(15).to_dict(orient="records")
        cols = df.columns.tolist()

        prompt = f"""
        You are Chatbot Gemini.
        You analyze uploaded spreadsheet data.

        Columns:
        {cols}

        Sample Data:
        {sample}

        User Question:
        {side_question}

        Give a simple, direct answer based only on this data.
        """

        try:
            ai_side = gmodel.generate_content(prompt)
            st.sidebar.success(ai_side.text)
        except Exception as e:
            st.sidebar.error(str(e))
    else:
        st.sidebar.error("Upload a file first!")

# ----------------------------------------------------
# Process File When Uploaded
# ----------------------------------------------------
if uploaded:

    # READ FILE (skip first 9 junk rows)
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded, skiprows=9)
        else:
            df = pd.read_excel(uploaded, skiprows=9)
    except Exception as e:
        st.error(f"File Read Error: {e}")
        st.stop()

    # CLEAN HEADER
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # HANDLE COMMON COLUMN NAMES
    rename_map = {
        "sno": "s_no",
        "s.no": "s_no",
        "transaction_id": "transaction_id",
        "txn_id": "transaction_id",
    }

    df.rename(columns={col: rename_map.get(col, col) for col in df.columns}, inplace=True)

    # Show formatted table
    st.subheader("📄 Data Preview")
    st.dataframe(df, use_container_width=True)

    # ------------------------------------------------
    # Check Transaction ID Column
    # ------------------------------------------------
    if "transaction_id" not in df.columns:
        st.error("❌ Transaction ID column not found in file!")
    else:
        st.success("✔ Transaction ID detected")

    # ------------------------------------------------
    # Upload to Supabase
    # ------------------------------------------------
    if st.button("Upload to Supabase"):
        try:
            records = df.replace({np.nan: None}).to_dict(orient="records")
            supabase.table("trip_data").insert(records).execute()
            st.success("Data Uploaded Successfully!")
        except Exception as e:
            st.error(str(e))

    # ------------------------------------------------
    # Search by Transaction ID
    # ------------------------------------------------
    st.write("---")
    st.subheader("🔍 Search by Transaction ID")

    if "transaction_id" in df.columns:
        txn_list = df["transaction_id"].dropna().astype(str).unique().tolist()

        selected_txn = st.selectbox("Select Transaction ID", txn_list)

        if selected_txn:
            result = df[df["transaction_id"] == selected_txn]
            st.write("### Transaction Details")
            st.dataframe(result, use_container_width=True)

    # ------------------------------------------------
    # Charts
    # ------------------------------------------------
    st.write("---")
    st.subheader("📈 Charts")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if len(num_cols) > 0:
        pie_col = st.selectbox("Select column for Pie Chart", num_cols)

        pie_data = df[pie_col].value_counts().reset_index()
        pie_data.columns = ["Value", "Count"]

        fig_pie = px.pie(pie_data, names="Value", values="Count", title=f"{pie_col} Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("No numeric columns found for charts.")

    # ------------------------------------------------
    # Main AI Chat (Chatbot Gemini)
    # ------------------------------------------------
    st.write("---")
    st.header("🤖 Chatbot Gemini – AI Insights")

    user_q = st.text_input("Ask anything about your uploaded data:")

    if st.button("Get AI Answer"):

        sample_main = df.head(20).to_dict(orient="records")
        cols_main = df.columns.tolist()

        prompt_main = f"""
        You are Chatbot Gemini.
        Here is the structure of the uploaded spreadsheet.

        Columns:
        {cols_main}

        First 20 rows of data:
        {sample_main}

        User question:
        {user_q}

        Give a clear, simple answer based only on this dataset.
        """

        try:
            ai_main = gmodel.generate_content(prompt_main)
            st.success(ai_main.text)
        except Exception as e:
            st.error(str(e))

else:
    st.info("Please upload a CSV or Excel file to begin.")
