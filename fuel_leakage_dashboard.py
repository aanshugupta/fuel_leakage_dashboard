import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai
import json

# -----------------------------------------------------------
# LOAD SECRETS
# -----------------------------------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

if not GEMINI_API_KEY:
    st.error("Gemini API key missing!")
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase credentials missing!")

# -----------------------------------------------------------
# GEMINI SETUP - MODEL FIXED
# -----------------------------------------------------------
genai.configure(api_key=GEMINI_API_KEY)
gmodel = genai.GenerativeModel("gemini-2.5-flash")

# -----------------------------------------------------------
# SUPABASE CONNECT
# -----------------------------------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------------------------------------
# UI SETTINGS
# -----------------------------------------------------------
st.set_page_config(page_title="Sales Transaction Dashboard", layout="wide")
st.title("📊 Universal Sales Transaction Dashboard")

# -----------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------
st.sidebar.header("📂 Upload File")
uploaded = st.sidebar.file_uploader("Upload Excel/CSV", type=["csv", "xlsx"])

st.sidebar.write("---")
st.sidebar.header("🤖 Ask AI (General)")
side_q = st.sidebar.text_input("Ask here")

if st.sidebar.button("Ask AI"):
    try:
        resp = gmodel.generate_content(side_q)
        st.sidebar.success(resp.text)
    except:
        st.sidebar.error("AI Error. Check model or API key.")

# -----------------------------------------------------------
# WHEN FILE UPLOADED
# -----------------------------------------------------------
if uploaded:

    # -------------------------------------------------------
    # READ FILE — SKIP FIRST 9 META ROWS
    # -------------------------------------------------------
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded, skiprows=9)
        else:
            df = pd.read_excel(uploaded, skiprows=9)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # -------------------------------------------------------
    # CLEAN COLUMN NAMES
    # -------------------------------------------------------
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    rename_map = {
        "sno": "s_no",
        "s.no": "s_no",
        "txn_id": "transaction_id",
        "transactionid": "transaction_id",
        "transaction id": "transaction_id",
        "transaction_no": "transaction_id",
    }
    df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)

    # -------------------------------------------------------
    # CHECK TRANSACTION ID COLUMN EXISTS
    # -------------------------------------------------------
    txn_col = None
    for col in df.columns:
        if "transaction" in col and "id" in col:
            txn_col = col
            break

    if txn_col is None:
        st.error("❌ Could not detect 'Transaction ID' column!")
        st.stop()

    df.rename(columns={txn_col: "transaction_id"}, inplace=True)

    st.success("✔ Transaction ID detected")

    # -------------------------------------------------------
    # SHOW CLEANED TABLE
    # -------------------------------------------------------
    st.subheader("🧾 Processed Table")
    st.dataframe(df, use_container_width=True)

    st.write("---")

    # -------------------------------------------------------
    # SUPABASE UPLOAD
    # -------------------------------------------------------
    if st.button("⬆ Upload Data to Supabase"):
        try:
            clean = df.replace({np.nan: None}).to_dict(orient="records")
            supabase.table("sales_data").insert(clean).execute()
            st.success("🎉 Upload Successful!")
        except Exception as e:
            st.error(str(e))

    st.write("---")

    # -------------------------------------------------------
    # SEARCH BY TRANSACTION ID
    # -------------------------------------------------------
    st.header("🔍 Search Transaction")

    txn_list = df["transaction_id"].dropna().astype(str).unique().tolist()
    selected_id = st.selectbox("Select Transaction ID", txn_list)

    if selected_id:
        st.subheader("📌 Transaction Detail")
        result = df[df["transaction_id"] == selected_id]
        st.dataframe(result, use_container_width=True)

    st.write("---")

    # -------------------------------------------------------
    # CHARTS SECTION
    # -------------------------------------------------------
    st.header("📈 Charts")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if numeric_cols:
        chosen = st.selectbox("Select column for Pie Chart", numeric_cols)
        pie = df[chosen].value_counts().reset_index()
        pie.columns = ["Value", "Count"]
        fig = px.pie(pie, values="Count", names="Value", title=f"{chosen} Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numeric columns found for charts")

    st.write("---")

    # -------------------------------------------------------
    # AI INSIGHTS BASED ON FILE DATA (FIXED PROMPT)
    # -------------------------------------------------------
    st.header("🤖 AI Insights on Uploaded Data")

    user_q = st.text_input("Ask something like: 'Which transaction has max amount?'")

    if st.button("Get AI Insights"):

        # FULL DATA SUMMARY SENT TO AI
        preview = df.to_dict(orient="records")

        prompt = f"""
        You are a data expert. 
        Answer ONLY using the data provided below.
        Do NOT ask the user again for more context.

        Here is the dataset:
        {json.dumps(preview, indent=2)}

        User question: {user_q}

        Give clear, correct answer based strictly on this dataset.
        """

        try:
            out = gmodel.generate_content(prompt)
            st.success(out.text)
        except Exception as e:
            st.error(str(e))

else:
    st.info("Upload a file to begin.")
