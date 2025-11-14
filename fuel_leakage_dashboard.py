import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client
import google.generativeai as genai
import re

# -------------------------------------------------
# Load Secrets
# -------------------------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

genai.configure(api_key=GEMINI_API_KEY)
gmodel = genai.GenerativeModel("gemini-2.0-flash")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------------------------
# Layout
# -------------------------------------------------
st.set_page_config(page_title="Universal Data Dashboard", layout="wide")
st.title("📄 Universal Fuel / Sales / Transaction Dashboard")

# ------------------- SIDEBAR ---------------------
st.sidebar.header("Upload File")
uploaded = st.sidebar.file_uploader("Upload CSV / Excel (ANY structure supported)", type=["csv", "xlsx"])

st.sidebar.write("---")
st.sidebar.header("🤖 Ask AI")
q = st.sidebar.text_input("Ask anything about the data:")

if st.sidebar.button("Ask AI"):
    try:
        ans = gmodel.generate_content(q)
        st.sidebar.success(ans.text)
    except Exception as e:
        st.sidebar.error(str(e))

# -------------------------------------------------
# If File Uploaded
# -------------------------------------------------
if uploaded:

    # AUTO-DETECT & READ FILE (skip junk rows)
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded, skiprows=lambda x: x < 8)
        else:
            df = pd.read_excel(uploaded, skiprows=lambda x: x < 8)
    except:
        df = pd.read_excel(uploaded)

    # CLEAN COLUMNS
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "")
        .str.replace("-", "_")
    )

    # Convert everything to string to avoid Supabase datetime issue
    df = df.astype(str)

    # --------------------- Show Data ---------------------
    st.subheader("📊 Data Preview (Top 20 Rows)")
    st.dataframe(df.head(20))

    st.write("---")

    # ---------------- FIND TXN IDs ANYWHERE ----------------
    st.subheader("🔎 Search by Transaction ID")

    # Extract TXN pattern from ANY column
    txn_ids = []

    for col in df.columns:
        txn_ids += df[col].astype(str).str.findall(r"(TXN\d{6,20})").sum()

    txn_ids = sorted(list(set(txn_ids)))

    if len(txn_ids) == 0:
        st.warning("⚠ No valid Transaction ID found. Showing full data only.")
    else:
        selected_txn = st.selectbox("Select Transaction ID:", txn_ids)
        result = df[df.apply(lambda row: selected_txn in row.values, axis=1)]
        st.write("### Transaction Details")
        st.dataframe(result)

    st.write("---")

    # ---------------------- CHARTS -----------------------
    st.subheader("📈 Charts")

    # Detect numeric columns
    num_cols = []
    for col in df.columns:
        try:
            df[col].astype(float)
            num_cols.append(col)
        except:
            pass

    if len(num_cols) > 0:
        pie_col = st.selectbox("Pie Chart Column:", num_cols)
        pie_data = df[pie_col].value_counts().rename_axis("Value").reset_index(name="Count")

        fig = px.pie(pie_data, values="Count", names="Value", title=f"{pie_col} Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns found. Charts disabled.")

    st.write("---")

    # ---------------------- SUPABASE UPLOAD -----------------------
    if st.button("Upload to Supabase"):
        try:
            records = df.replace({np.nan: None}).to_dict(orient="records")
            supabase.table("universal_data").insert(records).execute()
            st.success("🎉 Uploaded Successfully!")
        except Exception as e:
            st.error(str(e))

    st.write("---")

    # ---------------------- AI ANALYSIS -----------------------
    st.header("🤖 AI Insights")

    user_q = st.text_input("Ask anything about your uploaded data:")

    if st.button("Get AI Answer"):
        preview = df.head(15).to_dict(orient="records")

        prompt = f"""
        You are an expert data analyst.
        Here is sample data:
        {preview}

        Answer the user's question clearly:
        {user_q}
        """

        try:
            ai_out = gmodel.generate_content(prompt)
            st.success(ai_out.text)
        except Exception as e:
            st.error(str(e))

else:
    st.info("Upload a CSV or Excel file to begin.")
