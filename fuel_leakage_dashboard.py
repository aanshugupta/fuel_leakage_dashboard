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
# Page Layout
# -------------------------------------------------
st.set_page_config(page_title="Transaction Dashboard", layout="wide")
st.title("📄 Fuel / Sales Transaction Dashboard")

# ------------------- SIDEBAR ---------------------
st.sidebar.header("Upload File")
uploaded = st.sidebar.file_uploader("Upload Excel / CSV", type=["csv", "xlsx"])

st.sidebar.write("--")
st.sidebar.header("Ask AI")
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

    # Read file skipping first 9 useless rows
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded, skiprows=9)
    else:
        df = pd.read_excel(uploaded, skiprows=9)

    # Clean columns
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "")
        .str.replace("-", "_")
    )

    # Rename important columns
    rename_map = {
        "s_no": "s_no",
        "sno": "s_no",
        "sno_": "s_no",
        "transaction_id": "transaction_id",
        "transactionid": "transaction_id",
        "amount": "amount",
    }
    df.rename(columns={col: rename_map.get(col, col) for col in df.columns}, inplace=True)

    # Convert everything to string (fix Supabase datetime issue)
    df = df.astype(str)

    # Check Transaction ID
    if "transaction_id" not in df.columns:
        st.error("❌ Transaction ID column not found!")
        st.stop()

    st.success("✔ Transaction ID detected")

    # ---------------- SHOW DATA -------------------
    st.subheader("📊 Cleaned Data Preview (Top 20 Rows)")
    st.dataframe(df.head(20))

    st.write("---")

    # ---------------- FIND ALL TXN IDs -------------
    txn_list = [
        x for x in df["transaction_id"].unique().tolist()
        if re.match(r"^TXN\d+$", x.strip())  # EXACT format: TXN + digits
    ]

    st.subheader("🔎 Search by Transaction ID")
    txn = st.selectbox("Select Transaction ID:", txn_list)

    if txn:
        details = df[df["transaction_id"] == txn]
        st.write("### Transaction Details")
        st.dataframe(details)

    st.write("---")

    # ---------------------- CHARTS -----------------------
    st.subheader("📈 Graphs & Charts")

    # Numeric columns
    num_cols = []
    for c in df.columns:
        try:
            df[c].astype(float)
            num_cols.append(c)
        except:
            pass

    if len(num_cols) > 0:
        pie_col = st.selectbox("Pie Chart Column:", num_cols)
        pie_data = df[pie_col].value_counts().rename_axis("Value").reset_index(name="Count")

        fig = px.pie(pie_data, values="Count", names="Value", title=f"{pie_col} Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns found for charts.")

    st.write("---")

    # ---------------------- SUPABASE UPLOAD -----------------------
    if st.button("Upload to Supabase"):
        try:
            records = df.replace({np.nan: None}).to_dict(orient="records")
            supabase.table("transaction_data").insert(records).execute()
            st.success("🎉 Uploaded Successfully!")
        except Exception as e:
            st.error(str(e))

else:
    st.info("Upload a file to begin.")
