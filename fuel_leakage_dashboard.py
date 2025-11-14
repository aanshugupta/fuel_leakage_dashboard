import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client

# -------------------------------
# Streamlit Secrets
# -------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

# -------------------------------
# Gemini NEW SDK Setup
# -------------------------------
try:
    from google import genai
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
    GEMINI_OK = True
except:
    GEMINI_OK = False

def ask_gemini(prompt):
    if not GEMINI_OK:
        return "Gemini SDK not found. Add to Streamlit packages: google-genai"
    try:
        res = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return res.text
    except Exception as e:
        return f"Gemini Error: {e}"


# -------------------------------
# Supabase Setup
# -------------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------
# Layout
# -------------------------------
st.set_page_config(page_title="Fuel + Transaction Dashboard", layout="wide")
st.title("⛽ Fuel Leakage + Transaction Analysis Dashboard")


# -------------------------------
# Sidebar Upload & Chatbot
# -------------------------------
st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.write("---")
st.sidebar.header("🤖 Ask AI (Sidebar)")
side_question = st.sidebar.text_input("Your question")

if st.sidebar.button("Ask AI"):
    st.sidebar.success(ask_gemini(side_question))


# -------------------------------
# After file upload
# -------------------------------
if uploaded:

    # -------------------------------
    # READ FILE — SKIP FIRST 9 GARBAGE LINES
    # -------------------------------
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded, skiprows=9)
        else:
            df = pd.read_excel(uploaded, skiprows=9)
    except Exception as e:
        st.error(f"File Read Error: {e}")
        st.stop()

    # -------------------------------
    # CLEAN COLUMN NAMES
    # -------------------------------
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Column Mapping
    rename_map = {
        "s_no": "s_no",
        "sno": "s_no",
        "s.no": "s_no",

        "transaction_id": "transaction_id",
        "txn_id": "transaction_id",
        "txnid": "transaction_id",

        "transaction_date": "transaction_date",
        "transaction_time": "transaction_time",
        "transaction_type": "transaction_type",
        "name": "name",
    }

    df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)

    # -------------------------------
    # Ensure S.No starts from 1
    # -------------------------------
    if "s_no" not in df.columns:
        df.insert(0, "s_no", range(1, len(df)+1))


    # -------------------------------
    # Show Cleaned Data
    # -------------------------------
    st.subheader("📊 Cleaned Data Preview (Top 20 Rows)")
    st.dataframe(df.head(20))


    # -------------------------------
    # Transaction ID Check
    # -------------------------------
    if "transaction_id" not in df.columns:
        st.error("❌ No Transaction ID found in your file! Please check column names.")
    else:
        st.success("Transaction ID detected successfully!")

    st.write("---")

    # -------------------------------
    # Upload to Supabase
    # -------------------------------
    if st.button("Upload to Supabase"):
        try:
            records = df.replace({np.nan: None}).to_dict(orient="records")
            supabase.table("trip_data").insert(records).execute()
            st.success("🚀 Uploaded Successfully!")
        except Exception as e:
            st.error(str(e))


    # -------------------------------
    # SEARCH BY TRANSACTION ID
    # -------------------------------
    st.subheader("🔍 Search by Transaction ID")

    if "transaction_id" in df.columns:
        txn_list = df["transaction_id"].dropna().astype(str).unique().tolist()

        selected_txn = st.selectbox("Select Transaction ID", txn_list)

        if selected_txn:
            result = df[df["transaction_id"] == selected_txn]
            st.write("### Transaction Detail")
            st.dataframe(result)


    st.write("---")

    # -------------------------------
    # PIE & BAR Charts
    # -------------------------------
    st.subheader("📈 Charts")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # PIE Chart: Use transaction_type if exists
    if "transaction_type" in df.columns:
        pie_data = df["transaction_type"].value_counts().reset_index()
        pie_data.columns = ["Type", "Count"]

        fig_pie = px.pie(pie_data, names="Type", values="Count",
                         title="Transaction Type Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Bar chart for S.No (example)
    fig_bar = px.bar(df, x="s_no", title="Record Count by S.No")
    st.plotly_chart(fig_bar, use_container_width=True)


    st.write("---")

    # -------------------------------
    # AI Insights (Main Section)
    # -------------------------------
    st.header("🤖 AI Insights About Your Data")

    user_q = st.text_input("Ask anything about your uploaded data:")

    if st.button("Get AI Answer"):
        preview = df.head(10).to_dict(orient="records")

        prompt = f"""
        You are a data-expert AI.

        Here is sample data:
        {preview}

        User question:
        {user_q}

        Give a clear answer.
        """

        st.success(ask_gemini(prompt))


else:
    st.info("Please upload a CSV or Excel file to begin.")
