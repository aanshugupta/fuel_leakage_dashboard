import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai

# -------------------------------
# Streamlit Secrets Load
# -------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

if not GEMINI_API_KEY:
    st.error("Gemini API key missing in Streamlit secrets!")
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase credentials missing!")

# -------------------------------
# Gemini Setup
# -------------------------------
genai.configure(api_key=GEMINI_API_KEY)
gmodel = genai.GenerativeModel("gemini-2.0-flash")  # SAFE MODEL

# -------------------------------
# Supabase Setup
# -------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------
# Layout
# -------------------------------
st.set_page_config(page_title="Fuel + Transaction Dashboard", layout="wide")
st.title("⛽ Fuel Leakage + Transaction Analysis Dashboard")

# -------------------------------
# Sidebar: File Upload + AI Chatbot
# -------------------------------
st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.write("---")
st.sidebar.header("🤖 Ask AI")
side_question = st.sidebar.text_input("Your question")

if st.sidebar.button("Ask AI"):
    try:
        ai_answer = gmodel.generate_content(side_question)
        st.sidebar.success(ai_answer.text)
    except Exception as e:
        st.sidebar.error(str(e))

# -------------------------------
# If file uploaded
# -------------------------------
if uploaded:

    # -------------------------------
    # Read file + Skip first 9 rows
    # -------------------------------
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded, skiprows=9)
        else:
            df = pd.read_excel(uploaded, skiprows=9)
    except Exception as e:
        st.error(f"File read error: {e}")
        st.stop()

    st.subheader("📊 Cleaned Data Preview")
    st.dataframe(df.head(20))

    # -------------------------------
    # Normalize column names
    # -------------------------------
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Required mapping
    rename_map = {
        "sno": "s_no",
        "s.no": "s_no",
        "transaction_id": "transaction_id",
        "txn_id": "transaction_id",
        "amount": "amount",
        "fuel_liters": "actual_fuel_liters",
        "distance": "distance_km",
    }

    df.rename(columns={col: rename_map.get(col, col) for col in df.columns}, inplace=True)

    # -------------------------------
    # Check Transaction ID column
    # -------------------------------
    if "transaction_id" not in df.columns:
        st.error("❌ No Transaction ID found in your file! Please check column names.")
    else:
        st.success("Transaction ID column detected!")

    # -------------------------------
    # Supabase Upload Button
    # -------------------------------
    if st.button("Upload to Supabase"):
        try:
            records = df.replace({np.nan: None}).to_dict(orient="records")
            supabase.table("trip_data").insert(records).execute()
            st.success("Uploaded Successfully!")
        except Exception as e:
            st.error(str(e))

    st.write("---")

    # -------------------------------
    # Search by Transaction ID
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
    # Graphs Section
    # -------------------------------
    st.subheader("📈 Charts")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if len(num_cols) > 0:

        pie_col = st.selectbox("Pie chart column", num_cols)

        pie_data = df[pie_col].value_counts().reset_index()
        pie_data.columns = ["Value", "Count"]

        fig_pie = px.pie(pie_data, names="Value", values="Count", title=f"{pie_col} Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    else:
        st.warning("No numeric columns found for charts!")

    st.write("---")

    # -------------------------------
    # AI Based Querying on Clean Data
    # -------------------------------
    st.header("🤖 AI Insights (Main Area)")

    user_q = st.text_input("Ask anything about your uploaded data:")

    if st.button("Get AI Answer"):

        preview = df.head(10).to_dict(orient="records")

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
    st.info("Please upload a CSV or Excel file to begin.")
