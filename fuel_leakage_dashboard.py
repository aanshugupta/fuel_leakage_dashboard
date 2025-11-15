import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
from google import genai
from io import BytesIO

# ---------------------------------------------------------
#  LOAD SECRETS
# ---------------------------------------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------
#  STREAMLIT PAGE SETTINGS
# ---------------------------------------------------------
st.set_page_config(page_title="Premium Fuel Dashboard", layout="wide")
st.title("⛽ **Premium Fuel / Sales Intelligence Dashboard**")


# ---------------------------------------------------------
#  SIDEBAR — LOAD DATA
# ---------------------------------------------------------
st.sidebar.header("📂 Data Source")

table_list = ["sales_data", "trip_data", "driver_summary"]
selected_table = st.sidebar.selectbox("Select Supabase Table", table_list)
load_supabase_btn = st.sidebar.button("Load from Supabase")

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

df = None


# ---------------------------------------------------------
#  SUPABASE LOAD FUNCTION
# ---------------------------------------------------------
def load_data_from_supabase(table):
    result = supabase.table(table).select("*").execute()
    return pd.DataFrame(result.data)


# ---------------------------------------------------------
#  LOAD LOGIC
# ---------------------------------------------------------
if load_supabase_btn:
    df = load_data_from_supabase(selected_table)
    st.success(f"Loaded data from Supabase → {selected_table}")

elif uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, skiprows=9)
        else:
            df = pd.read_excel(uploaded_file, skiprows=9)
        st.success(f"File Loaded: {uploaded_file.name}")
    except Exception as e:
        st.error("Error loading file: " + str(e))


# ---------------------------------------------------------
#  IF DATA EXISTS — SHOW PREMIUM DASHBOARD
# ---------------------------------------------------------
if df is not None:

    # Auto convert types
    df = df.apply(pd.to_numeric, errors="ignore")

    st.subheader("📄 Data Preview")
    st.dataframe(df, use_container_width=True)

    # ---------------------------------------------------------
    #  TRANSACTION LOOKUP
    # ---------------------------------------------------------
    st.subheader("🚚 Transaction / Truck Lookup")

    if "Transaction ID" in df.columns:

        txn_ids = df["Transaction ID"].dropna().unique().tolist()
        selected_txn = st.selectbox("Choose Transaction ID", ["Select..."] + txn_ids)

        if selected_txn != "Select...":
            txn_data = df[df["Transaction ID"] == selected_txn]
            st.success(f"Found {len(txn_data)} matching records")
            st.dataframe(txn_data)

            # ---------------- AI SUMMARY FOR TRUCK ----------------
            ask = f"""
            You are an intelligent fleet analyst.
            Here is the truck fuel record:
            {txn_data.to_string()}
            Provide a smart summary including:
            - Fuel usage
            - Any leakage suspicion
            - Efficiency score
            - Recommendations
            """
            res = ai.models.generate_content(model="gemini-1.5-flash", contents=ask)
            st.info(res.text)


    else:
        st.warning("⚠ 'Transaction ID' column not found.")


    # ---------------------------------------------------------
    #  QUICK STATS
    # ---------------------------------------------------------
    st.subheader("📊 Quick Stats")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", len(df))

    if "Amount" in df.columns:
        col2.metric("Total Fuel Amount", round(df["Amount"].sum(), 2))

    if "Liters" in df.columns:
        col3.metric("Total Liters", round(df["Liters"].sum(), 2))

    if "Fuel Station ID" in df.columns:
        col4.metric("Unique Fuel Stations", df["Fuel Station ID"].nunique())


    # ---------------------------------------------------------
    #  PREMIUM FEATURE — FUEL LEAKAGE DETECTION
    # ---------------------------------------------------------
    st.subheader("🚨 Fuel Leakage Detection")

    leakage_results = []

    if "Billed Liters" in df.columns and "Delivered Liters" in df.columns:

        df["Leak"] = df["Billed Liters"] - df["Delivered Liters"]
        suspicious = df[df["Leak"] > 2]  # tolerance = 2 liters

        if len(suspicious) > 0:
            st.error("⚠ **Possible Fuel Leakage Detected**")
            st.dataframe(suspicious)
        else:
            st.success("No leakage detected based on available fields.")

    else:
        st.warning("Leakage check skipped — Required columns missing.")


    # ---------------------------------------------------------
    #  MONTHLY ANALYSIS
    # ---------------------------------------------------------
    if "Transaction Date" in df.columns:
        st.subheader("📅 Monthly Summary")

        df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
        df["Month"] = df["Transaction Date"].dt.to_period("M").astype(str)

        monthly = df.groupby("Month").sum(numeric_only=True)
        st.line_chart(monthly["Liters"] if "Liters" in monthly else monthly.iloc[:, 0])


    # ---------------------------------------------------------
    #  PREMIUM CHART ENGINE
    # ---------------------------------------------------------
    st.subheader("📈 Advanced Chart Builder")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if numeric_cols:
        col_to_plot = st.selectbox("Select Numeric Column", numeric_cols)
        chart_type = st.selectbox("Select Chart Type", ["Line", "Bar", "Area", "Histogram"])

        clean_df = df[[col_to_plot]].dropna()

        if chart_type == "Line":
            fig = px.line(clean_df, y=col_to_plot)
        elif chart_type == "Bar":
            fig = px.bar(clean_df, y=col_to_plot)
        elif chart_type == "Area":
            fig = px.area(clean_df, y=col_to_plot)
        else:
            fig = px.histogram(clean_df, x=col_to_plot)

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No numeric columns for charts.")


    # ---------------------------------------------------------
    #  DOWNLOAD REPORT
    # ---------------------------------------------------------
    st.subheader("⬇ Download Filtered Report")

    download_df = df.to_csv(index=False).encode()
    st.download_button("Download CSV", download_df, "report.csv", "text/csv")



# ---------------------------------------------------------
#  CHATBOT GEMINI
# ---------------------------------------------------------
st.sidebar.subheader("🤖 ChatbotGemini")

ask_user = st.sidebar.text_input("Ask about your data")

if ask_user and df is not None:

    prompt = f"""
    You are a premium data assistant.
    Dataset sample:
    {df.head(20).to_string()}
    User Question: {ask_user}
    Provide a professional, human-like insight.
    """

    ans = ai.models.generate_content(model="gemini-1.5-flash", contents=prompt)

    st.sidebar.info(ans.text)
