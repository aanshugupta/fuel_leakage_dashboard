import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client
from google import genai

# --------------------------- SETTINGS ---------------------------
st.set_page_config(page_title="Fuel PRO Dashboard", layout="wide")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
ai = genai.Client(api_key=GEMINI_API_KEY)


# --------------------------- PRO UI ---------------------------
st.markdown("""
    <style>
        .big-title { font-size:38px; font-weight:700; margin-bottom:15px; }
        .section-title { font-size:26px; font-weight:600; margin-top:30px; }
        .metric-card {
            padding:18px; border-radius:12px; background:#f7f7f7; 
            border:1px solid #e0e0e0; text-align:center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>⛽ Fuel / Sales PRO Intelligence Dashboard</div>", unsafe_allow_html=True)


# --------------------------- LOAD DATA ---------------------------
st.sidebar.header("📂 Load Your Data")

table_list = ["sales_data", "trip_data", "driver_summary"]
selected_table = st.sidebar.selectbox("Select Supabase Table", table_list)

load_supabase = st.sidebar.button("Load From Supabase")
upload_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

df = None

def load_from_supabase(table):
    result = supabase.table(table).select("*").execute()
    return pd.DataFrame(result.data)

# Load logic
if load_supabase:
    df = load_from_supabase(selected_table)
    st.success(f"Loaded data from Supabase table: {selected_table}")

elif upload_file:
    if upload_file.name.endswith(".csv"):
        df = pd.read_csv(upload_file, skiprows=0)
    else:
        df = pd.read_excel(upload_file, skiprows=0)
    st.success(f"File Loaded: {upload_file.name}")


# =================================================================
#                    SHOW DASHBOARD ONLY IF DATA EXISTS
# =================================================================
if df is not None:

    df = df.apply(pd.to_numeric, errors="ignore")

    st.subheader("📘 Data Preview")
    st.dataframe(df, use_container_width=True)


    # --------------------------- LOOKUP ---------------------------
    st.markdown("<div class='section-title'>🚚 Transaction / Truck Lookup</div>", unsafe_allow_html=True)

    txn_col = None
    for col in df.columns:
        if "transaction" in col.lower() and "id" in col.lower():
            txn_col = col
            break

    if txn_col:
        ids = df[txn_col].dropna().unique().tolist()
        selected_id = st.selectbox("Select Transaction ID", ["Select..."] + ids)

        if selected_id != "Select...":
            record = df[df[txn_col] == selected_id]
            st.dataframe(record)

            question = f"Analyse this record:\n{record.to_string()}"
            res = ai.models.generate_content(model="gemini-1.5-flash", contents=question)
            st.info(res.text)
    else:
        st.warning("No Transaction ID column found.")


    # --------------------------- QUICK STATS ---------------------------
    st.markdown("<div class='section-title'>📊 Quick Stats</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", len(df))

    if "Amount" in df.columns:
        c2.metric("Total Amount", round(df["Amount"].sum(), 2))
    if "Liters" in df.columns:
        c3.metric("Total Liters", round(df["Liters"].sum(), 2))
    if "Fuel Station ID" in df.columns:
        c4.metric("Fuel Stations", df["Fuel Station ID"].nunique())


    # --------------------------- LEAKAGE CHECK ---------------------------
    st.markdown("<div class='section-title'>🚨 Fuel Leakage Detection</div>", unsafe_allow_html=True)

    leak_cols = ["Billed Liters", "Delivered Liters"]

    if all(col in df.columns for col in leak_cols):

        df["Leak"] = df["Billed Liters"] - df["Delivered Liters"]
        leaks = df[df["Leak"] > 2]

        if len(leaks) > 0:
            st.error("⚠ Possible Leakage Detected")
            st.dataframe(leaks)
        else:
            st.success("No leakage found.")
    else:
        st.warning("Leakage check skipped — columns missing.")


    # --------------------------- MONTHLY SUMMARY ---------------------------
    st.markdown("<div class='section-title'>🗓 Monthly Summary</div>", unsafe_allow_html=True)

    date_col = None
    for col in df.columns:
        if "date" in col.lower():
            date_col = col
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["Month"] = df[date_col].dt.to_period("M").astype(str)

        monthly = df.groupby("Month").sum(numeric_only=True)

        if not monthly.empty:
            st.line_chart(monthly.iloc[:, 0])
        else:
            st.info("No numeric data to plot.")

    else:
        st.warning("No date column found.")


    # --------------------------- PRO CHART BUILDER ---------------------------
    st.markdown("<div class='section-title'>📈 Advanced Chart Builder</div>", unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if numeric_cols:
        col_sel = st.selectbox("Select Numeric Column", numeric_cols)
        chart_type = st.selectbox("Select Chart Type", ["Line", "Bar", "Area", "Histogram"])

        safe_df = df[[col_sel]].dropna()

        if chart_type == "Line":
            fig = px.line(safe_df, y=col_sel)
        elif chart_type == "Bar":
            fig = px.bar(safe_df, y=col_sel)
        elif chart_type == "Area":
            fig = px.area(safe_df, y=col_sel)
        else:
            fig = px.histogram(safe_df, x=col_sel)

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No numeric columns found.")


# --------------------------- GEMINI BOT ---------------------------
st.sidebar.subheader("🤖 ChatbotGemini")
ask = st.sidebar.text_input("Ask about your data")

if ask and df is not None:
    prompt = f"Dataset: {df.head(10).to_string()}\nUser question: {ask}"
    ans = ai.models.generate_content(model="gemini-1.5-flash", contents=prompt)
    st.sidebar.info(ans.text)
