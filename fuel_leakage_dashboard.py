import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-1.5-pro"
model = genai.GenerativeModel(MODEL_NAME)

# ---------------------------------------------------------
# LOAD SUPABASE TABLE
# ---------------------------------------------------------
def load_table(table_name):
    try:
        data = supabase.table(table_name).select("*").execute()
        df = pd.DataFrame(data.data)
        return df
    except:
        return pd.DataFrame()

# ---------------------------------------------------------
# AI ENGINES (Already in your Pack-3)
# ---------------------------------------------------------
def detect_leakage(df):
    required = ["Product Volume", "Rate (Rs./ Ltr)", "Purchase Amount"]

    if not all(col in df.columns for col in required):
        return None, "Required columns missing"

    df["Expected"] = df["Product Volume"] * df["Rate (Rs./ Ltr)"]
    df["Diff"] = df["Purchase Amount"] - df["Expected"]
    df["Leak %"] = (df["Diff"] / df["Expected"]) * 100

    ai_summary = model.generate_content(
        f"""
        Analyze this leakage dataset and describe leakage probability,
        suspicious transactions, and overcharging patterns.
        Data:
        {df.to_dict()}
        """
    ).text

    return df, ai_summary


def explain_transaction(row):
    prompt = f"""
    Explain this fuel transaction clearly:
    {row.to_dict()}
    """
    return model.generate_content(prompt).text


def detect_fraud(df):
    prompt = f"""
    Detect fraud, misuse, duplicate transactions, overcharging:
    {df.to_dict()}
    """
    return model.generate_content(prompt).text


def monthly_summary(df):
    prompt = f"""
    Create a monthly report from this dataset:
    {df.to_dict()}
    """
    return model.generate_content(prompt).text


def ask_ai_about_data(question, df):
    prompt = f"""
    You are a fuel analytics expert.
    Use ONLY this dataset to answer:
    {df.to_dict()}
    Question: {question}
    """
    return model.generate_content(prompt).text


# =========================================================
# UI STARTS — Existing features remain untouched
# =========================================================
st.title("⛽ Premium Fuel Intelligence Dashboard — Pack 3 + Transport Pack Level-2")

# SIDEBAR
table = st.sidebar.selectbox("Choose Supabase Table", ["sales_data", "drive_summary", "trip_data"])
df = load_table(table)

if df.empty:
    st.error("⚠ No data found.")
    st.stop()

st.success(f"Loaded {len(df)} records from: {table}")

# ---------------------------------------------------------
# TRANSACTION LOOKUP (existing)
# ---------------------------------------------------------
st.subheader("🔍 Transaction Lookup")
txn_field = next((c for c in df.columns if "transaction" in c.lower()), None)

if txn_field:
    txn_id = st.selectbox("Select Transaction ID", df[txn_field].astype(str).unique())
    if txn_id:
        row = df[df[txn_field].astype(str) == txn_id].iloc[0]
        st.write("### Transaction Details")
        st.dataframe(pd.DataFrame([row]))

        st.write("### 🧠 AI Explanation")
        st.info(explain_transaction(row))

# ---------------------------------------------------------
# QUICK STATS (existing)
# ---------------------------------------------------------
st.subheader("📊 Quick Stats")
st.metric("Total Rows", len(df))
st.metric("Numeric Columns", len(df.select_dtypes(include=['int64','float64']).columns))

# ---------------------------------------------------------
# LEAKAGE DETECTION (existing)
# ---------------------------------------------------------
st.subheader("🚨 Fuel Leakage Detection (AI)")
leak_df, leak_text = detect_leakage(df)
if leak_df is not None:
    st.dataframe(leak_df[["Product Volume", "Rate (Rs./ Ltr)", "Purchase Amount", "Leak %"]])
    st.error(leak_text)
else:
    st.warning(leak_text)

# ---------------------------------------------------------
# MONTHLY SUMMARY (existing)
# ---------------------------------------------------------
st.subheader("🗓 Monthly Summary (AI)")
st.info(monthly_summary(df))

# ---------------------------------------------------------
# FRAUD DETECTION (existing)
# ---------------------------------------------------------
st.subheader("🔍 Fraud Detection (AI)")
st.warning(detect_fraud(df))

# ---------------------------------------------------------
# CHART BUILDER (existing)
# ---------------------------------------------------------
st.subheader("📈 Chart Builder")
num_cols = df.select_dtypes(include=["int64","float64"]).columns
if len(num_cols) > 0:
    col = st.selectbox("Select numeric column", num_cols)
    chart = st.radio("Chart Type", ["Line", "Bar", "Area"])
    if chart == "Line":
        fig = px.line(df, y=col)
    elif chart == "Bar":
        fig = px.bar(df, y=col)
    else:
        fig = px.area(df, y=col)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# CHATBOT (existing)
# ---------------------------------------------------------
st.subheader("🤖 Fuel Assistant — Gemini Pro")
q = st.text_input("Ask anything about your dataset")
if q:
    st.success(ask_ai_about_data(q, df))

# =========================================================
# 🚚 TRANSPORT PACK LEVEL-2 (NEW — ADDED BELOW)
# =========================================================

st.markdown("---")
st.header("🚚 Transport Intelligence Pack — Level-2")

# ---------------------------------------------------------
# DRIVER PERFORMANCE SCORE
# ---------------------------------------------------------
st.subheader("🧑‍✈️ Driver Performance Score")

driver_fields = ["driver", "driver name", "driver_id"]
driver_col = next((c for c in df.columns if any(x in c.lower() for x in driver_fields)), None)

if driver_col and "Distance" in df.columns and "Fuel Used" in df.columns:
    df["Score"] = (df["Distance"] / df["Fuel Used"]).round(2)
    fig = px.bar(df, x=driver_col, y="Score", title="Driver Efficiency Score")
    st.plotly_chart(fig)
else:
    st.info("Driver / Distance / Fuel columns not found.")

# ---------------------------------------------------------
# VEHICLE EFFICIENCY RANKING
# ---------------------------------------------------------
st.subheader("🚛 Vehicle Efficiency Ranking")

if "Vehicle Number" in df.columns and "Distance" in df.columns and "Fuel Used" in df.columns:
    df["KMPL"] = (df["Distance"] / df["Fuel Used"]).round(2)
    ranked = df.groupby("Vehicle Number")["KMPL"].mean().sort_values(ascending=False)
    st.dataframe(ranked)
else:
    st.info("Vehicle / Distance / Fuel columns missing.")

# ---------------------------------------------------------
# ROUTE COST ANALYSIS
# ---------------------------------------------------------
st.subheader("🗺 Route Cost Analysis")

if "Route" in df.columns and "Purchase Amount" in df.columns:
    route_cost = df.groupby("Route")["Purchase Amount"].sum()
    fig = px.bar(route_cost, title="Route-wise Fuel Cost")
    st.plotly_chart(fig)
else:
    st.info("Route / Purchase Amount columns missing.")

# ---------------------------------------------------------
# MONTHLY HEATMAP
# ---------------------------------------------------------
st.subheader("🔥 Monthly Trip Frequency Heatmap")

if "Transaction Date" in df.columns:
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="ignore")
    df["Month"] = df["Transaction Date"].dt.month
    month_count = df["Month"].value_counts().sort_index()
    fig = px.imshow([month_count.values], labels=dict(x="Month", y="", color="Trips"))
    st.plotly_chart(fig)
else:
    st.info("Transaction Date missing.")

# ---------------------------------------------------------
# HIGH FUEL USAGE ALERTS
# ---------------------------------------------------------
st.subheader("⚠ High Fuel Usage Alerts")

if "Fuel Used" in df.columns:
    threshold = st.slider("Fuel Usage Alert Threshold", 10, 200, 50)
    alert_df = df[df["Fuel Used"] > threshold]
    st.dataframe(alert_df)
else:
    st.info("Fuel Used column missing.")

# ---------------------------------------------------------
# OVER-TIME DRIVING
# ---------------------------------------------------------
st.subheader("⏱ Overtime Driving Detection")

if "Start Time" in df.columns and "End Time" in df.columns:
    df["Start Time"] = pd.to_datetime(df["Start Time"], errors="ignore")
    df["End Time"] = pd.to_datetime(df["End Time"], errors="ignore")
    df["Hours"] = (df["End Time"] - df["Start Time"]).dt.total_seconds() / 3600
    overtime = df[df["Hours"] > 10]
    st.dataframe(overtime)
else:
    st.info("Start/End Time columns missing.")

# ---------------------------------------------------------
# IDLE TIME DETECTION
# ---------------------------------------------------------
st.subheader("🛑 Idle Time Detection")

if "Idle Time" in df.columns:
    idle_alert = df[df["Idle Time"] > 30]
    st.dataframe(idle_alert)
else:
    st.info("Idle Time column missing.")

# ---------------------------------------------------------
# SPEED vs MILEAGE
# ---------------------------------------------------------
st.subheader("🚦 Speed vs Mileage Insight")

if "Avg Speed" in df.columns and "KMPL" in df.columns:
    fig = px.scatter(df, x="Avg Speed", y="KMPL", trendline="ols",
                     title="Speed vs Mileage")
    st.plotly_chart(fig)
else:
    st.info("Avg Speed / KMPL column missing.")
