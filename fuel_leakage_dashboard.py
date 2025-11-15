import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
from google import genai   # FIXED GEMINI IMPORT

# ---------------------
# Load Secrets
# ---------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configure Gemini
client = genai.Client(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="Fuel / Sales Dashboard", layout="wide")

# ========================================================
# HEADER
# ========================================================
st.title("⛽ Fuel | Sales Transaction Dashboard")

st.sidebar.header("🚀 Select Data Source")

# ========================================================
# Supabase Table Selection
# ========================================================
table_list = ["sales_data", "trip_data", "driver_summary"]
selected_table = st.sidebar.selectbox("Choose Supabase Table", table_list)

load_btn = st.sidebar.button("Load from Supabase")

df = None

# ========================================================
# File Upload
# ========================================================
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])


# ========================================================
# Supabase Data Loader
# ========================================================
def load_supabase_table(table_name):
    response = supabase.table(table_name).select("*").execute()
    df = pd.DataFrame(response.data)
    return df


# ========================================================
# Load Data Logic
# ========================================================
if load_btn:
    try:
        df = load_supabase_table(selected_table)
        st.success(f"Loaded data from Supabase: {selected_table}")
    except Exception as e:
        st.error(f"Supabase load error: {e}")

elif uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, skiprows=9)  # Actual data from 10th row
        else:
            df = pd.read_excel(uploaded_file, skiprows=9)

        st.success(f"Loaded file: {uploaded_file.name}")

    except Exception as e:
        st.error(f"File error: {e}")


# ========================================================
# DATA PREVIEW
# ========================================================
if df is not None:

    st.subheader("📄 Data Preview")
    st.dataframe(df, use_container_width=True)

    # ------------------- Search Transaction -------------------
    st.subheader("🔍 Search Transaction")
    txn = st.text_input("Enter Transaction ID")

    if txn:
        result = df[df["Transaction ID"] == txn]
        if result.empty:
            st.warning("Transaction not found.")
        else:
            st.success("Transaction Found:")
            st.dataframe(result)

    # ========================================================
    #  ADVANCED STATS
    # ========================================================
    st.subheader("📊 Quick Stats")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    if "Amount" in df.columns:
        col2.metric("Total Sales", df["Amount"].sum())
    if "Liters" in df.columns:
        col3.metric("Total Fuel (L)", df["Liters"].sum())

    # ========================================================
    #  CHARTS FIXED
    # ========================================================
    st.subheader("📈 Charts")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if numeric_cols:
        chart_col = st.selectbox("Select numeric column", numeric_cols)

        chart_type = st.radio("Choose Chart Type", ["Line", "Bar", "Area", "Histogram"])

        if chart_type == "Line":
            fig = px.line(df, y=chart_col)
        elif chart_type == "Bar":
            fig = px.bar(df, y=chart_col)
        elif chart_type == "Area":
            fig = px.area(df, y=chart_col)
        else:
            fig = px.histogram(df, x=chart_col)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numeric columns to plot.")

# ========================================================
#  CHATBOT (Gemini)
# ========================================================
st.sidebar.subheader("🤖 ChatbotGemini")
question = st.sidebar.text_input("Ask anything about the data")

if question and df is not None:
    prompt = f"""
    You are an AI data analyst.
    Dataset sample:
    {df.head(20).to_string()}
    User question: {question}
    Provide detailed human-friendly answer.
    """

    response = client.models.generate_content(
        model="gemini-1.5-flash", contents=prompt
    )

    st.sidebar.write(response.text)

elif question and df is None:
    st.sidebar.warning("Please load data first.")
