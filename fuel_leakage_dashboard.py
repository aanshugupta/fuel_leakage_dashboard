import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativelanguage as genai

# ---------------------
# 1. Load Secrets
# ---------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

st.set_page_config(page_title="Fuel / Sales Dashboard", layout="wide")

# ========================================================
#  HEADER
# ========================================================
st.title("⛽ Fuel | Sales Transaction Dashboard")

st.sidebar.header("🚀 Select Data Source")

# ========================================================
# 2. Supabase table selection
# ========================================================
table_list = ["sales_data", "trip_data", "driver_summary"]
selected_table = st.sidebar.selectbox("Choose Supabase Table", table_list)

load_btn = st.sidebar.button("Load from Supabase")

df = None

# ========================================================
# 3. File Upload Option
# ========================================================
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

# ========================================================
#  LOAD DATA SECTION
# ========================================================
def load_supabase_table(table_name):
    data = supabase.table(table_name).select("*").execute()
    df = pd.DataFrame(data.data)
    return df


if load_btn:
    try:
        df = load_supabase_table(selected_table)
        st.success(f"Loaded data from Supabase: {selected_table}")
    except Exception as e:
        st.error(f"Supabase load error: {e}")

elif uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, skiprows=9)    # 10th row se data
        else:
            df = pd.read_excel(uploaded_file, skiprows=9)
        st.success(f"Loaded data from file: {uploaded_file.name}")
    except Exception as e:
        st.error(f"File load error: {e}")

# ========================================================
#  SHOW DATA PREVIEW
# ========================================================
if df is not None:
    st.subheader("📄 Data Preview")
    st.dataframe(df, use_container_width=True)

    # ========================================================
    #  🔍 SEARCH BY TRANSACTION ID
    # ========================================================
    st.subheader("🔍 Search Transaction")
    txn = st.text_input("Enter Transaction ID")

    if txn:
        result = df[df["Transaction ID"] == txn]
        if result.empty:
            st.warning("No matching transaction found.")
        else:
            st.success("Transaction Found:")
            st.dataframe(result)

    # ========================================================
    #  📊 CHARTS SECTION — FIXED
    # ========================================================
    st.subheader("📊 Charts")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found. Cannot generate charts.")
    else:
        chart_col = st.selectbox("Select numeric column for charts", numeric_cols)

        chart_type = st.radio("Choose Chart Type", ["Line", "Bar", "Area", "Histogram"])

        if chart_type == "Line":
            fig = px.line(df, y=chart_col, title=f"{chart_col} Line Trend")
        elif chart_type == "Bar":
            fig = px.bar(df, y=chart_col, title=f"{chart_col} Bar Chart")
        elif chart_type == "Area":
            fig = px.area(df, y=chart_col, title=f"{chart_col} Area Chart")
        else:
            fig = px.histogram(df, x=chart_col, title=f"{chart_col} Distribution")

        st.plotly_chart(fig, use_container_width=True)

# ========================================================
#  🤖 CHATBOT (Gemini)
# ========================================================
st.sidebar.subheader("🤖 ChatbotGemini")
question = st.sidebar.text_input("Ask anything about the data")

if question and df is not None:
    prompt = f"""
    You are an AI helping to analyze fuel/sales data.
    Here is the dataset: {df.head(20).to_string()}
    User question: {question}
    Answer in simple human-friendly language.
    """

    reply = model.generate_content(prompt)
    st.sidebar.write(reply.text)

elif question and df is None:
    st.sidebar.warning("Please load data first.")
