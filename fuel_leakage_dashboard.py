import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai
import re
import json

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
gmodel = genai.GenerativeModel("gemini-2.0-flash")

# -------------------------------
# Supabase Setup
# -------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------
# Auto Detect Header Row Function
# -------------------------------
def clean_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            raw = pd.read_csv(uploaded_file, header=None)
        else:
            raw = pd.read_excel(uploaded_file, header=None)
    except Exception as e:
        st.error(f"File reading error: {e}")
        st.stop()

    header_row = None
    for i in range(len(raw)):
        row = raw.iloc[i].astype(str).tolist()

        # Detect header using strong patterns
        if (
            any("s.no" in x.lower() for x in row) or
            any("transaction" in x.lower() for x in row) or
            any(re.match(r"txn", x.lower()) for x in row)
        ):
            header_row = i
            break

    if header_row is None:
        st.error("Could not detect column headers. Upload a proper file.")
        st.stop()

    # Reload clean dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, header=header_row)
    else:
        df = pd.read_excel(uploaded_file, header=header_row)

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.astype(str).str.contains("Unnamed")]

    # Clean column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Fix known names
    rename_map = {
        "sno": "s_no",
        "s.no": "s_no",
        "sr_no": "s_no",
        "transactionid": "transaction_id",
        "transaction_id": "transaction_id",
        "txn_id": "transaction_id",
    }

    df.rename(columns={col: rename_map.get(col, col) for col in df.columns}, inplace=True)

    return df


# -------------------------------
# Streamlit Layout
# -------------------------------
st.set_page_config(page_title="Universal Data Dashboard", layout="wide")

st.title("📄 Universal Data Dashboard (Fuel + Sales + Maintenance + Anything)")


# -------------------------------
# Sidebar Upload + AI Chatbot
# -------------------------------
st.sidebar.header("Upload File")
uploaded = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

st.sidebar.write("---")
st.sidebar.header("🤖 Ask AI (Sidebar)")

side_q = st.sidebar.text_input("Ask anything about the data")

if st.sidebar.button("Ask AI"):
    if uploaded:
        preview = st.session_state.get("preview_data", "No data loaded")
        prompt = f"Here is the data sample: {preview}\n\nUser question: {side_q}"
        try:
            ans = gmodel.generate_content(prompt)
            st.sidebar.success(ans.text)
        except Exception as e:
            st.sidebar.error(str(e))
    else:
        st.sidebar.warning("Upload a file first!")


# -------------------------------
# MAIN: If File Uploaded
# -------------------------------
if uploaded:

    df = clean_file(uploaded)

    st.subheader("📊 Cleaned Data Preview (Top 20 Rows)")
    st.dataframe(df.head(20))

    # Save preview for AI use
    st.session_state["preview_data"] = df.head(10).to_dict(orient="records")

    # -----------------------------------
    # Transaction ID Section
    # -----------------------------------
    st.write("---")
    st.subheader("🔎 Search by Transaction ID")

    if "transaction_id" not in df.columns:
        st.error("❌ Transaction ID column not detected.")
    else:
        txn_list = df["transaction_id"].dropna().astype(str).unique().tolist()

        selected_txn = st.selectbox("Select Transaction ID", txn_list)

        if selected_txn:
            result = df[df["transaction_id"] == selected_txn]
            st.write("### Transaction Details")
            st.dataframe(result)

    # -----------------------------------
    # Supabase Upload
    # -----------------------------------
    st.write("---")
    if st.button("Upload Clean Data to Supabase"):
        try:
            cleaned = df.copy()
            cleaned = cleaned.replace({np.nan: None})
            cleaned = cleaned.applymap(lambda x: x.isoformat() if hasattr(x, 'isoformat') else x)

            records = cleaned.to_dict(orient="records")
            supabase.table("trip_data").insert(records).execute()

            st.success("Uploaded Successfully!")
        except Exception as e:
            st.error(str(e))

    # -----------------------------------
    # Chart Section
    # -----------------------------------
    st.write("---")
    st.subheader("📈 Charts")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        sel_col = st.selectbox("Select column for Pie Chart", numeric_cols)
        temp = df[sel_col].value_counts().reset_index()
        temp.columns = ["Label", "Count"]

        fig = px.pie(temp, names="Label", values="Count", title=f"{sel_col} Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numeric columns found!")

    # -----------------------------------
    # AI Main Chat
    # -----------------------------------
    st.write("---")
    st.header("🤖 Ask AI About Your Data (Main Panel)")

    user_q = st.text_input("Ask AI:")

    if st.button("Get Answer"):
        sample = df.head(10).to_dict(orient="records")

        prompt = f"""
        You are an expert data analyst.
        Here is the uploaded data sample:
        {json.dumps(sample)}

        User question: {user_q}

        Give a clear and direct answer.
        """

        try:
            ai_ans = gmodel.generate_content(prompt)
            st.success(ai_ans.text)
        except Exception as e:
            st.error(str(e))

else:
    st.info("Please upload a file to begin.")
