import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai

# --------------------------
# Load Secrets
# --------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

# Gemini setup
genai.configure(api_key=GEMINI_API_KEY)
gmodel = genai.GenerativeModel("gemini-2.0-flash")

# Supabase setup
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="Fuel + Transaction Dashboard", layout="wide")
st.title("⛽ Fuel Leakage + Transaction Analysis Dashboard")

# --------------------------
# Sidebar upload + AI
# --------------------------
st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.write("---")
st.sidebar.header("🤖 Ask AI")
q = st.sidebar.text_input("Your question")

if st.sidebar.button("Ask AI"):
    try:
        ans = gmodel.generate_content(q)
        st.sidebar.success(ans.text)
    except Exception as e:
        st.sidebar.error(str(e))

# --------------------------
# If file uploaded
# --------------------------
if uploaded:

    # read file, skip first 9 rows
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded, skiprows=9)
    else:
        df = pd.read_excel(uploaded, skiprows=9)

    st.subheader("📊 Cleaned Data Preview")
    st.dataframe(df.head(20))

    # normalize column names
    clean_cols = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "")
        .str.replace("-", "_")
    )
    df.columns = clean_cols

    # rename important columns
    rename_map = {
        "sno": "s_no",
        "sno_": "s_no",
        "s_no": "s_no",
        "sno": "s_no",
        "transaction_id": "transaction_id",
        "transactionid": "transaction_id",
        "transaction_id_": "transaction_id",
        "transaction_id__": "transaction_id",
        "registration_number": "vehicle_no",
        "amount": "amount",
    }

    df.rename(columns=rename_map, inplace=True)

    # ensure strings (fix datetime error)
    df = df.astype(str)

    # transaction id check
    if "transaction_id" not in df.columns:
        st.error("❌ Transaction ID column NOT found. Please check your Excel headers.")
        st.stop()
    else:
        st.success("✅ Transaction ID detected!")

    # --------------------------
    # Upload to Supabase
    # --------------------------
    if st.button("Upload to Supabase"):
        try:
            records = df.replace({np.nan: None}).to_dict(orient="records")
            supabase.table("trip_data").insert(records).execute()
            st.success("Uploaded Successfully!")
        except Exception as e:
            st.error(str(e))

    st.write("---")

    # --------------------------
    # Search by Transaction ID
    # --------------------------
    st.subheader("🔍 Search by Transaction ID")

    txn_list = df["transaction_id"].dropna().unique().tolist()

    selected = st.selectbox("Select Transaction ID", txn_list)

    if selected:
        out = df[df["transaction_id"] == selected]
        st.write("### Transaction Detail")
        st.dataframe(out)

    st.write("---")

    # --------------------------
    # Charts Section
    # --------------------------
    st.subheader("📈 Charts")

    # numeric-like columns
    num_cols = []
    for c in df.columns:
        try:
            df[c].astype(float)
            num_cols.append(c)
        except:
            pass

    if len(num_cols) > 0:
        pie_col = st.selectbox("Pie chart column", num_cols)
        pie_data = df[pie_col].value_counts().reset_index()
        pie_data.columns = ["Value", "Count"]

        fig = px.pie(pie_data, names="Value", values="Count", title=f"{pie_col} Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numeric columns available for charts.")

    st.write("---")

    # --------------------------
    # Main AI Insights
    # --------------------------
    st.header("🤖 AI Insights (Main)")

    ask_main = st.text_input("Ask anything about your uploaded data:")

    if st.button("Get AI Answer"):
        sample = df.head(10).to_dict(orient="records")
        prompt = f"""
        You are a data analyst.
        Here is sample data:
        {sample}

        Answer the question:
        {ask_main}
        """

        try:
            ans = gmodel.generate_content(prompt)
            st.success(ans.text)
        except Exception as e:
            st.error(str(e))

else:
    st.info("Please upload a file to continue.")
