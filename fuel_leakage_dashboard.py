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
# Gemini Setup
# -------------------------------
try:
    from google import genai
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
    GEMINI_OK = True
except:
    GEMINI_OK = False


def ask_gemini(prompt):
    if not GEMINI_OK:
        return "Gemini SDK missing. Add google-genai package."
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
st.title("⛽ Fuel Leakage + Transaction Dashboard")

# -------------------------------
# Sidebar upload
# -------------------------------
st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.write("---")
st.sidebar.header("🤖 Ask AI (Sidebar)")
q = st.sidebar.text_input("Your Question")

if st.sidebar.button("Ask AI"):
    st.sidebar.success(ask_gemini(q))


# -------------------------------
# When file is uploaded
# -------------------------------
if uploaded:

    # Read file, skip 9 rows
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded, skiprows=9)
        else:
            df = pd.read_excel(uploaded, skiprows=9)
    except Exception as e:
        st.error(f"File error: {e}")
        st.stop()

    # Clean column names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Transaction ID fix — include ALL possibilities
    rename_map = {
        "transaction_id": "transaction_id",
        "transactionid": "transaction_id",
        "txn_id": "transaction_id",
        "txn_id_": "transaction_id",
        "trans_id": "transaction_id",
        "t_id": "transaction_id",
        "txn": "transaction_id",
        "transaction_id_": "transaction_id",
        "transaction__id": "transaction_id",
        "transaction_id__": "transaction_id",
        "transactionid_": "transaction_id",
        "transaction_id__1": "transaction_id",

        # Your actual column name:
        "transaction_id": "transaction_id",
        "transaction_id.": "transaction_id",
        "transaction id": "transaction_id",
        "transaction_id ": "transaction_id",
        "transactionid ": "transaction_id",
    }

    df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)

    # Make sure S.No exists
    if "s_no" not in df.columns:
        df.insert(0, "s_no", range(1, len(df)+1))

    # DATETIME FIX
    for col in df.columns:
        if df[col].dtype == "datetime64[ns]":
            df[col] = df[col].astype(str)

    # Show data
    st.subheader("📊 Cleaned Data Preview (Top 20 Rows)")
    st.dataframe(df.head(20))

    # Transaction ID check
    if "transaction_id" not in df.columns:
        st.error("❌ No Transaction ID found! Check column names.")
    else:
        st.success("Transaction ID detected ✔")

    st.write("---")

    # Supabase upload
    if st.button("Upload to Supabase"):
        try:
            records = df.replace({np.nan: None}).to_dict(orient="records")
            supabase.table("trip_data").insert(records).execute()
            st.success("🚀 Successfully uploaded!")
        except Exception as e:
            st.error(f"Upload Error: {e}")

    st.write("---")

    # Search by Transaction ID
    st.subheader("🔍 Search by Transaction ID")

    if "transaction_id" in df.columns:
        txn_list = df["transaction_id"].astype(str).dropna().unique().tolist()

        selected = st.selectbox("Select Transaction ID", txn_list)

        if selected:
            result = df[df["transaction_id"] == selected]
            st.write("### Transaction Details")
            st.dataframe(result)

    st.write("---")

    # Charts
    st.subheader("📈 Charts")

    num_cols = df.select_dtypes(include=[int, float]).columns.tolist()

    if num_cols:
        pie_col = st.selectbox("Select column for Pie Chart", num_cols)
        pie_data = df[pie_col].value_counts().reset_index()
        pie_data.columns = ["value", "count"]
        fig = px.pie(pie_data, names="value", values="count", title=f"{pie_col} Distribution")
        st.plotly_chart(fig)
    else:
        st.warning("No numeric columns found.")

    st.write("---")

    # AI Insights
    st.header("🤖 AI Insights")
    user_q = st.text_input("Ask about your data")

    if st.button("Get AI Answer"):
        preview = df.head(10).to_dict(orient="records")
        prompt = f"Data: {preview}\n\nQuestion: {user_q}"
        st.success(ask_gemini(prompt))

else:
    st.info("Upload a file to start.")
