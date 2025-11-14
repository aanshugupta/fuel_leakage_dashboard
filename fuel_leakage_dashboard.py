import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai

# ------------------------------------------------
# Load Secrets
# ------------------------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

if not GEMINI_API_KEY:
    st.error("Gemini API Key Missing!")
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase Credentials Missing!")

# ------------------------------------------------
# Gemini Setup
# ------------------------------------------------
genai.configure(api_key=GEMINI_API_KEY)
gmodel = genai.GenerativeModel("gemini-2.0-flash")  # stable model


# ------------------------------------------------
# Supabase Setup
# ------------------------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ------------------------------------------------
# Clean File Function (Auto Header Detection at Row 10)
# ------------------------------------------------
def clean_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            raw = pd.read_csv(uploaded_file, header=None)
        else:
            raw = pd.read_excel(uploaded_file, header=None)
    except Exception as e:
        st.error(f"❌ File read error: {e}")
        st.stop()

    # Auto detect header row (Row 10 -> index 9)
    header_row = None
    for i in range(len(raw)):
        row = raw.iloc[i].astype(str).str.lower().tolist()

        if (
            any(
                key in x
                for x in row
                for key in ["s.no", "transaction", "txn", "date", "amount"]
            )
        ):
            header_row = i
            break

    if header_row is None:
        header_row = 9  # default (your file format)

    # Load final cleaned df
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, header=header_row)
    else:
        df = pd.read_excel(uploaded_file, header=header_row)

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.astype(str).str.contains("unnamed")]

    # Clean column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Rename must-have columns
    rename_map = {
        "sno": "s_no",
        "s.no": "s_no",
        "sr_no": "s_no",
        "transactionid": "transaction_id",
        "transaction_id": "transaction_id",
        "txn_id": "transaction_id",
    }

    df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)

    return df


# ------------------------------------------------
# Streamlit Page Setup
# ------------------------------------------------
st.set_page_config(page_title="Universal Dashboard", layout="wide")
st.title("📄 Universal File → Cleaned Dashboard")


# ------------------------------------------------
# Sidebar: Upload + AI
# ------------------------------------------------
st.sidebar.header("Upload File")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.write("---")
st.sidebar.header("🤖 Ask AI")

user_q_sidebar = st.sidebar.text_input("Ask anything")

if st.sidebar.button("Ask AI"):
    try:
        ai_answer = gmodel.generate_content(user_q_sidebar)
        st.sidebar.success(ai_answer.text)
    except Exception as e:
        st.sidebar.error(str(e))


# ------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------
if uploaded:

    df = clean_file(uploaded)

    # Show cleaned table
    st.subheader("📊 Cleaned Data (Auto Detected Table)")
    st.dataframe(df.head(50), use_container_width=True)

    st.write("---")

    # ------------------------------------------------
    # Transaction ID Handling
    # ------------------------------------------------
    st.subheader("🔍 Search by Transaction ID")

    if "transaction_id" not in df.columns:
        st.error("❌ Transaction ID column not found! Please upload correct file.")
    else:
        txn_list = df["transaction_id"].dropna().astype(str).unique().tolist()

        selected_txn = st.selectbox("Select Transaction ID:", txn_list)

        if selected_txn:
            result = df[df["transaction_id"] == selected_txn]
            st.write("### Transaction Details")
            st.dataframe(result, use_container_width=True)

    st.write("---")

    # ------------------------------------------------
    # Charts Section
    # ------------------------------------------------
    st.subheader("📈 Charts")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for charts.")
    else:
        chart_col = st.selectbox("Select Column for Pie Chart", numeric_cols)

        pie_data = df[chart_col].value_counts().reset_index()
        pie_data.columns = ["value", "count"]

        fig = px.pie(
            pie_data, names="value", values="count",
            title=f"{chart_col} Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.write("---")

    # ------------------------------------------------
    # Upload to Supabase
    # ------------------------------------------------
    st.subheader("⬆ Upload Clean Data to Supabase")

    if st.button("Upload Now"):
        try:
            upload_df = df.replace({np.nan: None})
            supabase.table("trip_data").insert(upload_df.to_dict(orient="records")).execute()
            st.success("Uploaded Successfully!")
        except Exception as e:
            st.error(str(e))

    st.write("---")

    # ------------------------------------------------
    # AI on Main Page
    # ------------------------------------------------
    st.header("🤖 AI Assisted Analysis")

    user_q_main = st.text_input("Ask anything about your uploaded data:")

    if st.button("Get AI Insight"):
        sample = df.head(15).to_dict(orient="records")

        prompt = f"""
        You are a data expert. Here is sample data:
        {sample}

        Answer the user's question clearly:
        {user_q_main}
        """

        try:
            ai_out = gmodel.generate_content(prompt)
            st.success(ai_out.text)
        except Exception as e:
            st.error(str(e))

else:
    st.info("Please upload a file to start.")
