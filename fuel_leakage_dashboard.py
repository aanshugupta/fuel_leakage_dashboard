import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai
import json

# ============================================================
#                  STREAMLIT SECRETS
# ============================================================
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# ---------------- Gemini Setup ----------------
genai.configure(api_key=GEMINI_API_KEY)
gmodel = genai.GenerativeModel("gemini-2.0-flash")

# ---------------- Supabase Setup ----------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================
#                  STREAMLIT UI SETUP
# ============================================================
st.set_page_config(page_title="Transaction + Fuel Dashboard", layout="wide")
st.title("📊 Advanced Transaction Dashboard with AI + Supabase")

# ============================================================
#                  SIDEBAR
# ============================================================
st.sidebar.header("📂 Upload File")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

st.sidebar.write("----")
st.sidebar.header("🤖 Ask AI About Your Data")
ai_q = st.sidebar.text_input("Ask AI")

if st.sidebar.button("Ask"):
    try:
        ai_ans = gmodel.generate_content(ai_q)
        st.sidebar.success(ai_ans.text)
    except:
        st.sidebar.error("AI error occurred")

st.sidebar.write("----")
st.sidebar.header("📁 View Supabase Saved Data")

# READ SAVED DATASETS FROM SUPABASE
try:
    supa_rows = supabase.table("trip_data").select("*").execute()
    supa_df = pd.DataFrame(supa_rows.data)

    if not supa_df.empty:
        st.sidebar.success(f"{len(supa_df)} records found")

        # Show dropdown for Supabase stored files
        # we assume trip_id or transaction_id exist
        id_columns = [c for c in supa_df.columns if "id" in c.lower()]

        if id_columns:
            main_id = id_columns[0]
            supa_ids = supa_df[main_id].astype(str).unique().tolist()

            selected_supa_id = st.sidebar.selectbox("Select saved ID from Supabase", supa_ids)
            st.sidebar.write("Selected record:")
            st.sidebar.dataframe(supa_df[supa_df[main_id] == selected_supa_id])

    else:
        st.sidebar.info("No data available in Supabase")
except:
    st.sidebar.error("Failed to load Supabase data")

# ============================================================
#                  FUNCTION → CLEAN FILE
# ============================================================
def clean_uploaded_file(file):
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file, skiprows=9)
        else:
            df = pd.read_excel(file, skiprows=9)
    except:
        df = pd.read_excel(file)

    # Normalize columns
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    rename_map = {
        "sno": "s_no",
        "s.no": "s_no",
        "txn_id": "transaction_id",
        "transaction": "transaction_id",
        "transactionid": "transaction_id",
        "fuel_liters": "actual_fuel_liters",
        "distance": "distance_km",
    }
    df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)

    return df


# ============================================================
#          MAIN DASHBOARD (FILE UPLOADED)
# ============================================================
if uploaded:

    df = clean_uploaded_file(uploaded)

    st.subheader("📌 Cleaned file preview:")
    st.dataframe(df.head(20))

    # save to supabase
    if st.button("Upload this file to Supabase"):
        try:
            records = df.replace({np.nan: None}).to_dict(orient="records")
            supabase.table("trip_data").insert(records).execute()
            st.success("File saved inside Supabase!")
        except Exception as e:
            st.error(str(e))

    st.write("----")

    # PICK TRANSACTION ID COLUMN
    id_cols = [c for c in df.columns if "id" in c]
    if id_cols:
        id_col = id_cols[0]

        st.subheader("🔍 Search by Transaction ID")
        txn = st.selectbox("Select Transaction ID", df[id_col].unique().astype(str))

        # show detail
        detail = df[df[id_col].astype(str) == txn]
        st.dataframe(detail)

    st.write("----")

    # ============================
    # GRAPHS
    # ============================
    st.subheader("📈 Charts")

    numeric_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
    if numeric_cols:
        col = st.selectbox("Choose column for Pie Chart", numeric_cols)
        pie_data = df[col].value_counts().reset_index()
        pie_data.columns = ["value", "count"]

        pie = px.pie(pie_data, values="count", names="value", title=f"{col} distribution")
        st.plotly_chart(pie, use_container_width=True)
    else:
        st.warning("No numeric columns!")

    # ============================
    # AI INSIGHTS
    # ============================
    st.write("----")
    st.header("🤖 AI Insights")

    q = st.text_input("Ask anything about your uploaded file")

    if st.button("AI Answer"):
        preview = df.head(10).to_dict(orient="records")

        prompt = f"""
        You are data expert.
        Here is sample data:
        {json.dumps(preview, indent=2)}

        User question:
        {q}
        """

        try:
            ai = gmodel.generate_content(prompt)
            st.success(ai.text)
        except:
            st.error("AI failed to process")

else:
    st.info("Upload any CSV or Excel file to begin.")
