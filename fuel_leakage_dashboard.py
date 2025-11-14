import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
from google import genai
import json
import typing

# -------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------
st.set_page_config(page_title="Fuel Leakage Dashboard", layout="wide")
st.title("⛽ Fuel Leakage Detection, Efficiency & Profit/Loss Dashboard")

# -------------------------------------------------
# LOAD SECRETS
# -------------------------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

if not GEMINI_API_KEY:
    st.error("Missing GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase credentials missing in secrets.")
    st.stop()

# -------------------------------------------------
# GEMINI NEW CLIENT
# -------------------------------------------------
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    GEMINI_MODEL = "gemini-2.5-flash"
except Exception as e:
    st.error(f"Gemini initialization error: {e}")
    st.stop()

# -------------------------------------------------
# SUPABASE INIT
# -------------------------------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------------------------
# FILE UPLOADER (SIDEBAR)
# -------------------------------------------------
st.sidebar.header("📂 Upload Trip Data File")
uploaded = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])

# -------------------------------------------------
# AI CHATBOT SECTION
# -------------------------------------------------
st.sidebar.write("---")
st.sidebar.subheader("🤖 Fuel AI Assistant")

user_question_sidebar = st.sidebar.text_input("Ask AI something:")

if st.sidebar.button("Get AI Answer"):
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_question_sidebar
        )
        st.sidebar.success(response.text)
    except Exception as e:
        st.sidebar.error(f"AI Error: {e}")

# -------------------------------------------------
# COLUMN NORMALIZATION
# -------------------------------------------------
def normalize_columns(df):
    col_map = {c.lower().strip(): c for c in df.columns}

    def find(possible):
        for p in possible:
            if p.lower() in col_map:
                return col_map[p.lower()]
        return None

    mapping = {
        "trip_id": find(["trip_id", "id", "trip", "sr no", "srno"]),
        "truck_id": find(["truck_id", "truck no", "vehicle", "truck", "truck number"]),
        "distance_km": find(["distance_km", "distance", "km"]),
        "actual_fuel_liters": find(["actual_fuel_liters", "fuel", "fuel_ltr", "fuel liters"]),
        "diesel_price_per_liter": find(["diesel_price_per_liter", "diesel price", "fuel_price", "price"]),
        "leakage_flag": find(["leakage_flag", "leakage", "status"])
    }

    for target_col, source_col in mapping.items():
        if source_col:
            df = df.rename(columns={source_col: target_col})
        else:
            # Create empty columns if missing
            if target_col == "trip_id":
                df[target_col] = df.index.astype(str)
            elif target_col == "truck_id":
                df[target_col] = "Unknown"
            elif target_col in ["distance_km", "actual_fuel_liters", "diesel_price_per_liter"]:
                df[target_col] = 0
            else:
                df[target_col] = "Normal"

    return df


# -------------------------------------------------
# MAIN — PROCESSING AFTER FILE UPLOAD
# -------------------------------------------------
if uploaded:

    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"File Read Error: {e}")
        st.stop()

    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head(15))

    df = normalize_columns(df)

    # Numeric cleanup
    for col in ["distance_km", "actual_fuel_liters", "diesel_price_per_liter"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df.loc[df["diesel_price_per_liter"] == 0, "diesel_price_per_liter"] = 95

    # Business logic
    df["revenue_per_km"] = np.where(
        df["leakage_flag"].astype(str).str.contains("Leak", case=False),
        90,
        150
    )

    df["expected_revenue"] = df["distance_km"] * df["revenue_per_km"]
    df["fuel_cost"] = df["actual_fuel_liters"] * df["diesel_price_per_liter"]
    df["profit_loss"] = df["expected_revenue"] - df["fuel_cost"]
    df["pnl_status"] = np.where(df["profit_loss"] > 0, "Profit", "Loss")

    st.success("Cleaned & Calculated Data:")
    st.dataframe(df.head(15))

    # -------------------------------------------------
    # Upload to Supabase
    # -------------------------------------------------
    if st.button("Upload Clean Data to Supabase"):
        try:
            data_json = df.where(pd.notnull(df), None).to_dict(orient="records")
            res = supabase.table("trip_data").insert(data_json).execute()
            st.success("Uploaded Successfully!")
        except Exception as e:
            st.error(f"Upload Error: {e}")

    # -------------------------------------------------
    # Summary Cards
    # -------------------------------------------------
    st.write("---")
    t1, t2, t3 = st.columns(3)
    t1.metric("Total Trips", len(df))
    t2.metric("Total Profit (₹)", round(df[df.pnl_status == "Profit"].profit_loss.sum(), 2))
    t3.metric("Total Loss (₹)", round(df[df.pnl_status == "Loss"].profit_loss.sum(), 2))

    # -------------------------------------------------
    # Graph
    # -------------------------------------------------
    st.subheader("📈 Profit / Loss Graph")
    fig = px.bar(df, x="trip_id", y="profit_loss", color="pnl_status")
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # Truck Search
    # -------------------------------------------------
    st.write("---")
    st.subheader("🚚 Search Truck Details")

    truck_search = st.text_input("Enter Truck Number:")
    if truck_search:
        result = df[df["truck_id"].astype(str).str.contains(truck_search, case=False)]
        if len(result) == 0:
            st.warning("No matching truck found.")
        else:
            st.dataframe(result)

    # -------------------------------------------------
    # Chatbot (Main)
    # -------------------------------------------------
    st.write("---")
    st.header("🤖 Ask Anything About Your Data")

    q = st.text_input("Type your question:")

    if st.button("Get Answer"):
        context = df.head(10).to_dict(orient="records")
        prompt = f"""
        You are analyzing fuel trip data.
        Data sample: {json.dumps(context, indent=2)}
        Question: {q}
        Answer clearly:
        """

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        st.success(response.text)

else:
    st.info("Upload a CSV or Excel file to begin.")
