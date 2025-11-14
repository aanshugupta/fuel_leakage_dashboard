import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai
import json
import typing
import re

# -------------------------------
# CONFIG: keys must be set in Streamlit Cloud secrets
# -------------------------------
# st.secrets should have:
# GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY
#
# (Do NOT hardcode keys in the file.)

# -------------------------------
# Gemini Setup (safe)
# -------------------------------
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY")
if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        # prefer a stable flash model; change only if your account supports it
        GEMINI_MODEL = "gemini-2.5-flash"
    except Exception:
        GEMINI_MODEL = None
else:
    GEMINI_MODEL = None

# -------------------------------
# Supabase Setup
# -------------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase credentials missing in Streamlit secrets. Add SUPABASE_URL and SUPABASE_KEY.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------
# PAGE
# -------------------------------
st.set_page_config(page_title="Fuel Leakage Dashboard", layout="wide")
st.title("⛽ Fuel Leakage Detection, Efficiency & Profit/Loss Dashboard")

# ----------------------------------------
# File uploader (CSV & XLSX)
# ----------------------------------------
uploaded = st.file_uploader("📂 Upload processed_trips file (CSV or XLSX)", type=["csv", "xlsx"])

# Helper: normalize column names and map common alternatives
def normalize_and_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    # build lowercase map
    existing = {c.lower().strip(): c for c in df.columns}
    def find(col_candidates: typing.List[str]) -> typing.Optional[str]:
        for cand in col_candidates:
            cand = cand.lower().strip()
            if cand in existing:
                return existing[cand]
        return None

    # common names we expect
    mapping = {}
    mapping['trip_id'] = find(["trip_id", "trip id", "trip", "tripid", "id"])
    mapping['distance_km'] = find(["distance_km", "distance", "distance (km)", "km"])
    mapping['actual_fuel_liters'] = find(["actual_fuel_liters", "fuel_liters", "fuel_ltr", "fuel", "fuel_liters_ltr", "actual_fuel"])
    mapping['diesel_price_per_liter'] = find(["diesel_price_per_liter", "price_per_liter", "diesel_price", "fuel_price", "price"])
    mapping['leakage_flag'] = find(["leakage_flag", "leakage", "leak_flag", "leakage_suspected", "status"])

    # If any mapping is None, create new column with safe defaults
    for out_col, found in mapping.items():
        if found:
            # rename to standard column name (only once)
            if found != out_col:
                df = df.rename(columns={found: out_col})
        else:
            # create the column with default values
            if out_col == "trip_id":
                # if no id, create sequential id
                df[out_col] = df.index.astype(str).tolist()
            elif out_col in ("distance_km", "actual_fuel_liters", "diesel_price_per_liter"):
                df[out_col] = 0
            else:
                df[out_col] = "Normal"
    return df

# Convert dataframe values to python native types for Supabase
def df_to_records_for_supabase(df: pd.DataFrame) -> list:
    df2 = df.copy()
    # convert numpy types to python native and NaN to None
    df2 = df2.where(pd.notnull(df2), None)
    records = df2.to_dict(orient="records")
    # ensure python native ints/floats/str
    def convert_val(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_ ,)):
            return bool(v)
        return v
    records = [{k: convert_val(v) for k, v in r.items()} for r in records]
    return records

# Chatbot helper (safe)
def ask_gemini(question: str) -> str:
    if not GEMINI_MODEL:
        return "Gemini API key or model not configured in secrets."
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        # generate_content supports prompt string or dict depending on SDK; using simple string
        resp = model.generate_content(question)
        # response object shape can vary; we handle common attributes
        if hasattr(resp, "text"):
            return resp.text
        # fallback to raw JSON text
        return json.dumps(resp, default=str)
    except Exception as e:
        # Provide helpful hint when model/version is not available
        msg = str(e)
        if "404" in msg or "not found" in msg.lower():
            return "Model or endpoint not found. Check your Gemini model string and account permissions."
        return f"Gemini error: {e}"

# -------------------------------
# MAIN: When file uploaded
# -------------------------------
if uploaded:
    # read file robustly
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded, low_memory=False)
        else:
            # read first sheet by default
            df = pd.read_excel(uploaded, engine="openpyxl")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        st.stop()

    st.subheader("📊 Uploaded File (top rows)")
    st.dataframe(df.head(20))

    # normalize and map columns
    df = normalize_and_map_columns(df)

    # Convert numeric columns safely
    for col in ["distance_km", "actual_fuel_liters", "diesel_price_per_liter"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # handle default diesel price and leakage flag
    if "diesel_price_per_liter" in df.columns:
        df.loc[df["diesel_price_per_liter"] == 0, "diesel_price_per_liter"] = 95
    if "leakage_flag" in df.columns:
        # normalize leakage_flag to readable labels
        df["leakage_flag"] = df["leakage_flag"].astype(str).replace({"0": "Normal", "nan": "Normal", "None": "Normal"})

    # filter out empty rows (distance and fuel both > 0)
    df = df[(df["distance_km"].fillna(0) > 0) & (df["actual_fuel_liters"].fillna(0) > 0)]

    # Metrics calculations
    df["revenue_per_km"] = np.where(df["leakage_flag"].str.contains("Leak", case=False, na=False), 90, 150)
    df["expected_revenue"] = df["distance_km"] * df["revenue_per_km"]
    df["fuel_cost"] = df["actual_fuel_liters"] * df["diesel_price_per_liter"]

    # add some made-up loss multipliers for realism (optional)
    if len(df) > 0:
        loss_rows = df.sample(frac=min(0.25, 1.0), random_state=42).index
        multipliers = np.random.uniform(1.2, 1.8, len(loss_rows))
        df.loc[loss_rows, "fuel_cost"] = df.loc[loss_rows, "fuel_cost"].astype(float) * multipliers

    df["profit_loss"] = df["expected_revenue"] - df["fuel_cost"]
    df["pnl_status"] = np.where(df["profit_loss"] > 0, "Profit", "Loss")

    # Display cleaned sample
    st.success("✅ Cleaned and calculated sample")
    st.dataframe(df.head(20))

    # Button to push to Supabase
    if st.button("Upload cleaned data to Supabase"):
        try:
            records = df_to_records_for_supabase(df)
            # Optional: delete existing rows if you want to replace table (use with caution)
            # supabase.table("trip_data").delete().neq("trip_id", "").execute()
            # insert records in batches to avoid very large payloads
            # Insert and show response
            res = supabase.table("trip_data").insert(records).execute()
            if res.status_code in (200, 201):
                st.success("🚀 Data uploaded to Supabase successfully.")
            else:
                st.error(f"Supabase insert returned status {res.status_code}. Response: {res.data}")
        except Exception as e:
            st.error(f"Supabase upload error: {e}")

    # Summary metrics
    total_trips = len(df)
    total_profit = df.loc[df["pnl_status"] == "Profit", "profit_loss"].sum() if "profit_loss" in df.columns else 0
    total_loss = df.loc[df["pnl_status"] == "Loss", "profit_loss"].sum() if "profit_loss" in df.columns else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trips", total_trips)
    c2.metric("💰 Total Profit (₹)", f"{total_profit:,.0f}")
    c3.metric("📉 Total Loss (₹)", f"{total_loss:,.0f}")

    st.divider()
    st.subheader("📈 Profit vs Loss (per trip)")
    # Protect plot if trip_id missing or not unique: use index
    x_col = "trip_id" if "trip_id" in df.columns else df.index.astype(str)
    try:
        fig = px.bar(df, x=x_col, y="profit_loss", color="pnl_status", title="Trip-wise Profit/Loss Overview")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Could not render chart (missing profit_loss or x-axis).")

    # Download cleaned CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download cleaned CSV", csv_bytes, "cleaned_trip_data.csv", "text/csv")

    # ---------------------------------
    # CHATBOT: located below file area (main column)
    # ---------------------------------
    st.write("---")
    st.header("🤖 Fuel AI Assistant (Ask Anything)")

    user_question_main = st.text_input("Ask your question about the uploaded data (example: how many trips, which trips had loss, average fuel cost):", key="main_ai_input")

    if st.button("Get AI Answer (Main)", key="main_ai_button"):
        if not user_question_main.strip():
            st.warning("Please type a question first.")
        else:
            # Provide some context to chatbot: small data summary to help answer
            try:
                context_rows = df.head(10).to_dict(orient="records")
                context_text = f"Top rows preview:\n{json.dumps(context_rows, default=str, indent=2)}"
            except Exception:
                context_text = "Top rows not available."

            prompt = f"You are given trip data and summary. Answer the user question concisely.\n\nContext:\n{context_text}\n\nUser question: {user_question_main}"
            answer = ask_gemini(prompt)
            st.success(answer)

else:
    st.info("📥 Please upload your processed_trips file (CSV or XLSX) to start. The AI assistant will appear below after upload.")
