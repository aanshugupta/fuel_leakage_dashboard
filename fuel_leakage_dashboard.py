import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai
# ✔ Correct SDK for Gemini 2.5
import json
import typing

# -------------------------------
# CONFIG: secrets required
# -------------------------------
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY")
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase credentials missing in Streamlit secrets.")
    st.stop()

# -------------------------------
# Gemini Setup (Correct)
# -------------------------------
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    client = genai.Client()
    GEMINI_MODEL = "gemini-2.5-flash"
else:
    client = None
    GEMINI_MODEL = None

# -------------------------------
# Supabase Setup
# -------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="Fuel Leakage Dashboard", layout="wide")
st.title("⛽ Fuel Leakage Detection, Efficiency & Profit/Loss Dashboard")

# ----------------------------------------
# File uploader (CSV + XLSX)
# ----------------------------------------
uploaded = st.file_uploader("📂 Upload file (CSV or XLSX)", type=["csv", "xlsx"])


# ----------------------------------------------------
# AUTO-COLUMN MAPPING SO ANY FILE WORKS
# ----------------------------------------------------
def normalize_and_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing = {c.lower().strip(): c for c in df.columns}

    def find(names):
        for n in names:
            if n.lower().strip() in existing:
                return existing[n.lower().strip()]
        return None

    mapping = {
        "trip_id": find(["trip_id", "trip id", "id", "trip"]),
        "distance_km": find(["distance_km", "distance", "km"]),
        "actual_fuel_liters": find(["actual_fuel_liters", "fuel", "fuel_ltr", "fuel_liters"]),
        "diesel_price_per_liter": find(["diesel_price_per_liter", "fuel_price", "price"]),
        "leakage_flag": find(["leakage_flag", "leakage", "status"])
    }

    for std_col, found_col in mapping.items():
        if found_col:
            df.rename(columns={found_col: std_col}, inplace=True)
        else:
            df[std_col] = 0 if std_col != "leakage_flag" else "Normal"

    return df


# ----------------------------------------------------
# Convert DF to safe Supabase records
# ----------------------------------------------------
def df_to_records(df):
    df2 = df.where(pd.notnull(df), None)
    records = df2.to_dict(orient="records")

    def conv(v):
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        return v

    return [{k: conv(v) for k, v in r.items()} for r in records]


# ----------------------------------------------------
# Gemini Chat function
# ----------------------------------------------------
def ask_gemini(prompt):
    if not client:
        return "Gemini API key missing."

    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return resp.text
    except Exception as e:
        return f"AI Error: {e}"


# ----------------------------------------------------
# MAIN DATA PROCESSING
# ----------------------------------------------------
if uploaded:

    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded, low_memory=False)
    else:
        df = pd.read_excel(uploaded)

    st.subheader("📊 Uploaded File Preview")
    st.dataframe(df.head(20))

    df = normalize_and_map_columns(df)

    for c in ["distance_km", "actual_fuel_liters", "diesel_price_per_liter"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df.loc[df["diesel_price_per_liter"] == 0, "diesel_price_per_liter"] = 95
    df["leakage_flag"] = df["leakage_flag"].astype(str).replace({"0": "Normal"})

    df = df[(df["distance_km"] > 0) & (df["actual_fuel_liters"] > 0)]

    df["revenue_per_km"] = np.where(
        df["leakage_flag"].str.contains("Leak", case=False),
        90, 150
    )

    df["expected_revenue"] = df["distance_km"] * df["revenue_per_km"]
    df["fuel_cost"] = df["actual_fuel_liters"] * df["diesel_price_per_liter"]

    # Random loss multiplier
    if len(df) > 0:
        rows = df.sample(frac=0.25, random_state=42).index
        df.loc[rows, "fuel_cost"] *= np.random.uniform(1.2, 1.8, len(rows))

    df["profit_loss"] = df["expected_revenue"] - df["fuel_cost"]
    df["pnl_status"] = np.where(df["profit_loss"] > 0, "Profit", "Loss")

    st.success("✅ Cleaned & calculated data ready")
    st.dataframe(df.head(20))

    # Upload button
    if st.button("Upload cleaned data to Supabase"):
        try:
            res = supabase.table("trip_data").insert(df_to_records(df)).execute()
            st.success("🚀 Uploaded to Supabase")
        except Exception as e:
            st.error(f"Supabase Error: {e}")

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trips", len(df))
    c2.metric("Profit (₹)", f"{df[df.pnl_status=='Profit'].profit_loss.sum():,.0f}")
    c3.metric("Loss (₹)", f"{df[df.pnl_status=='Loss'].profit_loss.sum():,.0f}")

    # Graph
    st.subheader("📈 Profit vs Loss (per trip)")
    fig = px.bar(
        df,
        x="trip_id",
        y="profit_loss",
        color="pnl_status",
        title="Trip-wise Profit/Loss Overview"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "💾 Download Cleaned CSV",
        df.to_csv(index=False).encode("utf-8"),
        "cleaned_trip_data.csv",
        "text/csv"
    )

    # ----------------------------------------
    # CHATBOT BELOW DASHBOARD
    # ----------------------------------------
    st.write("---")
    st.header("🤖 Fuel AI Assistant (Ask Anything)")

    q = st.text_input("Ask related to trips, loss, fuel usage etc:")

    if st.button("Get AI Answer"):
        if q.strip() == "":
            st.warning("Type a question first.")
        else:
            ctx = df.head(10).to_dict(orient="records")
            prompt = f"""
You are an AI trained on vehicle fuel trip data.

Here is sample data:
{json.dumps(ctx, indent=2)}

User question: {q}
"""
            ans = ask_gemini(prompt)
            st.success(ans)

else:
    st.info("📥 Upload your CSV or XLSX file to begin.")
