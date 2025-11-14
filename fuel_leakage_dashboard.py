import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import google.generativeai as genai
import json
import typing

# ------------------------------------------------------------
# GEMINI SETUP (REQUIRED: add GEMINI_API_KEY in Streamlit Secrets)
# ------------------------------------------------------------
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    GEMINI_MODEL = "gemini-1.5-flash"  # Safe + supported model
else:
    GEMINI_MODEL = None

# Chatbot function
def ask_gemini(question: str):
    if not GEMINI_MODEL:
        return "Gemini API key missing or invalid."
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(question)
        return resp.text if hasattr(resp, "text") else str(resp)
    except Exception as e:
        if "404" in str(e):
            return "Model not found. Use gemini-1.5-flash."
        return f"Gemini Error: {e}"

# ------------------------------------------------------------
# SUPABASE SETUP (add URL + KEY in Streamlit Secrets)
# ------------------------------------------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------------------------------------------
# PAGE UI & TITLE
# ------------------------------------------------------------
st.set_page_config(page_title="Fuel Leakage Dashboard", layout="wide")
st.title("⛽ Fuel Leakage Detection, Efficiency & Profit/Loss Dashboard")

# ------------------------------------------------------------
# FILE UPLOADER
# ------------------------------------------------------------
uploaded = st.sidebar.file_uploader(
    "📂 Upload CSV or Excel",
    type=["csv", "xlsx"]
)

# ------------------------------------------------------------
# SIDEBAR CHATBOT (BELOW UPLOADER)
# ------------------------------------------------------------
st.sidebar.write("---")
st.sidebar.subheader("🤖 AI Assistant")

question_sidebar = st.sidebar.text_input("Ask AI about your data:")
if st.sidebar.button("Ask"):
    if question_sidebar.strip():
        st.sidebar.success(ask_gemini(question_sidebar))
    else:
        st.sidebar.warning("Type a question first.")

# ------------------------------------------------------------
# COLUMN NAME NORMALIZER
# ------------------------------------------------------------
def normalize_columns(df):
    original = {c.lower().strip(): c for c in df.columns}

    def match(names):
        for n in names:
            if n.lower() in original:
                return original[n.lower()]
        return None

    mapping = {
        "trip_id": match(["trip_id", "trip id", "trip", "id", "tripid"]),
        "truck_id": match(["truck_id", "truck", "vehicle_no", "vehicle", "truck number"]),
        "distance_km": match(["distance_km", "distance", "km", "distance (km)"]),
        "actual_fuel_liters": match(["actual_fuel_liters", "fuel", "fuel_liters", "fuel_ltr"]),
        "diesel_price_per_liter": match(["diesel_price_per_liter", "price", "fuel_price"]),
        "leakage_flag": match(["leakage_flag", "leakage", "status"])
    }

    for new_col, old_col in mapping.items():
        if old_col:
            df = df.rename(columns={old_col: new_col})
        else:
            df[new_col] = "Normal" if new_col == "leakage_flag" else 0

    return df

# ------------------------------------------------------------
# MAIN — AFTER FILE UPLOAD
# ------------------------------------------------------------
if uploaded:

    # Load CSV or Excel
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.subheader("📄 Uploaded File Preview")
    st.dataframe(df.head())

    # Normalize columns
    df = normalize_columns(df)

    # Convert numbers
    for col in ["distance_km", "actual_fuel_liters", "diesel_price_per_liter"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Fix missing values
    df["diesel_price_per_liter"] = df["diesel_price_per_liter"].replace(0, 95)

    # Calculate metrics
    df["revenue_per_km"] = df["distance_km"].apply(lambda x: 150 if x > 0 else 0)
    df["expected_revenue"] = df["distance_km"] * df["revenue_per_km"]
    df["fuel_cost"] = df["actual_fuel_liters"] * df["diesel_price_per_liter"]
    df["profit_loss"] = df["expected_revenue"] - df["fuel_cost"]
    df["pnl_status"] = df["profit_loss"].apply(lambda x: "Profit" if x > 0 else "Loss")

    # Show cleaned data
    st.success("Cleaned & Processed Data")
    st.dataframe(df.head())

    # ------------------------------------------------------------
    # SUPABASE UPLOAD BUTTON
    # ------------------------------------------------------------
    if st.button("Upload to Supabase"):
        try:
            records = df.to_dict(orient="records")
            res = supabase.table("trip_data").insert(records).execute()
            st.success("Data uploaded to Supabase!")
        except Exception as e:
            st.error(f"Supabase error: {e}")

    # Summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trips", len(df))
    c2.metric("Profit (₹)", f"{df[df.pnl_status=='Profit'].profit_loss.sum():,.0f}")
    c3.metric("Loss (₹)", f"{df[df.pnl_status=='Loss'].profit_loss.sum():,.0f}")

    st.write("---")
    st.subheader("📈 Profit-Loss Graph")
    st.plotly_chart(
        px.bar(df, x=df.index, y="profit_loss", color="pnl_status"),
        use_container_width=True
    )

    # ------------------------------------------------------------
    # 🔍 TRUCK / TRIP SEARCH SECTION
    # ------------------------------------------------------------
    st.write("---")
    st.header("🚚 Search Truck / Trip Details")

    # Detect truck_id or trip_id
    id_col = None
    for col in ["truck_id", "trip_id"]:
        if col in df.columns:
            id_col = col
            break

    if not id_col:
        st.warning("No truck_id or trip_id found in this file.")
    else:
        unique_list = df[id_col].astype(str).unique()

        selected = st.selectbox("Select Truck / Trip:", unique_list)

        filtered = df[df[id_col].astype(str) == str(selected)]

        st.subheader(f"📄 Details for {selected}")
        st.dataframe(filtered)

        # Truck summary
        st.success(
            f"""
            Trips: {len(filtered)}
            Total Distance: {filtered['distance_km'].sum():,.2f} km  
            Fuel Used: {filtered['actual_fuel_liters'].sum():,.2f} L  
            Total Profit/Loss: ₹{filtered['profit_loss'].sum():,.2f}
            """
        )

        # Truck graph
        st.subheader("📊 Truck Performance Graph")
        st.plotly_chart(
            px.line(filtered, x=filtered.index, y="profit_loss",
                    title=f"Profit/Loss Trend for {selected}"),
            use_container_width=True
        )

    # ------------------------------------------------------------
    # MAIN CHATBOT
    # ------------------------------------------------------------
    st.write("---")
    st.header("🤖 AI Assistant (Full Data Q/A)")

    q = st.text_input("Ask anything about the uploaded data:")
    if st.button("Get AI Answer"):
        context = df.head(20).to_dict(orient="records")
        prompt = f"Here is sample trip data:\n{json.dumps(context, indent=2)}\n\nUser question: {q}"
        st.success(ask_gemini(prompt))

else:
    st.info("Please upload a file to begin.")
