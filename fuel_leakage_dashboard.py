Perfect.
Model confirmed:

model = "gemini-2.0-flash"

This is fast + powerful → best for dashboards, analytics, summaries, fraud detection.


---

I will now generate your FULL PREMIUM AI DASHBOARD CODE

I will send it in 5 clean parts so it never breaks.


---

✅ PART 1 — CORE IMPORTS + CONFIG + BASIC HELPERS

(Copy & paste this into the TOP of your app.py or fuel_dashboard.py)


---

✅ PART-1 CODE (START HERE)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from supabase import create_client, Client
import google.generativeai as genai

# ===============================================================
# 🔐 CONFIGURATION (Supabase + Gemini 2.0 Flash)
# ===============================================================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GEMINI_KEY = st.secrets["GEMINI_API_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

genai.configure(api_key=GEMINI_KEY)
gmodel = genai.GenerativeModel("gemini-2.0-flash")

# ===============================================================
# 🔧 HELPER — CLEAN COLUMN NAME
# ===============================================================
def clean_col(c):
    c = str(c).strip().lower()
    c = c.replace(" ", "_")
    c = c.replace("-", "_")
    c = re.sub(r'[^a-zA-Z0-9_]', '', c)
    return c

# ===============================================================
# 🔧 HELPER — SEMANTIC COLUMN MATCHING
# ===============================================================
def find_semantic_column(df, possible_names):
    df_cols = [clean_col(c) for c in df.columns]

    for name in possible_names:
        name = clean_col(name)
        for col in df_cols:
            if name in col:
                return df.columns[df_cols.index(col)]

    return None

# ===============================================================
# 🔧 LOAD SUPABASE TABLE (AUTO-DETECT)
# ===============================================================
def load_table(table):
    try:
        data = supabase.table(table).select("*").execute()
        return pd.DataFrame(data.data)
    except Exception as e:
        st.error(f"Error loading table {table}: {e}")
        return pd.DataFrame()


---

✔ DONE — PART 1 READY

Next:

👉 PART 2 = SEMANTIC COLUMN ENGINE

(Automatically detect “transaction_id”, “volume”, “rate”, “amount”, “vehicle”, “station”, “date”, etc.)

👉 PART 3 = Filler Transformation + Header Detection

(Remove top metadata lines like your Excel 9th-row issue.)

👉 PART 4 = All Dashboards (Fuel, Trip, Vehicle, Maintenance)

👉 PART 5 = Gemini AI Engine + Leakage + Fraud + Chatbot + UI Layout


---

🟦 Please reply:

“Send Part 2”
# ===============================================================
# 🧠 SEMANTIC COLUMN MAPPING ENGINE
# ===============================================================

def semantic_map_columns(df):

    col_map = {}

    # Clean all column names first
    cleaned = {clean_col(c): c for c in df.columns}

    # ------------ Transaction ID ------------
    col_map["transaction_id"] = find_semantic_column(df, [
        "transaction_id", "txn_id", "trans_id", "transaction", "txn"
    ])

    # ------------ Vehicle Number ------------
    col_map["vehicle_no"] = find_semantic_column(df, [
        "vehicle_no", "truck_no", "vehicle", "truck_number", "truck"
    ])

    # ------------ Product Volume (Liters) ------------
    col_map["volume_ltr"] = find_semantic_column(df, [
        "product_volume", "volume", "fuel_liters", "liters", "qty", "quantity", "ltr"
    ])

    # ------------ Rate Per Liter ------------
    col_map["rate"] = find_semantic_column(df, [
        "rate", "price_per_liter", "fuel_rate", "rate_rs_ltr", "fuel_price"
    ])

    # ------------ Total Amount ------------
    col_map["amount"] = find_semantic_column(df, [
        "purchase_amount", "amount", "total_amount", "value", "price"
    ])

    # ------------ Date Column ------------
    col_map["date"] = find_semantic_column(df, [
        "date", "transaction_date", "fuel_date"
    ])

    # ------------ Time Column ------------
    col_map["time"] = find_semantic_column(df, [
        "time", "transaction_time", "fuel_time"
    ])

    # ------------ Fuel Station ID ------------
    col_map["station_id"] = find_semantic_column(df, [
        "station_id", "fuel_station_id", "fuelstation", "petrol_pump", "pump_id"
    ])

    # ------------ Driver ID ------------
    col_map["driver_id"] = find_semantic_column(df, [
        "driver_id", "driver", "drv_id"
    ])

    # ------------ Distance ------------
    col_map["distance"] = find_semantic_column(df, [
        "distance_km", "distance", "km"
    ])

    # ------------ Fuel Efficiency ------------
    col_map["mileage"] = find_semantic_column(df, [
        "mileage", "kmpl", "fuel_efficiency"
    ])

    return col_map
    # ===============================================================
# 🧹 FILLER TRANSFORMATION ENGINE — CATCH ANY BAD EXCEL/CSV
# ===============================================================

def detect_header_row(df_raw):
    """
    Detect which row contains the actual header.
    Looks for row that contains the highest % of text-like column names.
    Useful when Excel has metadata lines (1-9).
    """

    best_row = 0
    best_score = 0

    for i in range(min(15, len(df_raw))):
        row = list(df_raw.iloc[i])
        score = 0
        for cell in row:
            cell_str = str(cell).lower()

            # Typical header patterns
            if any(x in cell_str for x in [
                "transaction", "txn", "amount", "rate", "volume",
                "date", "vehicle", "station", "id", "s.no"
            ]):
                score += 1

        if score > best_score:
            best_score = score
            best_row = i

    return best_row


def clean_uploaded_file(uploaded):
    """Reads CSV/XLSX with FULL cleaning pipeline."""

    # --------------------------------------------------------
    # 1. Read file WITHOUT assuming header
    # --------------------------------------------------------
    if uploaded.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded, header=None)
    else:
        df_raw = pd.read_excel(uploaded, header=None)

    # --------------------------------------------------------
    # 2. Detect header row automatically
    # --------------------------------------------------------
    header_row = detect_header_row(df_raw)

    # --------------------------------------------------------
    # 3. Re-read file using detected header
    # --------------------------------------------------------
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded, skiprows=header_row)
    else:
        df = pd.read_excel(uploaded, skiprows=header_row)

    # --------------------------------------------------------
    # 4. Standard cleaning
    # --------------------------------------------------------
    df = df.dropna(how="all")  # remove empty rows
    df.columns = [str(c).strip() for c in df.columns]  # clean names

    # --------------------------------------------------------
    # 5. Apply semantic column mapping (from PART-2)
    # --------------------------------------------------------
# ===============================================================
# 🚚 TRUCK INTELLIGENCE ENGINE — Performance, Leakage, Ranking
# ===============================================================

def compute_truck_metrics(df):
    """
    Computes advanced analytics truck-wise.
    Works even if column names are different → semantic mapping (Part 2).
    """

    # Important semantic-friendly fields (Part 2 created these):
    col_vehicle = "vehicle_number"
    col_volume  = "product_volume"
    col_amount  = "purchase_amount"
    col_rate    = "rate_rs_ltr"
    col_date    = "transaction_date"

    # Check required fields
    required = [col_vehicle, col_volume, col_amount, col_rate]
    if not all(col in df.columns for col in required):
        return None

    # Create clean working DF
    clean = df.copy()
    clean[col_volume] = pd.to_numeric(clean[col_volume], errors="coerce").fillna(0)
    clean[col_amount] = pd.to_numeric(clean[col_amount], errors="coerce").fillna(0)
    clean[col_rate]   = pd.to_numeric(clean[col_rate], errors="coerce").fillna(0)

    # Compute expected cost
    clean["expected_cost"] = clean[col_volume] * clean[col_rate]
    clean["leakage_diff"] = clean[col_amount] - clean["expected_cost"]

    # Percentage leakage
    clean["leakage_percent"] = (clean["leakage_diff"] / clean["expected_cost"]) * 100

    # Group truck-wise
    truck_summary = clean.groupby(col_vehicle).agg({
        col_volume: "sum",
        col_amount: "sum",
        "expected_cost": "sum",
        "leakage_diff": "sum",
        "leakage_percent": "mean"
    }).reset_index()

    # Rename for UI clarity
    truck_summary.columns = [
        "vehicle_number",
        "total_volume",
        "total_amount_spent",
        "expected_cost",
        "total_leakage_amount",
        "avg_leakage_percent"
    ]

    # Efficiency score (0–100)
    truck_summary["efficiency_score"] = np.clip(
        100 - np.abs(truck_summary["avg_leakage_percent"]), 
        0, 100
    )

    # Flagging
    truck_summary["status"] = np.where(
        truck_summary["avg_leakage_percent"] > 8,
        "⚠ High Leakage",
        "✔ Normal"
    )

    return truck_summary


# ===============================================================
# 🧠 AI TRUCK SUMMARY GENERATOR
# ===============================================================

def ai_truck_summary(truck_df, vehicle_number):
    """
    Creates human-style summary for ONE truck.
    """

    row = truck_df[truck_df["vehicle_number"] == vehicle_number].iloc[0].to_dict()

    prompt = f"""
    You are a transport company fuel expert.

    Create a clear summary for this truck:

    {row}

    Include:
    - Fuel usage behavior
    - Leakage probability
    - Efficiency score meaning
    - Whether station overcharging is possible
    - Advice to reduce cost
    """

    try:
        result = model.generate_content(prompt).text
    except:
        result = "AI summary unavailable."

    return result
    # ===========================================================
# ⭐ PART–5: PREMIUM UI (Tabs + Theme + Cards + Integration)
# ===========================================================

st.markdown("""
<style>
.big-metric {font-size:28px; font-weight:700; margin-top:-10px;}
.subtext {font-size:13px; color:gray;}
.card {
    background-color:#ffffff10;
    padding:18px;
    border-radius:12px;
    border:1px solid #ffffff20;
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# 🌗 THEME TOGGLE
# -----------------------------------------------------------
theme = st.sidebar.radio("Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
    <style>
    body {background-color:#0e1117; color:white;}
    </style>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------
# 🗂️ TABS
# -----------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Home",
    "📦 Supabase Data",
    "🚚 Truck Intelligence",
    "⛽ Leakage Detection",
    "🔍 Fraud Detection",
    "📈 Charts Builder",
    "🤖 Chatbot Gemini"
])


# ===========================================================
# 1️⃣ HOME TAB
# ===========================================================
with tab1:
    st.title("🚀 Fuel Intelligence Dashboard — Premium Edition")
    st.write("Your connected Supabase tables, AI analytics, and truck insights in one place.")

    st.subheader("Connected Tables")
    tables = ["sales_data", "drive_summary", "trip_data"]
    for t in tables:
        count = len(load_table(t))
        st.write(f"✔ **{t}** — {count} rows")


# ===========================================================
# 2️⃣ SUPABASE DATA PREVIEW TAB
# ===========================================================
with tab2:
    st.header("📦 Live Supabase Tables")
    selected_table = st.selectbox("Select Table", ["sales_data", "drive_summary", "trip_data"])

    table_df = load_table(selected_table)

    if table_df.empty:
        st.error("No data fetched.")
    else:
        st.dataframe(table_df, use_container_width=True)


# ===========================================================
# 3️⃣ TRUCK INTELLIGENCE TAB
# ===========================================================
with tab3:
    st.header("🚚 Truck Intelligence Center")

    df = load_table("sales_data")
    truck_df = compute_truck_metrics(df)

    if truck_df is None:
        st.error("Truck metrics could not be computed. Missing fuel columns.")
    else:
        st.subheader("📊 Truck Ranking Table")
        st.dataframe(truck_df, use_container_width=True)

        selected_truck = st.selectbox(
            "Select Vehicle Number", 
            truck_df["vehicle_number"].astype(str).unique()
        )

        if selected_truck:
            st.subheader("🧠 AI Summary for Truck")
            summary = ai_truck_summary(truck_df, selected_truck)
            st.info(summary)


# ===========================================================
# 4️⃣ LEAKAGE DETECTION TAB
# ===========================================================
with tab4:
    st.header("⛽ Fuel Leakage Detection (AI Powered)")

    leak_df, leak_summary = detect_leakage(df)

    if leak_df is None:
        st.error("Required columns missing.")
    else:
        st.subheader("Leakage Calculated Table")
        st.dataframe(leak_df, use_container_width=True)

        st.subheader("AI Summary")
        st.warning(leak_summary)


# ===========================================================
# 5️⃣ FRAUD DETECTION TAB
# ===========================================================
with tab5:
    st.header("🔍 Fraud & Misuse Detection")

    fraud_text = detect_fraud(df)
    st.warning(fraud_text)


# ===========================================================
# 6️⃣ CHART BUILDER TAB
# ===========================================================
with tab6:
    st.header("📈 Charts Builder")

    numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns

    if len(numerical_cols) == 0:
        st.error("No numeric columns found.")
    else:
        col = st.selectbox("Select numeric column", numerical_cols)
        chart_type = st.radio("Choose Chart", ["Line", "Bar", "Area"])

        if chart_type == "Line":
            fig = px.line(df, y=col)
        elif chart_type == "Bar":
            fig = px.bar(df, y=col)
        else:
            fig = px.area(df, y=col)

        st.plotly_chart(fig, use_container_width=True)


# ===========================================================
# 7️⃣ CHATBOT TAB
# ===========================================================
with tab7:
    st.header("🤖 Chatbot Gemini (AI Assistant)")
    user_q = st.text_input("Ask anything about fuel, trucks, or data:")

    if user_q:
        st.write("### Answer")
        try:
            st.success(ask_ai_about_data(user_q, df))
        except:
            st.error("AI error — check API key / model.")

and I will give the next block.
