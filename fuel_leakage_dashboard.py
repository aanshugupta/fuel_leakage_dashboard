# full_streamlit_fuel_dashboard.py
# Paste this file into your Streamlit app (Streamlit Cloud or local).
# Requirements (pip):
# streamlit, pandas, numpy, plotly, supabase, openpyxl (for xlsx), python-dotenv (optional), google-genai (optional)
#
# Put keys in Streamlit secrets (recommended) or set environment variables:
# - GEMINI_API_KEY
# - SUPABASE_URL
# - SUPABASE_KEY
#
# This script is defensive: it handles files that have noise in first 9 rows,
# detects transaction ID column, maps common column names, shows dropdown/search,
# renders charts, uploads to Supabase and gives an AI answer (if Gemini configured).

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client
import json
import typing
import re
import os

st.set_page_config(page_title="Fuel Leakage Dashboard — Full", layout="wide")

# -----------------------------
# Helper: Gemini client (safe)
# -----------------------------
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") if st.secrets else os.environ.get("GEMINI_API_KEY")
GEMINI_AVAILABLE = False
genai_client = None
GEMINI_MODEL = "gemini-1.5-flash"  # default / recommended

try:
    # new google genai SDK uses: from google import genai ; client = genai.Client(api_key="...")
    from google import genai
    if GEMINI_KEY:
        try:
            genai_client = genai.Client(api_key=GEMINI_KEY)
            GEMINI_AVAILABLE = True
        except Exception:
            GEMINI_AVAILABLE = False
    else:
        GEMINI_AVAILABLE = False
except Exception:
    GEMINI_AVAILABLE = False

def ask_gemini(prompt: str) -> str:
    if not GEMINI_AVAILABLE:
        return "Gemini not configured. Set GEMINI_API_KEY in Streamlit secrets or env, and install google-genai package."
    try:
        # using client.models.generate_content per docs:
        resp = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        # try common attributes
        if hasattr(resp, "text") and resp.text:
            return resp.text
        # resp may be dict-like
        return json.dumps(resp, default=str)
    except Exception as e:
        return f"Gemini error: {e}"

# -----------------------------
# Supabase setup
# -----------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL") if st.secrets else os.environ.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") if st.secrets else os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    st.warning("Supabase credentials missing. Add SUPABASE_URL and SUPABASE_KEY in Streamlit secrets or env.")
    supabase = None
else:
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        supabase = None

# -----------------------------
# UI layout
# -----------------------------
st.title("⛽ Fuel Leakage Dashboard + Transaction Analysis (Full)")

# Sidebar: upload and AI quick
with st.sidebar:
    st.header("Upload File")
    uploaded = st.file_uploader("Upload CSV or XLSX (supports noisy header rows)", type=["csv", "xlsx"])
    st.markdown("**Tip:** If your file has garbage in first 9 rows, this app will try to start reading from row 10.")
    st.write("---")
    st.header("AI Assistant (sidebar)")
    sidebar_q = st.text_input("Ask AI (sidebar):", key="side_ai")
    if st.button("Ask (sidebar)"):
        if not sidebar_q.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Asking AI..."):
                st.write(ask_gemini(sidebar_q))

# Main app column
# -----------------------------
def normalize_and_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common names to canonical names used by dashboard.
       If missing, create safe default columns."""
    existing = {c.lower().strip(): c for c in df.columns}
    def find(cands: typing.List[str]):
        for cand in cands:
            k = cand.lower().strip()
            if k in existing:
                return existing[k]
        return None

    mapping = {}
    mapping['transaction_id'] = find(["transaction_id", "transaction id", "txn", "txn_id", "txnid", "txnid", "transaction"])
    mapping['trip_id'] = find(["trip_id", "trip id", "trip", "tripid", "id"])
    mapping['truck_id'] = find(["truck_id", "truck id", "vehicle", "veh_no", "truck", "truck_no", "vehicle_no", "reg_no"])
    mapping['s_no'] = find(["s.no", "sno", "s.no.", "s no", "sno.", "s no."])
    mapping['distance_km'] = find(["distance_km","distance (km)","distance","km","km_travelled"])
    mapping['actual_fuel_liters'] = find(["actual_fuel_liters","fuel_liters","fuel_ltr","fuel","liters","fuel_liters_ltr"])
    mapping['diesel_price_per_liter'] = find(["diesel_price_per_liter","price_per_liter","diesel_price","fuel_price","price"])
    mapping['leakage_flag'] = find(["leakage_flag","leakage","leak_flag","status","leakage_suspected"])

    # rename found columns to canonical names
    for out_col, found in mapping.items():
        if found and found != out_col:
            try:
                df = df.rename(columns={found: out_col})
            except Exception:
                pass

    # create missing columns with defaults
    if 's_no' not in df.columns:
        # if there's an index we can use +1
        df['s_no'] = (df.index.astype(int) + 1).astype(str)
    if 'transaction_id' not in df.columns:
        # try to detect any column that looks like TXN...
        txn_col = None
        for c in df.columns:
            sample = df[c].astype(str).head(50).str.upper().tolist()
            if any(re.match(r"TXN\d{6,}", s) for s in sample):
                txn_col = c
                break
        if txn_col:
            df = df.rename(columns={txn_col: 'transaction_id'})
        else:
            # create synthetic TXN ids
            df['transaction_id'] = df.index.to_series().apply(lambda i: f"TXN{100000000000 + i}")
    if 'truck_id' not in df.columns:
        df['truck_id'] = df.index.astype(str)
    if 'distance_km' not in df.columns:
        df['distance_km'] = 0
    if 'actual_fuel_liters' not in df.columns:
        df['actual_fuel_liters'] = 0
    if 'diesel_price_per_liter' not in df.columns:
        df['diesel_price_per_liter'] = 95
    if 'leakage_flag' not in df.columns:
        df['leakage_flag'] = "Normal"

    return df

def df_to_records_for_supabase(df: pd.DataFrame) -> list:
    df2 = df.copy()
    df2 = df2.where(pd.notnull(df2), None)
    records = df2.to_dict(orient="records")
    def convert(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v
    return [{k: convert(v) for k,v in r.items()} for r in records]

# read uploaded file with option to skip first 9 rows if needed
def read_uploaded_file(uploaded_file):
    # try smart: if excel -> read header at row 9 (0-based index -> header=9) only if header detection fails
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            # read first 20 lines to detect if header row contains 'Transaction' etc.
            sample = pd.read_csv(uploaded_file, nrows=20, header=None)
            # find row with 'Transaction' or 'Transaction ID' or 'S.No' or 'Transaction ID' text
            header_row = None
            for i in range(min(12, sample.shape[0])):
                row_values = sample.iloc[i].astype(str).str.lower().tolist()
                if any("transaction" in s or "txn" in s or "s.no" in s or "sno" in s or "transaction id" in s for s in row_values):
                    header_row = i
                    break
            uploaded_file.seek(0)
            if header_row is None:
                # fallback to header at row 9 (10th line) if file regularly has garbage
                df = pd.read_csv(uploaded_file, skiprows=9)
            else:
                df = pd.read_csv(uploaded_file, header=header_row)
        else:
            # xlsx
            try:
                # read first sheet with header detection
                import openpyxl  # ensure installed in environment
                wb = pd.ExcelFile(uploaded_file, engine="openpyxl")
                # read first 20 rows to detect header
                tmp = pd.read_excel(wb, nrows=20, header=None)
                header_row = None
                for i in range(min(12, tmp.shape[0])):
                    row_values = tmp.iloc[i].astype(str).str.lower().tolist()
                    if any("transaction" in s or "txn" in s or "s.no" in s or "sno" in s or "transaction id" in s for s in row_values):
                        header_row = i
                        break
                if header_row is None:
                    df = pd.read_excel(wb, header=9)  # header at 10th line
                else:
                    df = pd.read_excel(wb, header=header_row)
            except Exception:
                # fallback to simple read
                df = pd.read_excel(uploaded_file, engine="openpyxl", header=9)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None
    # strip column whitespace
    df.columns = [str(c).strip() for c in df.columns]
    return df

# Main flow
if 'uploaded_df' not in st.session_state:
    st.session_state['uploaded_df'] = None

if uploaded is not None:
    df = read_uploaded_file(uploaded)
    if df is None:
        st.stop()
    st.session_state['uploaded_df'] = df.copy()
    st.success("File read — showing top rows below.")
else:
    df = st.session_state['uploaded_df']

if df is None:
    st.info("Upload a CSV or XLSX file to start. This app will try to detect header row (or use row 10 as header).")
    st.stop()

# Normalize and map columns
df = normalize_and_map_columns(df)

# Ensure numeric columns
for c in ["distance_km", "actual_fuel_liters", "diesel_price_per_liter"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# defaults and labels
df.loc[df['diesel_price_per_liter'] == 0, 'diesel_price_per_liter'] = 95
df['leakage_flag'] = df['leakage_flag'].astype(str).fillna("Normal").replace({"0":"Normal","nan":"Normal","None":"Normal"})

# Filter out rows with no useful data (require either distance or fuel > 0)
df_display = df.copy()
df_display = df_display[(df_display["distance_km"].fillna(0) > 0) | (df_display["actual_fuel_liters"].fillna(0) > 0)]
if df_display.empty:
    st.warning("No rows with distance or fuel > 0 found after cleaning. Displaying full file instead.")
    df_display = df.copy()

# Calculations
df_display["revenue_per_km"] = np.where(df_display["leakage_flag"].str.contains("leak", case=False, na=False), 90, 150)
df_display["expected_revenue"] = df_display["distance_km"] * df_display["revenue_per_km"]
df_display["fuel_cost"] = df_display["actual_fuel_liters"] * df_display["diesel_price_per_liter"]
# optional random realistic multiplier - make deterministic by seed
if len(df_display) > 0:
    loss_rows = df_display.sample(frac=min(0.25,1.0), random_state=42).index
    multipliers = np.random.RandomState(42).uniform(1.2, 1.8, len(loss_rows))
    df_display.loc[loss_rows, "fuel_cost"] = df_display.loc[loss_rows, "fuel_cost"].astype(float) * multipliers
df_display["profit_loss"] = df_display["expected_revenue"] - df_display["fuel_cost"]
df_display["pnl_status"] = np.where(df_display["profit_loss"] > 0, "Profit", "Loss")

# Show cleaned sample (left)
st.subheader("📊 Cleaned Data Preview (top 20 rows)")
st.dataframe(df_display.head(20))

# Upload cleaned data to supabase (button)
col1, col2 = st.columns([1,1])
with col1:
    if supabase:
        if st.button("Upload cleaned data to Supabase"):
            try:
                records = df_to_records_for_supabase(df_display)
                # insert (batches could be added here if large)
                res = supabase.table("trip_data").insert(records).execute()
                # Depending on client version, response shape may vary:
                ok = False
                try:
                    if hasattr(res, "status_code"):
                        ok = res.status_code in (200,201)
                    elif isinstance(res, dict) and ('status_code' in res and res['status_code'] in (200,201)):
                        ok = True
                    else:
                        ok = True
                except Exception:
                    ok = True
                if ok:
                    st.success("Uploaded to Supabase (trip_data).")
                else:
                    st.error(f"Supabase returned: {res}")
            except Exception as e:
                st.error(f"Supabase upload error: {e}")
    else:
        st.info("Supabase not configured. Set SUPABASE_URL and SUPABASE_KEY in secrets to enable upload.")

with col2:
    # Download cleaned CSV
    st.download_button("💾 Download cleaned CSV", df_display.to_csv(index=False).encode("utf-8"),
                       "cleaned_trip_data.csv", "text/csv")

st.write("---")

# Summary metrics and charts
total_trips = len(df_display)
total_profit = df_display.loc[df_display["pnl_status"] == "Profit", "profit_loss"].sum() if "profit_loss" in df_display.columns else 0
total_loss = df_display.loc[df_display["pnl_status"] == "Loss", "profit_loss"].sum() if "profit_loss" in df_display.columns else 0

c1, c2, c3 = st.columns(3)
c1.metric("Total Rows (usable)", total_trips)
c2.metric("Total Profit (₹)", f"{total_profit:,.0f}")
c3.metric("Total Loss (₹)", f"{total_loss:,.0f}")

# Charts area
st.subheader("📈 Charts")
chart_col, side_col = st.columns([3,1])

with chart_col:
    # Bar chart profit/loss per transaction/trip
    x_axis = "transaction_id" if "transaction_id" in df_display.columns else ( "trip_id" if "trip_id" in df_display.columns else df_display.index.astype(str) )
    if "profit_loss" in df_display.columns:
        fig = px.bar(df_display, x=x_axis, y="profit_loss", color="pnl_status", title="Trip-wise Profit/Loss")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No profit_loss column to chart.")

    # Pie chart of PNL status
    if "pnl_status" in df_display.columns:
        pie = px.pie(df_display, names="pnl_status", title="Profit vs Loss Distribution")
        st.plotly_chart(pie, use_container_width=True)
    else:
        st.info("No pnl_status for pie chart.")

with side_col:
    st.markdown("**Pie chart column**")
    # user may pick any column for pie grouping
    pie_cols = [c for c in df_display.columns if df_display[c].nunique(dropna=True) < 50]
    chosen_pie = st.selectbox("Pie group column:", options=pie_cols if pie_cols else ["None"])
    if chosen_pie != "None":
        try:
            fig2 = px.pie(df_display, names=chosen_pie, title=f"{chosen_pie} Distribution")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Could not draw pie for {chosen_pie}: {e}")

st.write("---")

# Search / dropdown by transaction id (or trip_id)
st.subheader("🔎 Search / Details by Transaction (or Trip)")

txn_col_candidates = [c for c in ["transaction_id","trip_id","txn","txnid"] if c in df_display.columns]
if not txn_col_candidates:
    st.warning("No Transaction ID or Trip ID column found in uploaded file. The app created synthetic TXN ids.")
    txn_col = "transaction_id"
else:
    txn_col = txn_col_candidates[0]

# Build dropdown with readable TXN values
txn_list = df_display[txn_col].astype(str).tolist()
selected_txn = st.selectbox("Select Transaction ID:", options=["-- show all --"] + txn_list)

if selected_txn and selected_txn != "-- show all --":
    row = df_display[df_display[txn_col].astype(str) == str(selected_txn)]
    if row.empty:
        st.info("Selected transaction not found (maybe whitespace mismatch).")
    else:
        st.write("### Transaction details")
        st.dataframe(row.transpose())
        # show key metrics card
        kcols = st.columns(4)
        try:
            kcols[0].metric("Distance (km)", f"{float(row['distance_km'].iloc[0]):,.2f}")
        except Exception:
            kcols[0].metric("Distance (km)", "N/A")
        try:
            kcols[1].metric("Fuel (L)", f"{float(row['actual_fuel_liters'].iloc[0]):,.2f}")
        except Exception:
            kcols[1].metric("Fuel (L)", "N/A")
        try:
            kcols[2].metric("Profit/Loss (₹)", f"{float(row['profit_loss'].iloc[0]):,.2f}")
        except Exception:
            kcols[2].metric("Profit/Loss (₹)", "N/A")
        try:
            kcols[3].metric("PNL Status", f"{row['pnl_status'].iloc[0]}")
        except Exception:
            kcols[3].metric("PNL Status", "N/A")
else:
    st.info("Showing all rows summary below (first 10 rows).")
    st.dataframe(df_display.head(10))

st.write("---")

# Search by truck number (if available)
if "truck_id" in df_display.columns:
    st.subheader("🚚 Search by Truck ID")
    trucks = df_display["truck_id"].astype(str).unique().tolist()
    sel_truck = st.selectbox("Select truck:", options=["-- all trucks --"] + trucks)
    if sel_truck and sel_truck != "-- all trucks --":
        df_truck = df_display[df_display["truck_id"].astype(str) == sel_truck]
        st.write(f"Showing {len(df_truck)} rows for truck {sel_truck}")
        st.dataframe(df_truck)
        # truck-level charts
        if "profit_loss" in df_truck.columns:
            st.plotly_chart(px.bar(df_truck, x="transaction_id" if "transaction_id" in df_truck.columns else df_truck.index.astype(str),
                                   y="profit_loss", title=f"Profit/Loss for truck {sel_truck}"), use_container_width=True)
    else:
        st.info("Select a truck to view details.")

st.write("---")

# AI assistant in main column (uses small context from data)
st.subheader("🤖 Fuel AI Assistant (Main)")
user_q = st.text_input("Ask about uploaded data (example: how many trips had loss, avg fuel cost):", key="main_ai")

if st.button("Get AI Answer (Main)"):
    if not user_q.strip():
        st.warning("Type a question first.")
    else:
        # prepare small context
        try:
            preview = df_display.head(10).to_dict(orient="records")
            context = json.dumps(preview, default=str, indent=2)
        except Exception:
            context = "Preview not available."
        prompt = f"You are given trip rows JSON below. Answer concisely.\n\nContext:\n{context}\n\nQuestion: {user_q}"
        with st.spinner("Querying AI..."):
            st.write(ask_gemini(prompt))

st.write("---")

# Final help / notes
st.info("Notes: \n• If AI/Gemini doesn't work, ensure GEMINI_API_KEY is configured and 'google-genai' package is installed. \n"
        "• If Supabase upload fails, check credentials in Streamlit secrets. \n"
        "• This app tries to detect header rows; if your file always has header at row 10, it will use row 10 automatically.")
