import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from supabase import create_client, Client

# Supabase connection setup
SUPABASE_URL = "https://pyanhlpwloofwzpulcpi.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB5YW5obHB3bG9vZnd6cHVsY3BpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI3NjQyMzcsImV4cCI6MjA3ODM0MDIzN30.vUydKFP8kPOudO1bup4z1JYCYrWAMrI6RZol0pvQiCw"  # replace with your anon key from Supabase API
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="Fuel Leakage Dashboard", layout="wide")
st.title("⛽ Fuel Leakage Detection & Efficiency Dashboard")

st.sidebar.header("⚙️ Data Input")
upload = st.sidebar.file_uploader("📂 Upload processed_trips.csv", type=["csv"])
generate = st.sidebar.button("🚀 Generate Sample Data")

def simulate_data():
    np.random.seed(42)
    n_trucks, trips = 15, 20
    rows=[]
    for t in range(1,n_trucks+1):
        truck=f"T{t:02d}"
        for i in range(1,trips+1):
            distance=np.random.uniform(50,900)
            mil=np.random.uniform(3,3.5)
            exp=distance/mil
            noise=np.random.uniform(-0.2,0.2)
            if np.random.rand()<0.05: noise=np.random.uniform(0.25,0.5)
            act=exp*(1+noise)
            price=np.random.uniform(88,95)
            rows.append({
                "trip_id":f"{truck}_{i:03d}",
                "truck_id":truck,
                "driver_id":f"D{np.random.randint(1,31):02d}",
                "trip_date":pd.Timestamp("2025-10-01")+pd.Timedelta(days=np.random.randint(0,30)),
                "route_type":np.random.choice(["Fixed","Variable"]),
                "distance_km":distance,
                "expected_fuel_liters":exp,
                "actual_fuel_liters":act,
                "diesel_price_per_liter":price
            })
    df=pd.DataFrame(rows)
    df["variance_pct"]=((df.actual_fuel_liters-df.expected_fuel_liters)/df.expected_fuel_liters)*100
    df["leakage_flag"]=np.where(df.variance_pct>15,"Leakage Suspected",
                        np.where(df.variance_pct<-10,"Underuse","Normal"))
    df["leakage_cost"]=(df.actual_fuel_liters-df.expected_fuel_liters)*df.diesel_price_per_liter
    return df

# --- Fetch Data from Supabase instead of CSV upload ---
try:
    response = supabase.table("trip_data").select("*").execute()
    df = pd.DataFrame(response.data)

    if df.empty:
        st.warning("⚠️ No data found in Supabase 'trip_data' table.")
        st.stop()
    else:
        st.sidebar.success("✅ Data loaded from Supabase successfully!")

except Exception as e:
    st.error(f"❌ Failed to fetch data from Supabase: {e}")
    st.stop()

total=len(df)
avgv=df["variance_pct"].mean()
leak=df[df.leakage_flag=="Leakage Suspected"]
leak_l=max((leak.actual_fuel_liters-leak.expected_fuel_liters).sum(),0)
leak_cost=max(leak["leakage_cost"].sum(),0)
pct=len(leak)/total if total>0 else 0

c1,c2,c3,c4,c5=st.columns(5)
c1.metric("Total Trips",total)
c2.metric("Avg Variance (%)",f"{avgv:.2f}")
c3.metric("Total Leakage (L)",f"{leak_l:.1f}")
c4.metric("Leakage Cost (₹)",f"{leak_cost:,.0f}")
c5.metric("% Trips Leakage",f"{pct:.1%}")

st.divider()
tab1,tab2,tab3,tab4,tab5=st.tabs(["Actual vs Expected","Leakage Pie","Cost by Driver","Variance Trend","Distance vs Fuel"])

with tab1:
    st.plotly_chart(px.bar(df,x="trip_id",y=["expected_fuel_liters","actual_fuel_liters"],
                           title="Actual vs Expected Fuel",barmode="group"),use_container_width=True)
with tab2:
    st.plotly_chart(px.pie(df,names="leakage_flag",title="Leakage Categories"),use_container_width=True)
with tab3:
    tmp=df.groupby("driver_id")["leakage_cost"].sum().reset_index()
    st.plotly_chart(px.bar(tmp,x="driver_id",y="leakage_cost",title="Leakage Cost per Driver"),use_container_width=True)
with tab4:
    trend=df.groupby("trip_date")["variance_pct"].mean().reset_index()
    st.plotly_chart(px.line(trend,x="trip_date",y="variance_pct",title="Variance Trend"),use_container_width=True)
with tab5:
    st.plotly_chart(px.scatter(df,x="distance_km",y="actual_fuel_liters",color="leakage_flag",
                               title="Distance vs Fuel"),use_container_width=True)

st.divider()
st.subheader("🧭 Trip Details")
st.dataframe(df)
st.download_button("💾 Download Leakage Report (CSV)",
                   df.to_csv(index=False),"leakage_report.csv","text/csv")



