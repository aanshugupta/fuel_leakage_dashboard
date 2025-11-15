import streamlit as st
import google.generativeai as genai

st.title("Gemini Model Checker")

API_KEY = st.secrets.get("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ GEMINI_API_KEY missing in secrets!")
    st.stop()

genai.configure(api_key=API_KEY)

st.write("🔍 Fetching available Gemini models...")

try:
    models = genai.list_models()
    st.success("Models found:")
    for m in models:
        st.write("-", m.name)
except Exception as e:
    st.error(e)
