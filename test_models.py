import streamlit as st
import google.generativeai as genai
import os

st.title("🔍 Gemini Model Access Checker")

try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    models = genai.list_models()

    st.success("Your API key can access these models:")
    for m in models:
        st.write("➡️ " + m.name)

except Exception as e:
    st.error("❌ Error: " + str(e))
