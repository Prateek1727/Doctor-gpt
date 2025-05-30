import streamlit as st
try:
    from src.chatbot import doctor_gpt
except ImportError as e:
    st.error(f"Failed to import doctor_gpt: {str(e)}. Please ensure src.chatbot defines doctor_gpt.")
    st.stop()

st.set_page_config(page_title='🩺 Doctor GPT', layout='centered', page_icon='🩺')
st.title("🩺 Doctor GPT Chat AI")

# Test query
response = doctor_gpt("What are the symptoms of diabetes?")
st.write(f"Test Response: {response}")
