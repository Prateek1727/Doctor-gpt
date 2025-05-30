import streamlit as st
from streamlit_chat import message
import random
try:
    from src.chatbot import doctor_gpt
except ImportError as e:
    st.error(f"Failed to import doctor_gpt: {str(e)}. Please ensure src.chatbot defines doctor_gpt.")
    st.stop()
from loguru import logger
import time

# Configure logging
logger.add("doctor_gpt_app.log", rotation="10 MB", level="INFO")

st.set_page_config(page_title='🤖 Doctor GPT', layout='centered', page_icon='🤖')
st.title("🤖 Doctor GPT Chat AI")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = random.randint(0, 100000)
    logger.info(f"Initialized new session with ID: {st.session_state.session_id}")

# Initial message
INIT_MESSAGE = {
    "role": "assistant",
    "content": "Hello! I am your Doctor GPT Chat Agent, here to help answer any questions you have about health conditions."
}

if "messages" not in st.session_state:
    st.session_state.messages = [INIT_MESSAGE]

def generate_response(input_text):
    """
    Generate a response using the FAISS-based doctor_gpt function.

    Args:
        input_text (str): User query about a health condition.

    Returns:
        str: Response from doctor_gpt or an error message.
    """
    try:
        start_time = time.time()
        output = doctor_gpt(user_query=input_text)
        elapsed_time = time.time() - start_time
        logger.info(f"Generated response for query '{input_text}' in {elapsed_time:.2f} seconds")
        return output
    except Exception as e:
        logger.error(f"Error generating response for query '{input_text}': {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}. Please try again or check if the system is properly configured."

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Get user input
user_input = st.chat_input(placeholder="Ask about any health condition...", key="input")

# Process user input
if user_input:
    logger.info(f"Session {st.session_state.session_id}: User query: {user_input}")
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.spinner("Thinking..."):
            response = generate_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# Test block
if __name__ == "__main__":
    test_query = "What are the symptoms of diabetes?"
    logger.info(f"Running test query: {test_query}")
    response = generate_response(test_query)
    print(f"Test Query: {test_query}")
