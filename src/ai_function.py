import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from loguru import logger

# Load environment variables from .env file
load_dotenv()

# Validate GROQ_API_KEY
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in your environment or Streamlit Cloud secrets.")

# Set GROQ_API_KEY in os.environ (required by langchain_groq)
os.environ['GROQ_API_KEY'] = groq_api_key

# ---------------------------- LLM --------------------------------------
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

def medicalbot_ai(user_query: str, doc_list: list) -> str:
    template = """
    You are a medical consultant AI chatbot. Your role is to provide accurate and reliable answers to user questions based on the provided documents. Use the information from the `doc_list` to address the `user_query` thoroughly and correctly. Ensure that your response is:

    - **Accurate:** Base your answers solely on the information in the provided documents.
    - **Conversational:** Maintain a friendly and approachable tone.
    - **Mature and Consultancy-Oriented:** Present information in a professional and trustworthy manner.

    **Inputs:**
    1. `user_query`: {user_query} The question posed by the user.
    2. `doc_list`: {doc_list} A list of documents containing relevant information related to the user's question.

    **Instructions:**
    - Analyze the `user_query` and identify the key information needed to answer it.
    - Review the `doc_list` to find relevant information that addresses the query.
    - Construct a response that is clear, concise, and directly answers the user's question using the information from the documents.
    - Avoid introducing information not present in the `doc_list`.
    - If the `user_query` have nothing similar to what is in the `doc_list`, return document not found or something in an apologetic way, tell the user to ask for something related to the context.
    - If the  `user_query` is an empty string, respond with "Please provide a valid query.".
    - Maintain a tone that is both professional and empathetic, suitable for a consultancy setting.
    
    Return the answer as the only output. 
    Always make sure that you're returning the answer without any explanation. 
    The output should be the answer alone.
    Always return this: "Please provide a valid query." for empty query.
    """
    question_prompt = PromptTemplate(input_variables=["user_query", "doc_list"], template=template)
    initiator_router = question_prompt | llm | StrOutputParser()
    output = initiator_router.invoke({"user_query":user_query, "doc_list":doc_list})
    return output