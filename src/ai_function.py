import os
from dotenv import load_dotenv
import faiss
import numpy as np
import pickle
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import torch
from loguru import logger

# Load environment variables from .env file
load_dotenv(dotenv_path='d:\\doctor_gpt_final\\.env')

# Validate GROQ_API_KEY
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    logger.error("GROQ_API_KEY not set.")
    raise ValueError("GROQ_API_KEY not set.")

os.environ['GROQ_API_KEY'] = groq_api_key

# ---------------------------- FAISS Setup --------------------------------------
# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

# Load FAISS index and text chunks
try:
    index = faiss.read_index('d:\\doctor_gpt_final\\faiss_index.bin')
    with open('d:\\doctor_gpt_final\\text_chunks.pkl', 'rb') as f:
        text_chunks = pickle.load(f)
    logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
except Exception as e:
    logger.error(f"Error loading FAISS index or text chunks: {e}")
    raise

# Load embeddings model
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def search_db(user_query: str, k: int = 3) -> list:
    try:
        # Embed query
        query_embedding = np.array(model.encode([user_query], show_progress_bar=False), dtype=np.float32).reshape(1, -1)
        # Search FAISS index
        distances, indices = index.search(query_embedding, k)
        # Retrieve matching documents
        sim_docs = [text_chunks[idx].page_content for idx in indices[0] if idx < len(text_chunks)]
        logger.info(f"Retrieved {len(sim_docs)} documents for query: {user_query}")
        return sim_docs
    except Exception as e:
        logger.error(f"Error in search_db: {e}")
        return []

# ---------------------------- LLM --------------------------------------
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

def doctor_gpt_ai(user_query: str) -> str:
    if not user_query:
        return "Please provide a valid health condition query."
    
    logger.info("Searching for relevant documents in the medical vector database")
    doc_list = search_db(user_query=user_query)
    
    template = """
    You are a health consultant AI chatbot, Doctor GPT. Your role is to provide accurate and reliable answers to user questions about health conditions, symptoms, treatments, or any health-related topics typically asked of a doctor, based on the provided documents. Use the information from the `doc_list` to address the `user_query` thoroughly and correctly. Ensure that your response is:

    - **Accurate:** Base your answers solely on the information in the provided documents.
    - **Conversational:** Maintain a friendly and approachable tone.
    - **Mature and Consultancy-Oriented:** Present information in a professional and trustworthy manner.

    **Inputs:**
    1. `user_query`: {user_query}
    2. `doc_list`: {doc_list}

    **Instructions:**
    - Analyze the `user_query` and identify the key information needed to answer it.
    - Review the `doc_list` to find relevant information that addresses the query.
    - Construct a response that is clear, concise, and directly answers the user's question using the information from the documents.
    - Avoid introducing information not present in the `doc_list`.
    - If the `doc_list` has no relevant information, return "I'm sorry, I don't have information on that health topic. Please ask a question related to health conditions or topics covered in my documents."
    - Maintain a professional and empathetic tone.

    Return the answer as the only output.
    """
    question_prompt = PromptTemplate(input_variables=["user_query", "doc_list"], template=template)
    initiator_router = question_prompt | llm | StrOutputParser()
    output = initiator_router.invoke({"user_query": user_query, "doc_list": doc_list})
    logger.info("Returning the final medical answer")
    return output