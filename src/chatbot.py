import os
from dotenv import load_dotenv
import faiss
import numpy as np
import pickle
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import torch
from loguru import logger

# Load environment variables
load_dotenv(dotenv_path='D:\\doctor_gpt_final\\.env')

# Validate GROQ_API_KEY
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    logger.error("GROQ_API_KEY not set.")
    raise ValueError("GROQ_API_KEY not set.")
os.environ['GROQ_API_KEY'] = groq_api_key

# ---------------------------- FAISS Indexing (Run Once) --------------------------------------
def create_medical_index():
    data_path = 'D:\\doctor_gpt_final\\Data\\train.csv'
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, encoding='utf-8', encoding_errors='replace')
    documents = []
    for _, row in df.iterrows():
        input_text = str(row.get('input', '')) if 'input' in row else ''
        output_text = str(row.get('output', '')) if 'output' in row else ''
        text = f"{input_text} {output_text}".strip()
        if text:
            documents.append(Document(page_content=text))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(text_chunks)} chunks")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    texts = [chunk.page_content for chunk in text_chunks]
    embeddings_file = 'D:\\doctor_gpt_final\\embeddings.npy'

    if os.path.exists(embeddings_file):
        logger.info("Loading precomputed embeddings...")
        chunk_embeddings = np.load(embeddings_file)
        if chunk_embeddings.shape[0] != len(texts):
            logger.warning(f"Embedding mismatch: {chunk_embeddings.shape[0]} vs {len(texts)}. Re-embedding...")
            chunk_embeddings = model.encode(texts, batch_size=128, show_progress_bar=True)
            chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)
            np.save(embeddings_file, chunk_embeddings)
    else:
        logger.info(f"Embedding {len(texts)} chunks...")
        chunk_embeddings = model.encode(texts, batch_size=128, show_progress_bar=True)
        chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)
        np.save(embeddings_file, chunk_embeddings)
        logger.info(f"Saved {len(chunk_embeddings)} embeddings to {embeddings_file}")

    dimension = 384
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)
    logger.info(f"FAISS index created with {index.ntotal} vectors")

    faiss.write_index(index, 'D:\\doctor_gpt_final\\faiss_index.bin')
    with open('D:\\doctor_gpt_final\\text_chunks.pkl', 'wb') as f:
        pickle.dump(text_chunks, f)
    logger.info("FAISS index and text chunks saved")

class Document:
    def __init__(self, page_content):
        self.page_content = page_content

if not os.path.exists('D:\\doctor_gpt_final\\faiss_index.bin'):
    create_medical_index()

# ---------------------------- FAISS Setup --------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

try:
    index = faiss.read_index('D:\\doctor_gpt_final\\faiss_index.bin')
    with open('D:\\doctor_gpt_final\\text_chunks.pkl', 'rb') as f:
        text_chunks = pickle.load(f)
    logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
except Exception as e:
    logger.error(f"Error loading FAISS index or text chunks: {e}")
    raise

model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def search_db(user_query: str, k: int = 3) -> list:
    try:
        query_embedding = np.array(model.encode([user_query], show_progress_bar=False), dtype=np.float32).reshape(1, -1)
        distances, indices = index.search(query_embedding, k)
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

def doctor_gpt(user_query: str) -> str:
    """
    Wrapper function for doctor_gpt_ai to handle medical queries.

    Args:
        user_query (str): The user's query about a health condition.

    Returns:
        str: Response from doctor_gpt_ai.
    """
    logger.info(f"Processing medical query: {user_query}")
    return doctor_gpt_ai(user_query=user_query)

# Existing gita_chatbot code (preserved)
def gita_chatbot(user_query: str) -> str:
    logger.info("Searching for similar docs in the Bhagavad Gita vector DB")
    doc_list = search_db(user_query=user_query)
    logger.info("Calling the gitabot_ai to get answer")
    answer = gitabot_ai(user_query=user_query, doc_list=doc_list)
    logger.info("Returning the final answer")
    return answer