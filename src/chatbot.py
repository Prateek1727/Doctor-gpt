import os
import warnings
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
from langchain_core.documents import Document  # Use built-in Document class
import pandas as pd
import torch
from loguru import logger
import requests
import tempfile

# Suppress torch.classes warning
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Validate GROQ_API_KEY
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    logger.error("GROQ_API_KEY not set.")
    raise ValueError("GROQ_API_KEY not set.")
os.environ['GROQ_API_KEY'] = groq_api_key

# Define paths
INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'index')
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data', 'train.csv')
TRAIN_CSV_URL = "YOUR_GOOGLE_DRIVE_OR_S3_URL_HERE"  # <-- SET THIS TO YOUR ACTUAL PUBLIC CSV URL!
EMBEDDINGS_FILE = os.path.join(INDEX_DIR, 'embeddings.npy')
INDEX_FILE = os.path.join(INDEX_DIR, 'faiss_index.bin')
CHUNKS_FILE = os.path.join(INDEX_DIR, 'text_chunks.pkl')

# Ensure index and data directories exist
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

def download_train_csv():
    if not os.path.exists(DATA_PATH):
        if TRAIN_CSV_URL == "YOUR_GOOGLE_DRIVE_OR_S3_URL_HERE":
            logger.error("TRAIN_CSV_URL is not set. Please set it to your actual file URL.")
            raise ValueError("TRAIN_CSV_URL is not set.")
        logger.info(f"Downloading train.csv from {TRAIN_CSV_URL}")
        try:
            response = requests.get(TRAIN_CSV_URL, stream=True, timeout=60)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name
            os.rename(tmp_file_path, DATA_PATH)
            logger.info(f"Downloaded train.csv to {DATA_PATH}")
            # Check file size after download
            if os.path.getsize(DATA_PATH) < 10:
                logger.error("Downloaded train.csv is too small or empty!")
                raise ValueError("Downloaded train.csv is too small or empty!")
        except Exception as e:
            logger.error(f"Failed to download train.csv: {e}")
            raise

def create_medical_index():
    download_train_csv()
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found: {DATA_PATH}")
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding='utf-8', encoding_errors='replace')
    documents = []
    for _, row in df.iterrows():
        input_text = str(row.get('input', '')) if 'input' in row else ''
        output_text = str(row.get('output', '')) if 'output' in row else ''
        text = f"{input_text} {output_text}".strip()
        if text:
            documents.append(Document(page_content=text, metadata={}))

    if not documents:
        logger.error("No data found in CSV, cannot build index.")
        raise ValueError("No data found in CSV, cannot build index.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(text_chunks)} chunks")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    texts = [chunk.page_content for chunk in text_chunks]

    if os.path.exists(EMBEDDINGS_FILE):
        logger.info("Loading precomputed embeddings...")
        chunk_embeddings = np.load(EMBEDDINGS_FILE)
        if chunk_embeddings.shape[0] != len(texts):
            logger.warning(f"Embedding mismatch: {chunk_embeddings.shape[0]} vs {len(texts)}. Re-embedding...")
            chunk_embeddings = model.encode(texts, batch_size=128, show_progress_bar=True)
            chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)
            np.save(EMBEDDINGS_FILE, chunk_embeddings)
    else:
        logger.info(f"Embedding {len(texts)} chunks...")
        chunk_embeddings = model.encode(texts, batch_size=128, show_progress_bar=True)
        chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)
        np.save(EMBEDDINGS_FILE, chunk_embeddings)
        logger.info(f"Saved {len(chunk_embeddings)} embeddings to {EMBEDDINGS_FILE}")

    dimension = 384
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)
    logger.info(f"FAISS index created with {index.ntotal} vectors")

    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, 'wb') as f:
        pickle.dump(text_chunks, f)
    logger.info("FAISS index and text chunks saved")

# Check and create index if missing
if not os.path.exists(INDEX_FILE):
    create_medical_index()

# ---------------------------- FAISS Setup --------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

try:
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, 'rb') as f:
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
