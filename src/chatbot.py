import os
import subprocess
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
from langchain_core.documents import Document
import pandas as pd
import torch
from loguru import logger
import logging

# ------------------ Place FILE_IDS here immediately after imports ------------------
FILE_IDS = {
    'train.csv': '1J8ne-L_Wwl73JBl8aLFnLJpYtIsc5pxz'
}
# -----------------------------------------------------------------------------------

# Suppress torch warnings
logging.getLogger('torch').setLevel(logging.ERROR)

# Load environment variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '..', '.env'))

# Validate GROQ_API_KEY
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    logger.error("GROQ_API_KEY not set.")
    raise ValueError("GROQ_API_KEY not set.")
os.environ['GROQ_API_KEY'] = groq_api_key

# File download utility using gdown (for train.csv only)
def download_file(file_id, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info(f"Downloading {dest_path} with file ID {file_id}")
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        result = subprocess.run(['gdown', url, '-O', dest_path], check=True, capture_output=True, text=True)
        logger.info(f"Successfully downloaded {dest_path}")
        # Verify file is not HTML
        with open(dest_path, 'rb') as f:
            header = f.read(20).decode('utf-8', errors='ignore')
            if header.startswith('<!DOCTYPE') or header.startswith('<html'):
                logger.error(f"Downloaded file {dest_path} is HTML, not binary data")
                os.remove(dest_path)
                raise ValueError(f"Invalid file content for {dest_path}: got HTML")
        # Check file size
        file_size = os.path.getsize(dest_path)
        if file_size < 1000:
            logger.error(f"Downloaded file {dest_path} is too small ({file_size} bytes)")
            os.remove(dest_path)
            raise ValueError(f"Invalid file size for {dest_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download {file_id}: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Failed to download {file_id}: {e}")
        raise

# Validate file content
def is_valid_file(file_path, expected_ext):
    try:
        with open(file_path, 'rb') as f:
            header = f.read(20).decode('utf-8', errors='ignore')
            if header.startswith('<!DOCTYPE') or header.startswith('<html'):
                logger.error(f"File {file_path} contains HTML content")
                return False
        file_size = os.path.getsize(file_path)
        if file_size < 1000:
            logger.error(f"File {file_path} is too small ({file_size} bytes)")
            return False
        if expected_ext == '.npy':
            try:
                np.load(file_path, allow_pickle=True)
            except Exception as e:
                logger.error(f"Invalid .npy file {file_path}: {e}")
                return False
        elif expected_ext == '.pkl':
            try:
                with open(file_path, 'rb') as f:
                    pickle.load(f)
            except Exception as e:
                logger.error(f"Invalid .pkl file {file_path}: {e}")
                return False
        elif expected_ext == '.bin':
            try:
                faiss.read_index(file_path)
            except Exception as e:
                logger.error(f"Invalid .bin file {file_path}: {e}")
                return False
        return True
    except Exception as e:
        logger.error(f"Error validating file {file_path}: {e}")
        return False

# ---------------------------- FAISS Indexing (Run Once) --------------------------------------
def create_medical_index(max_chunks=None):
    data_path = os.path.join(BASE_DIR, '..', 'Data', 'train.csv')
    if not os.path.exists(data_path):
        logger.info(f"Data file not found locally, attempting to download...")
        try:
            download_file(FILE_IDS['train.csv'], data_path)
        except Exception as e:
            logger.error(f"Failed to download train.csv: {e}")
            raise ValueError("Cannot proceed without train.csv")

    try:
        df = pd.read_csv(data_path, encoding='utf-8', encoding_errors='replace')
    except Exception as e:
        logger.error(f"Failed to read train.csv: {e}")
        raise ValueError("Invalid train.csv file")

    documents = []
    for _, row in df.iterrows():
        input_text = str(row.get('input', '')) if 'input' in row else ''
        output_text = str(row.get('output', '')) if 'output' in row else ''
        text = f"{input_text} {output_text}".strip()
        if text:
            documents.append(Document(page_content=text))

    if not documents:
        logger.error("No valid documents extracted from train.csv")
        raise ValueError("Empty document list")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(documents)
    if max_chunks:
        text_chunks = text_chunks[:max_chunks]
    logger.info(f"Created {len(text_chunks)} chunks")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    embeddings_file = os.path.join(BASE_DIR, 'embeddings.npy')

    # Process embeddings in batches
    batch_size = 500
    chunk_embeddings = []
    for i in range(0, len(text_chunks), batch_size):
        batch_texts = [chunk.page_content for chunk in text_chunks[i:i+batch_size]]
        logger.info(f"Embedding batch {i//batch_size + 1}/{(len(text_chunks)-1)//batch_size + 1} ({len(batch_texts)} chunks)")
        batch_embeddings = model.encode(batch_texts, batch_size=128, show_progress_bar=True)
        chunk_embeddings.append(batch_embeddings)
    chunk_embeddings = np.vstack(chunk_embeddings).astype(np.float32)
    np.save(embeddings_file, chunk_embeddings)
    logger.info(f"Saved {len(chunk_embeddings)} embeddings to {embeddings_file}")

    dimension = 384
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)
    logger.info(f"FAISS index created with {index.ntotal} vectors")

    faiss.write_index(index, os.path.join(BASE_DIR, 'faiss_index.bin'))
    with open(os.path.join(BASE_DIR, 'text_chunks.pkl'), 'wb') as f:
        pickle.dump(text_chunks, f)
    logger.info("FAISS index and text chunks saved")

# ---------------------------- FAISS Setup --------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

index_path = os.path.join(BASE_DIR, 'faiss_index.bin')
chunks_path = os.path.join(BASE_DIR, 'text_chunks.pkl')
embeddings_path = os.path.join(BASE_DIR, 'embeddings.npy')

# Check if existing files are valid
files_valid = all([
    os.path.exists(index_path) and is_valid_file(index_path, '.bin'),
    os.path.exists(chunks_path) and is_valid_file(chunks_path, '.pkl'),
    os.path.exists(embeddings_path) and is_valid_file(embeddings_path, '.npy')
])

# Regenerate if any file is missing or invalid
if not files_valid:
    logger.info("Index files missing or invalid, regenerating...")
    create_medical_index(max_chunks=10000)  # Limit chunks for faster testing

# Load FAISS index and text chunks
try:
    index = faiss.read_index(index_path)
    with open(chunks_path, 'rb') as f:
        text_chunks = pickle.load(f)
    logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
except Exception as e:
    logger.error(f"Error loading FAISS index or text chunks: {e}")
    logger.info("Regenerating index files...")
    create_medical_index(max_chunks=10000)  # Limit chunks for faster testing
    try:
        index = faiss.read_index(index_path)
        with open(chunks_path, 'rb') as f:
            text_chunks = pickle.load(f)
        logger.info(f"Loaded regenerated FAISS index with {index.ntotal} vectors")
    except Exception as e:
        logger.error(f"Failed to load regenerated FAISS index: {e}")
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

# Note: gita_chatbot references undefined gitabot_ai. Commenting out until implemented.
"""
def gita_chatbot(user_query: str) -> str:
    logger.info("Searching for similar docs in the Bhagavad Gita vector DB")
    doc_list = search_db(user_query=user_query)
    logger.info("Calling the gitabot_ai to get answer")
    answer = gitabot_ai(user_query=user_query, doc_list=doc_list)  # gitabot_ai undefined
    logger.info("Returning the final answer")
    return answer
"""
