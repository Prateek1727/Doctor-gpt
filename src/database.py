import os
from dotenv import load_dotenv
import faiss
import numpy as np
import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import torch

# Load environment variables (optional for FAISS, kept for compatibility)
load_dotenv(dotenv_path='d:\\doctor_gpt_final\\.env')

# ---------------------------- FAISS Setup --------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load FAISS index and text chunks
try:
    index = faiss.read_index('d:\\doctor_gpt_final\\faiss_index.bin')
    with open('d:\\doctor_gpt_final\\text_chunks.pkl', 'rb') as f:
        text_chunks = pickle.load(f)
    print(f"Loaded FAISS index with {index.ntotal} vectors")
except Exception as e:
    print(f"Error loading FAISS index or text chunks: {e}")
    raise

# Load embeddings model
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def search_db(user_query: str) -> list:
    """
    Search the FAISS vector database for documents relevant to a health condition query.

    Args:
        user_query (str): The user's query about a health condition or related topic.

    Returns:
        list: A list of relevant document contents if the query is valid, otherwise an empty list.
    """
    if user_query.strip() != "":
        try:
            # Embed query
            query_embedding = np.array(model.encode([user_query], show_progress_bar=False), dtype=np.float32).reshape(1, -1)
            # Search FAISS index
            distances, indices = index.search(query_embedding, k=3)
            # Retrieve matching documents
            sim_docs = [text_chunks[idx].page_content for idx in indices[0] if idx < len(text_chunks)]
            return sim_docs
        except Exception as e:
            print(f"Error in search_db: {e}")
            return []
    else:
        return []

# Test block
if __name__ == "__main__":
    # Test empty query
    data = search_db(user_query="   ")
    print(f"Empty query result: {data}")

    # Test valid query
    data = search_db(user_query="What are the symptoms of diabetes?")
    print(f"Valid query result: {data}")