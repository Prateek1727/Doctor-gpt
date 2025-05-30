import os
import pandas as pd
import numpy as np
import faiss
import pickle
from loguru import logger
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define base directory relative to the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'Data')
INDEX_DIR = BASE_DIR

def create_medical_index():
    data_path = os.path.join(DATA_DIR, 'train.csv')
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
    embeddings_file = os.path.join(INDEX_DIR, 'embeddings.npy')

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

    faiss.write_index(index, os.path.join(INDEX_DIR, 'faiss_index.bin'))
    with open(os.path.join(INDEX_DIR, 'text_chunks.pkl'), 'wb') as f:
        pickle.dump(text_chunks, f)
    logger.info("FAISS index and text chunks saved")

# Check and create index if missing
if not os.path.exists(os.path.join(INDEX_DIR, 'faiss_index.bin')):
    create_medical_index()

# Load FAISS index and text chunks
try:
    index = faiss.read_index(os.path.join(INDEX_DIR, 'faiss_index.bin'))
    with open(os.path.join(INDEX_DIR, 'text_chunks.pkl'), 'rb') as f:
        text_chunks = pickle.load(f)
    logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
except Exception as e:
    logger.error(f"Error loading FAISS index or text chunks: {e}")
    raise
