import os
import faiss
from config import FAISS_INDEX_PATH, FAISS_DIM
import logging

logger = logging.getLogger(__name__)

def load_faiss_index(index_path: str = FAISS_INDEX_PATH):
    """
    Load (or create) a FAISS index.
    """
    if not os.path.exists(index_path):
        logger.info(f"Creating new FAISS index at {index_path}")
        index = faiss.IndexFlatL2(FAISS_DIM)
        faiss.write_index(index, index_path)
    logger.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)
    return index

def save_faiss_index(index, index_path: str = FAISS_INDEX_PATH):
    faiss.write_index(index, index_path)
