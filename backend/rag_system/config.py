import torch
import os
from dotenv import load_dotenv
load_dotenv()

FAISS_INDEX_PATH = "data/multimodal_faiss.index"
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "my_rag_database"
TEXT_COLLECTION_NAME = "text_chunks"
SOURCE_COLLECTION_NAME = "articles"
IMAGE_COLLECTION_NAME = "images"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.0-flash"

TOP_K = 5

TEXT_EMB_MODEL = "distiluse-base-multilingual-cased"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

TEXT_CHUNK_SIZE = 500
FAISS_DIM = 512

USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
