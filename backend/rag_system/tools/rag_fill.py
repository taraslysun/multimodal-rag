from dotenv import load_dotenv
import os
import sys
from rag_system.embedding import load_sentence_embedding_model, load_clip_model
from rag_system.faiss_utils import load_faiss_index
from rag_system.mongo_utils import init_mongo_collections
from rag_system.prompt_builder import init_gemini_client
from rag_system.multimodal_rag import MultimodalRAG
from backend.rag_system.tools.ingest import batch_ingest_all
from backend.rag_system.config import (
    FAISS_INDEX_PATH,
    DB_NAME,
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    TOP_K,
)


# Initialize all models, clients, and collections using the modular rag_system code
load_dotenv()

gemini_client = init_gemini_client(GEMINI_API_KEY)
faiss_index = load_faiss_index(FAISS_INDEX_PATH)
mongo_client, text_col, image_col = init_mongo_collections()
text_model = load_sentence_embedding_model()
clip_model, clip_processor = load_clip_model()

rag = MultimodalRAG(
    gemini_client=gemini_client,
    faiss_index=faiss_index,
    text_collection=text_col,
    image_collection=image_col,
    text_model=text_model,
    clip_model=clip_model,
    clip_processor=clip_processor,
    gemini_model_name=GEMINI_MODEL_NAME,
    top_k=TOP_K,
)

import time

start = time.time()
batch_ingest_all(rag, batch_size=10, mini_batch_size=10)
end = time.time()
print(f"Batch ingestion completed in {end - start:.2f} seconds.")

# Example: Query the filled RAG system
query = "Can AI generated works be copyrighted?"
results = rag.query_and_generate(query)
print("Query:", query)
print("Results:")
print(results)