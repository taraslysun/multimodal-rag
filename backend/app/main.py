from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient

import sys
sys.path.append("..")

from rag_system.config import GEMINI_MODEL_NAME, TOP_K, GEMINI_API_KEY
from rag_system.embedding import load_sentence_embedding_model, load_clip_model
from rag_system.utils.faiss_utils import load_faiss_index
from rag_system.utils.mongo_utils import init_mongo_collections
from rag_system.prompt_builder import init_gemini_client
from rag_system.multimodal_rag import MultimodalRAG
from schemas.schemas import GenerateRequest, GenerateResponse, IngestCollectionRequest, IngestSingleRequest
import uvicorn
import os
from rag_system.tools.ingest import process_article_texts, process_article_images, _flush_to_faiss_and_mongo

# Initialize local RAG components
gemini_client = init_gemini_client(GEMINI_API_KEY)
text_model = load_sentence_embedding_model()
clip_model, clip_processor = load_clip_model()
faiss_index = load_faiss_index()
mongo_client, text_col, image_col = init_mongo_collections()

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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    print("Received request:", request.text_query)
    answer = rag.query_and_generate(
        text_query=request.text_query,
        image_file=request.image_base64,
        chat_history=[item.model_dump() for item in (request.chat_history or [])],
    )
    return GenerateResponse(answer=answer)

@app.post("/ingest/collection")
def ingest_collection(request: IngestCollectionRequest):
    """
    Ingest all articles from the specified MongoDB collection (remote) into the local RAG system.
    """
    try:
        # Connect to source MongoDB using provided URL
        src_client = MongoClient(request.source_db_url)
        src_col = src_client[request.source_db_name][request.source_collection_name]
        # Fetch all articles from the source collection
        articles_cursor = src_col.find({})

        from rag_system.tools.ingest import process_article_texts, process_article_images, _flush_to_faiss_and_mongo
        for article in articles_cursor:
            text_embs, text_docs = process_article_texts(rag, article)
            img_embs, img_docs = process_article_images(rag, article)
            _flush_to_faiss_and_mongo(rag, text_embs, text_docs, img_embs, img_docs)

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/article")
def ingest_single_article(request: IngestSingleRequest):
    """
    Ingest a single article by URL (fetch, process, and index it).
    """
    import requests as pyrequests
    try:
        # Fetch article HTML and minimal metadata
        resp = pyrequests.get(request.url, timeout=20)
        resp.raise_for_status()
        html = resp.text
        # Minimal article dict (expand as needed)
        import uuid, datetime
        article = {
            "uuid": str(uuid.uuid4()),
            "url": request.url,
            "html": html,
            "scraped_at": datetime.datetime.now(),
        }
        text_embs, text_docs = process_article_texts(rag, article)
        img_embs, img_docs = process_article_images(rag, article)
        _flush_to_faiss_and_mongo(rag, text_embs, text_docs, img_embs, img_docs)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
