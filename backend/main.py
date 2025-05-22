# FastAPI app for Multimodal RAG
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import base64
import uvicorn
import os
from rag_system.config import GEMINI_MODEL_NAME, TOP_K, GEMINI_API_KEY

from rag_system.embedding import load_sentence_embedding_model, load_clip_model
from rag_system.utils.faiss_utils import load_faiss_index
from rag_system.utils.mongo_utils import init_mongo_collections
from rag_system.prompt_builder import init_gemini_client
from rag_system.multimodal_rag import MultimodalRAG
from rag_system.config import *

gemini_client = init_gemini_client(GEMINI_API_KEY)
print("Gemini client initialized")
text_model = load_sentence_embedding_model()
print("Text model loaded")
clip_model, clip_processor = load_clip_model()
print("CLIP model loaded")
faiss_index = load_faiss_index()
print("FAISS index loaded")
mongo_client, text_col, image_col = init_mongo_collections()
print("MongoDB collections initialized")

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

# Allow CORS for all origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatHistoryItem(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    text_query: Optional[str] = None
    image_base64: Optional[str] = None  # base64-encoded image
    chat_history: Optional[List[ChatHistoryItem]] = None

class GenerateResponse(BaseModel):
    answer: str

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    print("Received request:", request.text_query)
    answer = rag.query_and_generate(
        text_query=request.text_query,
        image_file=request.image_base64,
        chat_history=[item.model_dump() for item in (request.chat_history or [])],
    )
    return GenerateResponse(answer=answer)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
