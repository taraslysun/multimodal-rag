from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from backend.rag_system.config import TEXT_EMB_MODEL, CLIP_MODEL_NAME, DEVICE
import logging

logger = logging.getLogger(__name__)

def load_sentence_embedding_model(model_name: str = TEXT_EMB_MODEL):
    logger.info(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)
    if DEVICE.type == "cuda":
        model = model.to(DEVICE)
    return model

def load_clip_model(model_name: str = CLIP_MODEL_NAME):
    logger.info(f"Loading CLIP model: {model_name}")
    clip_model = CLIPModel.from_pretrained(model_name).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
    return clip_model, clip_processor
