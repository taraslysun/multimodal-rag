import logging
from .mongo_utils import init_mongo_collections
from .faiss_utils import load_faiss_index
from .embedding import load_sentence_embedding_model, load_clip_model
from .prompt_builder import init_gemini_client
from .multimodal_rag import MultimodalRAG
from .ingest import ingest_article
from config import GEMINI_MODEL_NAME, TOP_K

logger = logging.getLogger(__name__)

def main():
    gemini_client = init_gemini_client()
    faiss_index = load_faiss_index()
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
    import pymongo
    from pymongo import MongoClient
    client = MongoClient("mongodb://localhost:27017/")
    db = client["my_rag_database"]
    collection = db["articles"]
    for article in collection.find():
        article_data = {
            "uuid": article.get("uuid"),
            "title": article.get("title"),
            "url": article.get("url"),
            "html": article.get("html"),
            "feature_image": article.get("feature_image"),
        }
        ingest_article(rag, article_data)

if __name__ == "__main__":
    main()
