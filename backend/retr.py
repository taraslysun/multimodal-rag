import os
import io
import logging
import faiss
import torch
import requests
import numpy as np

from bs4 import BeautifulSoup
from typing import Dict, List, Optional
from PIL import Image
from pymongo import MongoClient
import pytesseract

from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

from google import genai
from google.genai import types as gemini_types

from config import (
    FAISS_INDEX_PATH,
    MONGO_URI,
    DB_NAME,
    TEXT_COLLECTION_NAME,
    IMAGE_COLLECTION_NAME,
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    TOP_K,
    TEXT_EMB_MODEL,
    CLIP_MODEL_NAME,
    DEVICE,
    
)
from backend.rag_system.utils import normalize_embedding

logger = logging.getLogger(__name__)
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# ----------------------------------------------
# Initialization / Helper Functions
# ----------------------------------------------

# Move these to rag_system/mongo_utils.py and rag_system/faiss_utils.py
def init_mongo_collections(uri: str = MONGO_URI):
    """
    Connect to MongoDB and return (client, text_collection, image_collection).
    """
    client = MongoClient(uri)
    db = client[DB_NAME]
    text_col = db[TEXT_COLLECTION_NAME]
    image_col = db[IMAGE_COLLECTION_NAME]
    text_col.create_index("vector_id")
    image_col.create_index("vector_id")
    return client, text_col, image_col

def load_faiss_index(index_path: str = FAISS_INDEX_PATH):
    """
    Load (or create) a FAISS index.
    """
    if not os.path.exists(index_path):
        # create
        logger.info(f"Creating new FAISS index at {index_path}")
        index = faiss.IndexFlatL2(512)  # 512d for CLIP
        faiss.write_index(index, index_path)
    logger.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)
    return index

# Move model loading to rag_system/embedding.py
def load_sentence_embedding_model(model_name: str = TEXT_EMB_MODEL):
    """
    Load a SentenceTransformer text‐embedding model.
    """
    logger.info(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)
    if DEVICE.type == "cuda":
        model = model.to(DEVICE)
    return model

def load_clip_model(model_name: str = CLIP_MODEL_NAME):
    """
    Load CLIP for image embeddings.
    """
    logger.info(f"Loading CLIP model: {model_name}")
    clip_model = CLIPModel.from_pretrained(model_name).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    return clip_model, clip_processor

def init_gemini_client(api_key: str = GEMINI_API_KEY):
    """
    Configure & return a Google Gemini client.
    """
    logger.info("Configuring Google Gemini client")
    client = genai.Client(api_key=api_key)
    return client

# ----------------------------------------------
# Main RAG Class
# ----------------------------------------------

# Move MultimodalRAG to rag_system/multimodal_rag.py
class MultimodalRAG:
    def __init__(
        self,
        gemini_client,
        faiss_index,
        text_collection,
        image_collection,
        text_model: SentenceTransformer,
        clip_model: CLIPModel,
        clip_processor: CLIPProcessor,
        gemini_model_name: str = GEMINI_MODEL_NAME,
        top_k: int = TOP_K,
    ):
        self.gemini_client = gemini_client
        self.faiss_index = faiss_index
        self.text_collection = text_collection
        self.image_collection = image_collection
        self.text_model = text_model
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.gemini_model_name = gemini_model_name
        self.top_k = top_k

    def embed_text(self, query: str) -> np.ndarray:
        """
        Embed a text query using SentenceTransformer → normalize → return float32 np array.
        """
        with torch.no_grad():
            emb = self.text_model.encode(query, convert_to_numpy=True)
        return normalize_embedding(emb)

    def embed_image(self, pil_image: Image.Image) -> np.ndarray:
        """
        Embed a PIL image using CLIP → normalize → return float32 np array.
        """
        inputs = self.clip_processor(images=pil_image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            img_features = self.clip_model.get_image_features(**inputs)
        img_features = img_features.cpu().numpy().squeeze()
        return normalize_embedding(img_features)

    def combine_embeddings(self, text_emb: Optional[np.ndarray], image_emb: Optional[np.ndarray]) -> np.ndarray:
        """
        Combine text & image embeddings into a single vector.  
        - If both present: elementwise sum → normalize.  
        - If only one present: return that one.  
        """
        if text_emb is not None and image_emb is not None:
            combined = text_emb + image_emb
            return normalize_embedding(combined)
        elif text_emb is not None:
            return text_emb
        elif image_emb is not None:
            return image_emb
        else:
            raise ValueError("At least one of text_emb or image_emb must be provided.")

    def retrieve_top_text_chunks(self, query_emb: np.ndarray):
        """
        Search FAISS for the top_k nearest neighbors → return list of (vector_id, score).
        """
        query_vec = np.expand_dims(query_emb, axis=0).astype(np.float32)
        distances, indices = self.faiss_index.search(query_vec, self.top_k)
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(score)))
        return results

    def fetch_text_chunks(self, vector_ids: List[int]):
        """
        Given FAISS vector_ids, return list of matching text documents from MongoDB.
        """
        docs = list(self.text_collection.find({"vector_id": {"$in": vector_ids}}))
        return docs

    def fetch_associated_images(self, article_ids: set):
        """
        Given a set of article_ids, return all image metadata docs from MongoDB.
        """
        docs = list(self.image_collection.find({"article_id": {"$in": list(article_ids)}}))
        return docs

    def download_image_bytes(self, url: str) -> Optional[bytes]:
        """
        Download an image from `url` and return raw bytes. None if failure.
        """
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            logger.warning(f"Failed to download image {url}: {e}")
            return None

    # Move prompt builder to rag_system/prompt_builder.py
    def build_gemini_prompt(
        self,
        text_chunks: List[dict],
        image_docs: List[dict],
        user_query: str
    ) -> str:
        prompt = (
            "You are a knowledgeable assistant. "
            "Below are excerpts from news articles and their associated images/OCR. "
            "Use them to answer the user’s query.\n\n"
            f"User Query: {user_query}\n\n"
        )

        for idx, chunk_doc in enumerate(text_chunks, start=1):
            txt = chunk_doc.get("text_chunk", "").strip()
            src = chunk_doc.get("source_url", "")
            # get all chunks from the same article
            article_id = chunk_doc.get("article_id", "")
            if article_id:
                article_docs = [doc for doc in text_chunks if doc.get("article_id") == article_id]
                if len(article_docs) > 1:
                    # get the title of the first chunk
                    title = article_docs[0].get("title", "")
                    prompt += f"[Article Title: {title}]\n"
            prompt += f"[Excerpt {idx}]\n{txt}\n(Source: {src})\n\n"

        if image_docs:
            prompt += "Below are images (with OCR captions) related to those excerpts:\n"
            for idx, img_doc in enumerate(image_docs, start=1):
                url = img_doc.get("image_url", "")
                ocr = img_doc.get("ocr_text", "").strip()
                desc = img_doc.get("desc_text", "").strip()
                prompt += f"[Image {idx}]: {url}\n"
                if ocr:
                    prompt += f"OCR: {ocr}\n"
                if desc:
                    prompt += f"Description: {desc}\n"
                prompt += "\n"

        prompt += (
            "Using the above information, answer the user’s query accurately and concisely.  "
            "If referencing an article, always mention its title AND URL in your answer. After the answer, "
            "always list the sources in the format: [Title] (URL) and images in the format: [Image] (URL). "
            "If you cannot find relevant information, say 'I don't know'.\n\n"
        )
        return prompt

    def query_and_generate(
        self,
        text_query: Optional[str] = None,
        image_file: Optional[bytes] = None,
    ) -> str:
        """
        1. Embed `text_query` (if provided) and/or `image_file`.  
        2. Combine embeddings into a single query vector.  
        3. Retrieve top‐K text chunks via FAISS.  
        4. Fetch their associated images from MongoDB.  
        5. Build a multimodal Gemini prompt.  
        6. Call Google Gemini → return raw text answer.  

        - `text_query`: plain‐text string (or None).  
        - `image_file`: raw bytes of a PIL‐readable image (or None).  
        """
        text_emb = None
        if text_query:
            text_emb = self.embed_text(text_query)

        image_emb = None
        if image_file:
            try:
                # Try reading bytes directly  
                pil_img = Image.open(torch.ByteTensor(bytearray(image_file))).convert("RGB")
            except Exception:
                # Fallback: load via PIL from bytes
                pil_img = Image.open(io.BytesIO(image_file)).convert("RGB")
            image_emb = self.embed_image(pil_img)

        # 1. Combine into one query embedding
        query_emb = self.combine_embeddings(text_emb, image_emb)

        # 2. Retrieve nearest text chunks from FAISS
        top_results = self.retrieve_top_text_chunks(query_emb)
        if not top_results:
            return "No relevant content found."

        vector_ids = [vid for vid, _ in top_results]

        text_docs = self.fetch_text_chunks(vector_ids)
        if not text_docs:
            return "No relevant text content found."

        # Sort text docs in the same order as vector_ids
        text_docs_sorted = sorted(text_docs, key=lambda d: vector_ids.index(d["vector_id"]))

        article_ids = set(d["article_id"] for d in text_docs_sorted)
        image_docs = self.fetch_associated_images(article_ids)

        prompt_text = self.build_gemini_prompt(text_docs_sorted, image_docs, text_query or "")
        logger.info("Prompt for Gemini:\n" + prompt_text)

        image_parts = []
        for img_doc in image_docs:
            url = img_doc.get("image_url", "")
            img_bytes = self.download_image_bytes(url)
            if img_bytes:
                part = gemini_types.Part.from_bytes(
                    data=img_bytes,
                    mime_type="image/jpeg",
                )
                image_parts.append(part)

        contents = [prompt_text] + image_parts
        logger.info("Sending multimodal request to Gemini...")
        response = self.gemini_client.models.generate_content(
            model=self.gemini_model_name,
            contents=contents,
        )

        answer = response.text or ""
        return answer

    # ----------------------------------------------
    # New Method: Ingest a Single Article + Its Images
    # ----------------------------------------------
    # Move ingest_article to rag_system/ingest.py
    def ingest_article(self, article: Dict):
        """
        Ingest one article dictionary into MongoDB + FAISS. The `article` dict should have at least:
          - 'uuid'          : a unique article ID (string)
          - 'title'         : article title (string)
          - 'url'           : article URL (string)
          - 'html'          : raw HTML string of the article’s body
          - 'feature_image' : URL of the feature image (string or None)
        """
        article_id = article.get("uuid") or article.get("_id") or str(article.get("slug"))
        article_title = article.get("title", "")
        source_url = article.get("url", "")

        raw_html = article.get("html", "")
        feature_img_url = article.get("feature_image", None)

        # ------------------------------------------
        # 1) Split `raw_html` into ~500‑word text chunks
        # ------------------------------------------
        soup = BeautifulSoup(raw_html, "html.parser")
        # Extract all <p> tags in order, combine them into one big text
        paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
        full_text = "\n".join(paragraphs)

        # Basic 500-word chunking
        words = full_text.split()
        chunk_size = 500
        text_chunks = [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

        # ------------------------------------------
        # 2) Embed each text chunk → normalize → FAISS → MongoDB
        # ------------------------------------------
        for idx, chunk in enumerate(text_chunks):
            if not chunk.strip():
                continue

            with torch.no_grad():
                emb = self.text_model.encode(chunk, convert_to_numpy=True)

            text_emb = normalize_embedding(emb)  # float32 numpy array
            # Add to FAISS
            self.faiss_index.add(np.expand_dims(text_emb, axis=0).astype(np.float32))
            vector_id = self.faiss_index.ntotal - 1

            # Store metadata in MongoDB (text collection)
            text_doc = {
                "article_id": article_id,
                "chunk_id": f"{article_id}_text_{idx}",
                "article_title": article_title,
                "title": article_title,
                "text_chunk": chunk,
                "source_url": source_url,
                "vector_id": int(vector_id),
                "type": "text",
                "scraped_at": article.get("scraped_at"),
                "published_at": article.get("published_at"),
                "url": source_url,
            }
            self.text_collection.insert_one(text_doc)

        # ------------------------------------------
        # 3) Gather all image URLs: feature_image + inline <img> tags
        # ------------------------------------------
        image_urls = set()
        if feature_img_url:
            image_urls.add(feature_img_url)

        # Also pick up every <img src="…"> inside raw_html
        for img_tag in soup.find_all("img"):
            src = img_tag.get("src") or img_tag.get("data-src") or img_tag.get("data-lazy-src")
            if src:
                image_urls.add(src)

        # ------------------------------------------
        # 4) For each image, download → OCR → Gemini description → embeddings
        # ------------------------------------------
        for idx, img_url in enumerate(image_urls):
            img_bytes = self.download_image_bytes(img_url)
            if not img_bytes:
                continue

            # 4.1) Run OCR on image to extract any embedded text
            try:
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                ocr_text = pytesseract.image_to_string(pil_img).strip()
            except Exception as e:
                logger.warning(f"OCR failed on {img_url}: {e}")
                ocr_text = ""

            # 4.2) Use Gemini to create a detailed textual description of the image
            desc_text = ""
            try:
                prompt = "Generate a detailed, descriptive alt‐text for the image."
                img_part = gemini_types.Part.from_bytes(
                    data=img_bytes,
                    mime_type="image/jpeg",
                )
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=[prompt, img_part],
                )
                desc_text = (response.text or "").strip()
            except Exception as e:
                logger.warning(f"Gemini description failed for {img_url}: {e}")
                desc_text = ""

            # 4.3) Embed the raw image, OCR text, and Gemini description separately
            try:
                img_emb = self.embed_image(pil_img)  # 512d → normalized
            except Exception as e:
                logger.warning(f"CLIP embed failed for image {img_url}: {e}")
                img_emb = None

            if ocr_text:
                try:
                    ocr_emb = self.embed_text(ocr_text)
                except Exception as e:
                    logger.warning(f"Text embedding failed for OCR text on {img_url}: {e}")
                    ocr_emb = None
            else:
                ocr_emb = None

            if desc_text:
                try:
                    desc_emb = self.embed_text(desc_text)
                except Exception as e:
                    logger.warning(f"Text embedding failed for description from Gemini of {img_url}: {e}")
                    desc_emb = None
            else:
                desc_emb = None

            # 4.4) Insert each embedding into FAISS & store metadata in MongoDB (image collection)
            #      We create three separate “vector_id” entries, one per embedding type.
            if img_emb is not None:
                self.faiss_index.add(np.expand_dims(img_emb, axis=0).astype(np.float32))
                vid_img = self.faiss_index.ntotal - 1
                image_doc = {
                    "article_id": article_id,
                    "image_id": f"{article_id}_img_{idx}",
                    "article_title": article_title,
                    "image_url": img_url,
                    "ocr_text": ocr_text,
                    "desc_text": desc_text,
                    "source_url": source_url,
                    "vector_id": int(vid_img),
                    "type": "image",
                }
                self.image_collection.insert_one(image_doc)
            if ocr_emb is not None:
                self.faiss_index.add(np.expand_dims(ocr_emb, axis=0).astype(np.float32))
                vid_ocr = self.faiss_index.ntotal - 1
                ocr_doc = {
                    "article_id": article_id,
                    "image_id": f"{article_id}_img_{idx}",
                    "article_title": article_title,
                    "image_url": img_url,
                    "ocr_text": ocr_text,
                    "desc_text": desc_text,
                    "source_url": source_url,
                    "vector_id": int(vid_ocr),
                    "type": "ocr",
                }
                self.image_collection.insert_one(ocr_doc)
            if desc_emb is not None:
                self.faiss_index.add(np.expand_dims(desc_emb, axis=0).astype(np.float32))
                vid_desc = self.faiss_index.ntotal - 1
                desc_doc = {
                    "article_id": article_id,
                    "image_id": f"{article_id}_img_{idx}",
                    "article_title": article_title,
                    "image_url": img_url,
                    "ocr_text": ocr_text,
                    "desc_text": desc_text,
                    "source_url": source_url,
                    "vector_id": int(vid_desc),
                    "type": "gemini_desc",
                }
                self.image_collection.insert_one(desc_doc)

        # ------------------------------------------
        # 5) Save (overwrite) the FAISS index on disk
        # ------------------------------------------
        faiss.write_index(self.faiss_index, FAISS_INDEX_PATH)
        # print details of the article and image
        print(f"Article ID: {article_id}")
        print(f"Title: {article_title}")
        print(f"URL: {source_url}")
        print(f"Number of text chunks: {len(text_chunks)}")
        print(f"Number of images: {len(image_urls)}")
        print(f"image details: {[(img_doc.get('image_url'), img_doc.get('ocr_text'), img_doc.get('desc_text')) for img_doc in self.image_collection.find({'article_id': article_id})]}")
        logger.info(f"Ingestion complete for article {article_id} → saved FAISS index to {FAISS_INDEX_PATH}")

# ----------------------------------------------
# Example Usage
# ----------------------------------------------

# Move main usage to rag_system/main.py
if __name__ == "__main__":
    # 1) Initialize all clients/models
    gemini_client = init_gemini_client()
    faiss_index = load_faiss_index()  # or create a new one if needed
    mongo_client, text_col, image_col = init_mongo_collections()
    text_model = load_sentence_embedding_model()
    clip_model, clip_processor = load_clip_model()

    # 2) Create the RAG object
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

    # 3) Suppose `article_data` is a dictionary (e.g. read from your scraper)
    #    It must contain at least: 'uuid', 'title', 'url', 'html', 'feature_image'
    # article_data = {
    #     "uuid": "123e4567-e89b-12d3-a456-426614174000",
    #     "title": "Sample News Article",
    #     "url": "https://example.com/news/123",
    #     "html": "<div>…raw HTML of the article body…</div>",
    #     "feature_image": "https://example.com/images/feature.jpg",
    #     # plus other keys you already have …
    # }

    import pymongo
    from pymongo import MongoClient
    import json

    client = MongoClient("mongodb://localhost:27017/")
    db = client["my_rag_database"]
    collection = db["articles"]


    for article in collection.find():
        # Convert the article to a dictionary
        article_data = {
            "uuid": article.get("uuid"),
            "title": article.get("title"),
            "url": article.get("url"),
            "html": article.get("html"),
            "feature_image": article.get("feature_image"),
        }

        # 4) Ingest that one article
        rag.ingest_article(article_data)
