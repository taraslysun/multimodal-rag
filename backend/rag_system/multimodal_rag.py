# The main MultimodalRAG class will go here
import numpy as np
import torch
import requests
from PIL import Image
from typing import List, Optional, Dict, Set
from .utils import normalize_embedding
import logging
from .prompt_builder import build_gemini_prompt
from google.genai import types as gemini_types
import io

logger = logging.getLogger(__name__)

class MultimodalRAG:
    def __init__(
        self,
        gemini_client,
        faiss_index,
        text_collection,
        image_collection,
        text_model,
        clip_model,
        clip_processor,
        gemini_model_name,
        top_k,
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
        with torch.no_grad():
            emb = self.text_model.encode(query, convert_to_numpy=True)
        return normalize_embedding(emb)

    def embed_image(self, pil_image: Image.Image) -> np.ndarray:
        inputs = self.clip_processor(images=pil_image, return_tensors="pt").to(self.clip_model.device)
        with torch.no_grad():
            img_features = self.clip_model.get_image_features(**inputs)
        img_features = img_features.cpu().numpy().squeeze()
        return normalize_embedding(img_features)

    def combine_embeddings(self, text_emb: Optional[np.ndarray], image_emb: Optional[np.ndarray]) -> np.ndarray:
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
        query_vec = np.expand_dims(query_emb, axis=0).astype(np.float32)
        distances, indices = self.faiss_index.search(query_vec, self.top_k)
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(score)))
        return results

    def fetch_text_chunks(self, vector_ids: List[int]):
        docs = list(self.text_collection.find({"vector_id": {"$in": vector_ids}}))
        return docs

    def fetch_associated_images(self, article_ids: Set):
        docs = list(self.image_collection.find({"article_id": {"$in": list(article_ids)}}))
        return docs

    def download_image_bytes(self, url: str) -> Optional[bytes]:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            logger.warning(f"Failed to download image {url}: {e}")
            return None

    def query_and_generate(
        self,
        text_query: Optional[str] = None,
        image_file: Optional[bytes] = None,
    ) -> str:
        text_emb = None
        if text_query:
            text_emb = self.embed_text(text_query)
        image_emb = None
        if image_file:
            try:
                pil_img = Image.open(torch.ByteTensor(bytearray(image_file))).convert("RGB")
            except Exception:
                pil_img = Image.open(io.BytesIO(image_file)).convert("RGB")
            image_emb = self.embed_image(pil_img)
        query_emb = self.combine_embeddings(text_emb, image_emb)
        top_results = self.retrieve_top_text_chunks(query_emb)
        if not top_results:
            return "No relevant content found."
        vector_ids = [vid for vid, _ in top_results]
        text_docs = self.fetch_text_chunks(vector_ids)
        if not text_docs:
            return "No relevant text content found."
        text_docs_sorted = sorted(text_docs, key=lambda d: vector_ids.index(d["vector_id"]))
        article_ids = set(d["article_id"] for d in text_docs_sorted)
        image_docs = self.fetch_associated_images(article_ids)
        prompt_text = build_gemini_prompt(text_docs_sorted, image_docs, text_query or "")
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
