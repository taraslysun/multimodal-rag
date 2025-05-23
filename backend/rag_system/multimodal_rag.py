import numpy as np
import torch
import requests
import io
import base64
from PIL import Image
from typing import List, Optional, Dict, Set, Any
import concurrent.futures
from functools import lru_cache

from .utils.embedding_utils import normalize_embedding
from .prompt_builder import build_gemini_prompt

from google.genai import types as gemini_types
import logging

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
        max_workers=4,
    ):
        self.gemini_client     = gemini_client
        self.faiss_index       = faiss_index
        self.text_collection   = text_collection
        self.image_collection  = image_collection
        self.text_model        = text_model
        self.clip_model        = clip_model
        self.clip_processor    = clip_processor
        self.gemini_model_name = gemini_model_name
        self.top_k             = top_k
        self.max_workers       = max_workers
        # Add cache for query results
        self.query_cache = {}

    @lru_cache(maxsize=100)  # Cache text embeddings
    def embed_text(self, query: str) -> np.ndarray:
        with torch.no_grad():
            emb = self.text_model.encode(query, convert_to_numpy=True)
        return normalize_embedding(emb)


    def embed_image(self, pil_image: Image.Image) -> np.ndarray:
        """Get image embedding"""
        inputs = self.clip_processor(images=pil_image, return_tensors="pt").to(self.clip_model.device)
        with torch.no_grad():
            img_features = self.clip_model.get_image_features(**inputs)
        img_features = img_features.cpu().numpy().squeeze()
        return normalize_embedding(img_features)

    def combine_embeddings(
        self, text_emb: Optional[np.ndarray], image_emb: Optional[np.ndarray]
        ) -> np.ndarray:
        if text_emb is not None and image_emb is not None:
            combined = text_emb + image_emb
            return normalize_embedding(combined)
        elif text_emb is not None:
            return text_emb
        elif image_emb is not None:
            return image_emb
        else:
            raise ValueError("At least one of text_emb or image_emb must be provided.")

    def retrieve_top_k(self, query_emb: np.ndarray) -> List[int]:
        """
        Returns a list of top_k vector_ids (FAISS indices), ignoring -1.
        """
        query_vec = np.expand_dims(query_emb, axis=0).astype(np.float32)
        distances, indices = self.faiss_index.search(query_vec, self.top_k)
        top_ids = [int(idx) for idx in indices[0] if idx != -1]
        return top_ids

    def fetch_docs_by_vector_ids(self, vector_ids: List[int]) -> tuple[List[dict], List[dict]]:
        """
        Fetch both text_docs and image_docs whose 'vector_id' is in vector_ids.
        Returns two lists:
          - text_docs: list of documents from text_collection
          - image_docs: list of documents from image_collection
        """
        if not vector_ids:
            return [], []

        text_docs = list(self.text_collection.find({"vector_id": {"$in": vector_ids}}))
        image_docs = list(self.image_collection.find({"vector_id": {"$in": vector_ids}}))
        return text_docs, image_docs

    @lru_cache(maxsize=100)  # Cache image downloads to avoid redundancy
    def download_image_bytes(self, url: str) -> Optional[bytes]:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            logger.warning(f"Failed to download image {url}: {e}")
            return None

    def get_image_description(self, image_bytes: bytes) -> str:
        """
        Get a detailed description of an image using Gemini.
        """
        try:
            prompt = "Provide a detailed description of the image."
            prompt += "Do not include any unnecesary text, just the description."
            prompt += "If the image has text, plots or graphs, describe them in detail."
            img_part = gemini_types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg",
            )
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=[prompt, img_part],
            )
            description = (response.text or "").strip()
            logger.info(f"Generated image description: {description[:100]}...")
            return description
        except Exception as e:
            logger.warning(f"Failed to generate image description: {e}")
            return ""

    def _download_images_parallel(self, image_hits_sorted):
        """Parallel download of images to improve performance"""
        image_parts = []
        urls_to_download = [img_doc.get("image_url", "") for img_doc in image_hits_sorted]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {executor.submit(self.download_image_bytes, url): url for url in urls_to_download if url}
            for future in concurrent.futures.as_completed(future_to_url):
                img_bytes = future.result()
                if img_bytes:
                    part = gemini_types.Part.from_bytes(
                        data=img_bytes,
                        mime_type="image/jpeg",
                    )
                    image_parts.append(part)
        
        return image_parts

    def query_and_generate(
        self,
        text_query: Optional[str] = None,
        image_file: Optional[str] = None,  # Now expects base64 string
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if text_query is None and image_file is None:
            return "Error: At least one of text_query or image_file must be provided."
        
        # Create a cache key if caching is desired
        cache_key = None
        if text_query:
            cache_key = text_query
            if chat_history:
                # Add last few messages to cache key
                for msg in chat_history[-3:]:
                    cache_key += msg.get("content", "")[:50]
        
        # Check cache first if we have a cache_key
        if cache_key and cache_key in self.query_cache:
            logger.info(f"Cache hit for query: {text_query[:30]}...")
            return self.query_cache[cache_key]
            
        try:
            # 1. Get image description if image is present
            image_description = ""
            pil_img = None
            image_bytes = None
            if image_file:
                try:
                    image_bytes = base64.b64decode(image_file)
                    image_description = self.get_image_description(image_bytes)
                    try:
                        pil_img = Image.open(torch.ByteTensor(bytearray(image_bytes))).convert("RGB")
                    except Exception:
                        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                except Exception as e:
                    logger.error(f"Error processing image: {e}")

            # 2. Embed text_query if present, now enhanced with image description
            text_emb = None
            enhanced_query = text_query or ""
            if image_description:
                if enhanced_query:
                    enhanced_query = f"{enhanced_query}\n\nInput Image description: {image_description}"
                else:
                    enhanced_query = f"Image description: {image_description}"
            
            if enhanced_query:
                text_emb = self.embed_text(enhanced_query)

            # 3. Embed image if present
            image_emb = None
            if pil_img is not None:
                image_emb = self.embed_image(pil_img)

            # 4. Combine into one query embedding
            query_emb = self.combine_embeddings(text_emb, image_emb)


            # 5. Retrieve top-K vector_ids from FAISS
            top_vector_ids = self.retrieve_top_k(query_emb)
            if not top_vector_ids:
                return "No relevant content found."
    


            text_hits, image_hits = self.fetch_docs_by_vector_ids(top_vector_ids)

            if not text_hits and not image_hits:
                return "No relevant content found."

            # 7. Sort each list by the order in top_vector_ids
            text_hits_sorted = sorted(
                text_hits, key=lambda d: top_vector_ids.index(d["vector_id"])
            )
            image_hits_sorted = sorted(
                image_hits, key=lambda d: top_vector_ids.index(d["vector_id"])
            )

            # print(f"Text hits: {len(text_hits_sorted)}, Image hits: {len(image_hits_sorted)}")
            # for i, hit in enumerate(text_hits_sorted):
            #     print(f"Text hit {i}: {hit['text_chunk'][:50]}...")
            #     print(f"Text hit {i} vector_id: {hit['vector_id']}")
            # for i, hit in enumerate(image_hits_sorted):
            #     print(f"Image hit {i}: {hit['desc_text'][:50]}...")
            #     print(f"Image hit {i} vector_id: {hit['vector_id']}")


            user_query_with_desc = enhanced_query if image_description else (text_query or "")
            prompt_text = build_gemini_prompt(
                text_hits_sorted, 
                image_hits_sorted, 
                user_query_with_desc, 
                chat_history
            )

            # 9. Download actual image bytes for any image hits, wrap as Parts (in parallel)
            image_parts = []
            
            # First add the query image if present
            if image_bytes:
                query_img_part = gemini_types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                )
                image_parts.append(query_img_part)
                
            # Then add retrieved image hits (using parallel downloading)
            retrieved_image_parts = self._download_images_parallel(image_hits_sorted)
            image_parts.extend(retrieved_image_parts)

            # 10. Send everything to Gemini
            contents = [prompt_text] + image_parts
            logger.info("Sending multimodal request to Gemini...")
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=contents,
            )

            answer = response.text or ""
            
            # Store in cache if we have a cache_key
            if cache_key:
                self.query_cache[cache_key] = answer
                
                # Limit cache size
                if len(self.query_cache) > 100:
                    # Remove oldest items (simple approach)
                    keys_to_remove = list(self.query_cache.keys())[:20]
                    for k in keys_to_remove:
                        del self.query_cache[k]
                
            return answer
            
        except Exception as e:
            logger.error(f"Error in query_and_generate: {e}", exc_info=True)
            return f"An error occurred: {str(e)}"
