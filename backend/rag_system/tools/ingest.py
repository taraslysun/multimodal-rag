import io
import logging
import torch
import numpy as np
import pytesseract

from PIL import Image
from bs4 import BeautifulSoup
from pymongo import MongoClient
import sys
sys.path.append("..")

from ..utils.faiss_utils import save_faiss_index
from ..utils.embedding_utils import normalize_embedding
from google.genai import types
from rag_system.config import (
    MONGO_URI,
    DB_NAME,
    SOURCE_COLLECTION_NAME,
    TEXT_COLLECTION_NAME,
    IMAGE_COLLECTION_NAME,
    FAISS_INDEX_PATH,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Split raw HTML into 500‐word chunks of plaintext
# ─────────────────────────────────────────────────────────────────────────────
def split_text_chunks(html: str, chunk_size: int = 500):
    """
    Given raw HTML, extract all <p> text in order and return a list of
    ~chunk_size‐word plaintext chunks.
    """
    soup = BeautifulSoup(html, "html.parser")
    paras = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
    full_text = "\n".join(paras)
    words = full_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        segment = words[i : i + chunk_size]
        if segment:
            chunks.append(" ".join(segment))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Find all unique image URLs (feature + inline <img> tags)
# ─────────────────────────────────────────────────────────────────────────────
def collect_image_urls(html: str, feature_img: str = None):
    """
    Return a set of all image URLs found in <img src="…"> plus the feature_img if provided.
    """
    urls = set()
    if feature_img:
        urls.add(feature_img)
    soup = BeautifulSoup(html, "html.parser")
    for img_tag in soup.find_all("img"):
        src = img_tag.get("src") or img_tag.get("data-src") or img_tag.get("data-lazy-src")
        if src:
            urls.add(src)
    return urls


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Process one article’s text chunks - return (embeddings, docs)
# ─────────────────────────────────────────────────────────────────────────────
def process_article_texts(rag, article: dict):
    """
    1) Splits article['html'] into 500-word chunks.
    2) Embeds each chunk via rag.text_model - normalized vector.
    3) Builds a corresponding text_doc skeleton for MongoDB (vector_id=None placeholder).
    Returns two lists: [np.array(dtype=float32)], [dict(documents)].
    """
    article_id   = article["uuid"]
    title        = article.get("title", "")
    url          = article.get("url", "")
    scraped_at   = article.get("scraped_at")
    published_at = article.get("published_at")
    html         = article.get("html", "")

    text_embs = []
    text_docs = []
    chunks = split_text_chunks(html, chunk_size=500)
    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        with torch.no_grad():
            emb = rag.text_model.encode(chunk, convert_to_numpy=True)
        emb = normalize_embedding(emb).astype(np.float32)
        text_embs.append(emb)

        text_docs.append({
            "article_id":    article_id,
            "chunk_id":      f"{article_id}_text_{idx}",
            "article_title": title,
            "text_chunk":    chunk,
            "source_url":    url,
            "vector_id":     None,      # to be filled after FAISS add
            "type":          "text",
            "scraped_at":    scraped_at,
            "published_at":  published_at,
        })

    return text_embs, text_docs


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Process one article’s images - return (embeddings, docs)
# Each image yields up to four modalities: raw image, OCR text, Gemini description, alt text
# ─────────────────────────────────────────────────────────────────────────────
def process_article_images(rag, article: dict):
    """
    1) Gathers all image URLs (feature_img + img tags).
    2) Downloads each image, runs OCR, calls Gemini for description.
    3) Embeds up to four modalities: raw image, OCR text, Gemini desc, alt text.
    4) Builds image_doc skeletons (vector_id=None placeholder).
    Returns two lists: [np.array(dtype=float32)], [dict(documents)].
    """
    article_id      = article["uuid"]
    title           = article.get("title", "")
    url             = article.get("url", "")
    feature_img     = article.get("feature_image")
    feature_img_alt = article.get("feature_image_alt", "")

    image_embs = []
    image_docs = []

    image_urls = collect_image_urls(article.get("html", ""), feature_img)

    for idx, img_url in enumerate(image_urls):
        img_bytes = rag.download_image_bytes(img_url)
        if not img_bytes:
            continue

        try:
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to open image {img_url}: {e}")
            continue

        # OCR
        try:
            ocr_text = pytesseract.image_to_string(pil).strip()
        except Exception as e:
            logger.warning(f"OCR failed on {img_url}: {e}")
            ocr_text = ""

        # Gemini description
        desc_text = ""
        try:
            prompt = "Generate a detailed, descriptive alt-text for the image. If it is a table or chart, describe the data and try to make a text table."
            img_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            resp = rag.gemini_client.models.generate_content(
                model=rag.gemini_model_name,
                contents=[prompt, img_part],
            )
            desc_text = (resp.text or "").strip()
        except Exception as e:
            logger.warning(f"Gemini description failed for {img_url}: {e}")
            desc_text = ""

        # Collect up to four embedding modalities: raw image, OCR text, Gemini desc, alt text
        mods = []

        # 1) raw image - CLIP
        try:
            img_emb = rag.embed_image(pil)  # returns normalized float32 np.array
            mods.append(("image", img_emb))
        except Exception as e:
            logger.warning(f"CLIP embed failed for image {img_url}: {e}")

        # 2) OCR text - SentenceTransformer
        if ocr_text:
            try:
                ocr_emb = rag.embed_text(ocr_text)
                mods.append(("ocr", ocr_emb))
            except Exception as e:
                logger.warning(f"Text embed failed for OCR on {img_url}: {e}")

        # 3) Gemini description - SentenceTransformer
        if desc_text:
            try:
                desc_emb = rag.embed_text(desc_text)
                mods.append(("gemini_desc", desc_emb))
            except Exception as e:
                logger.warning(f"Text embed failed for Gemini description on {img_url}: {e}")

        # 4) Alt text - SentenceTransformer
        if feature_img_alt:
            try:
                alt_emb = rag.embed_text(feature_img_alt)
                mods.append(("alt_text", alt_emb))
            except Exception as e:
                logger.warning(f"Text embed failed for alt_text on {img_url}: {e}")

        # Build embeddings + docs for each modality found
        for mtype, emb in mods:
            emb32 = emb.astype(np.float32)
            image_embs.append(emb32)
            image_docs.append({
                "article_id":    article_id,
                "image_id":      f"{article_id}_img_{idx}",
                "image_url":     img_url,
                "ocr_text":      ocr_text,
                "desc_text":     desc_text,
                "alt_text":      feature_img_alt,
                "source_url":    url,
                "vector_id":     None,    # to fill later
                "type":          mtype,
            })

    return image_embs, image_docs


# ─────────────────────────────────────────────────────────────────────────────
# Main: Batch Ingestion with Mini‐Batch Flushing
# ─────────────────────────────────────────────────────────────────────────────
def batch_ingest_all(rag, batch_size: int = 50, mini_batch_size: int = 50):
    """
    1) Reads all articles from SOURCE_COLLECTION_NAME in mini‐batches.
    2) For each article, calls process_article_texts() and process_article_images().
    3) Accumulates embeddings + docs in in-memory lists.
    4) Every `mini_batch_size` total embeddings, flush:
       a) bulk-add to FAISS, assign vector_ids,
       b) bulk-insert into MongoDB,
       c) clear accumulators.
    5) After all articles, do one final flush and save FAISS index.
    """
    client = MongoClient(MONGO_URI)
    src_col = client[DB_NAME][SOURCE_COLLECTION_NAME]

    all_text_embs = []
    all_text_docs = []
    all_img_embs  = []
    all_img_docs  = []

    total_processed = 0

    cursor = src_col.find({}, batch_size=batch_size)
    import time
    start = time.time()
    intra_batch_start = time.time()
    for article in cursor:
        # --- Process texts ---
        t_embs, t_docs = process_article_texts(rag, article)
        all_text_embs.extend(t_embs)
        all_text_docs.extend(t_docs)

        # --- Process images ---
        i_embs, i_docs = process_article_images(rag, article)
        all_img_embs.extend(i_embs)
        all_img_docs.extend(i_docs)

        total_processed += 1

        # If we have reached the mini_batch_size threshold, flush now
        current_count = len(all_text_embs) + len(all_img_embs)
        if current_count >= mini_batch_size:
            _flush_to_faiss_and_mongo(rag, 
                                      all_text_embs, all_text_docs, 
                                      all_img_embs, all_img_docs)
            all_text_embs.clear()
            all_text_docs.clear()
            all_img_embs.clear()
            all_img_docs.clear()
            logger.info(f"Flushed mini‐batch after {total_processed} articles.")
            print(f"Time taken for mini-batch: {time.time() - intra_batch_start:.2f} seconds")
            intra_batch_start = time.time()
            print(f"Total time taken so far: {time.time() - start:.2f} seconds")
            print(f"Total processed: {total_processed} articles\n")

    # Final flush for any remaining
    if all_text_embs or all_img_embs:
        _flush_to_faiss_and_mongo(rag, 
                                  all_text_embs, all_text_docs, 
                                  all_img_embs, all_img_docs)
        logger.info(f"Final flush done after total {total_processed} articles.")

    # Save FAISS index once at the very end
    save_faiss_index(rag.faiss_index, FAISS_INDEX_PATH)
    logger.info(f"Batch ingest complete: processed {total_processed} articles.")


# ─────────────────────────────────────────────────────────────────────────────
# Internal: Flush accumulators into FAISS + MongoDB
# ─────────────────────────────────────────────────────────────────────────────
def _flush_to_faiss_and_mongo(rag,
                              text_embs, text_docs,
                              img_embs, img_docs):
    """
    1) Bulk-add text_embs to FAISS - assign vector_id into text_docs.
    2) Bulk-add img_embs to FAISS - assign vector_id into img_docs.
    3) Bulk-insert text_docs into text_collection, img_docs into image_collection.
    """
    # 1) Text embeddings flush
    if text_embs:
        start = rag.faiss_index.ntotal
        rag.faiss_index.add(np.stack(text_embs, axis=0))
        for i, doc in enumerate(text_docs):
            doc["vector_id"] = start + i

        rag.text_collection.insert_many(text_docs)

    # 2) Image embeddings flush
    if img_embs:
        start = rag.faiss_index.ntotal
        rag.faiss_index.add(np.stack(img_embs, axis=0))
        for i, doc in enumerate(img_docs):
            doc["vector_id"] = start + i

        rag.image_collection.insert_many(img_docs)
    logger.info(f"Flushed {len(text_embs)} text and {len(img_embs)} image embeddings to FAISS and MongoDB.")
    save_faiss_index(rag.faiss_index, FAISS_INDEX_PATH)
