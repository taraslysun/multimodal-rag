import io
import logging

import torch
import numpy as np
import pytesseract

from PIL import Image
from bs4 import BeautifulSoup
from pymongo import MongoClient

from .faiss_utils import save_faiss_index, create_new_faiss_index
from utils import normalize_embedding
from config import (
    MONGO_URI,
    DB_NAME,
    TEXT_COLLECTION_NAME,      # e.g. "text_chunks"
    IMAGE_COLLECTION_NAME,     # e.g. "image_embeddings"
    FAISS_INDEX_PATH,
    FAISS_DIM,                 # dimension of your embeddings (e.g. 512 if using CLIP+CLIP-text)
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Recreate or clear FAISS index
# ─────────────────────────────────────────────────────────────────────────────
def init_faiss_index(dim: int = 512):
    """
    If an index already exists on disk at FAISS_INDEX_PATH, delete it and create a new empty IndexFlatIP.
    Otherwise, create a new one. Returns the new FAISS index object.
    """
    # Optionally, you can remove the old file to start fresh:
    try:
        import os
        if os.path.exists(FAISS_INDEX_PATH):
            os.remove(FAISS_INDEX_PATH)
            logger.info(f"Deleted existing FAISS index at {FAISS_INDEX_PATH}")
    except Exception as e:
        logger.warning(f"Could not delete old FAISS index: {e}")

    # Create a new raw IndexFlatIP
    index = create_new_faiss_index(dim)
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Fetch documents in paginated fashion
# ─────────────────────────────────────────────────────────────────────────────
def get_mongo_cursor(collection, batch_size: int):
    """
    Return a PyMongo cursor over the entire collection,
    with a specified batch_size for server‐side pagination.
    """
    return collection.find({}, batch_size=batch_size)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Embed a batch of text chunks
# ─────────────────────────────────────────────────────────────────────────────
def embed_text_chunks(rag, docs: list) -> np.ndarray:
    """
    Given a list of text_docs (each has "text_chunk"), embed all with SentenceTransformer in one batch.
    Returns a (N, dim) numpy array of normalized float32 embeddings.
    """
    chunks = [d["text_chunk"] for d in docs]
    with torch.no_grad():
        embs = rag.text_model.encode(chunks, convert_to_numpy=True)
    # Normalize each vector
    embs = np.vstack([normalize_embedding(e) for e in embs]).astype(np.float32)
    return embs


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Embed a batch of image modalities
# ─────────────────────────────────────────────────────────────────────────────
def embed_image_docs(rag, docs: list) -> np.ndarray:
    """
    Given a list of image_docs, each with:
      - type == "image": use rag.embed_image on the downloaded image
      - type in ("ocr", "gemini_desc", "alt_text"): embed the corresponding text field
    Returns a (N, dim) numpy array of normalized float32 embeddings.
    """
    embs = []
    for doc in docs:
        mtype = doc.get("type")
        if mtype == "image":
            url = doc.get("image_url", "")
            img_bytes = rag.download_image_bytes(url)
            if not img_bytes:
                # fallback zero‐vector (will be near zero)
                dim = FAISS_DIM
                embs.append(np.zeros(dim, dtype=np.float32))
                continue
            try:
                pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                emb = rag.embed_image(pil)
                embs.append(emb.astype(np.float32))
            except Exception as e:
                logger.warning(f"Failed CLIP embed for {url}: {e}")
                dim = FAISS_DIM
                embs.append(np.zeros(dim, dtype=np.float32))
        else:
            # “ocr”, “gemini_desc”, or “alt_text”
            text = ""
            if mtype == "ocr":
                text = doc.get("ocr_text", "")
            elif mtype == "gemini_desc":
                text = doc.get("desc_text", "")
            elif mtype == "alt_text":
                text = doc.get("alt_text", "")
            try:
                emb = rag.embed_text(text)
                embs.append(emb.astype(np.float32))
            except Exception as e:
                logger.warning(f"Failed text embed ({mtype}) for image_id {doc.get('image_id')}: {e}")
                dim = FAISS_DIM
                embs.append(np.zeros(dim, dtype=np.float32))

    return np.vstack(embs)


# ─────────────────────────────────────────────────────────────────────────────
# Internal: Flush accumulators into FAISS & MongoDB
# ─────────────────────────────────────────────────────────────────────────────
def _flush_to_faiss_and_mongo(rag, 
                              text_embs: np.ndarray, text_docs: list,
                              img_embs: np.ndarray, img_docs: list):
    """
    1) Bulk-add text_embs (shape: [N_text, dim]) to FAISS, assign vector_id to text_docs.
    2) Bulk-add img_embs  (shape: [N_img,  dim]) to FAISS, assign vector_id to img_docs.
    3) Bulk-update MongoDB docs: set new vector_id for each.
    """
    # 1) Flush text embeddings
    if len(text_embs) > 0:
        start = rag.faiss_index.ntotal
        rag.faiss_index.add(text_embs)
        # Now assign vector_id back to text_docs
        for i, doc in enumerate(text_docs):
            vid = start + i
            doc["vector_id"] = int(vid)

        # Bulk‐write updated vector_id fields into MongoDB
        bulk_ops = []
        for doc in text_docs:
            _id = doc["_id"]
            new_vid = doc["vector_id"]
            bulk_ops.append(
                {
                    "update_one": {
                        "filter": {"_id": _id},
                        "update": {"$set": {"vector_id": new_vid}}
                    }
                }
            )
        if bulk_ops:
            rag.text_collection.bulk_write(bulk_ops)

    # 2) Flush image embeddings
    if len(img_embs) > 0:
        start = rag.faiss_index.ntotal
        rag.faiss_index.add(img_embs)
        for i, doc in enumerate(img_docs):
            vid = start + i
            doc["vector_id"] = int(vid)

        bulk_ops = []
        for doc in img_docs:
            _id = doc["_id"]
            new_vid = doc["vector_id"]
            bulk_ops.append(
                {
                    "update_one": {
                        "filter": {"_id": _id},
                        "update": {"$set": {"vector_id": new_vid}}
                    }
                }
            )
        if bulk_ops:
            rag.image_collection.bulk_write(bulk_ops)


# ─────────────────────────────────────────────────────────────────────────────
# Main: Reindex All Text & Image Docs (with mini‐batch flushing)
# ─────────────────────────────────────────────────────────────────────────────
def reindex_all(rag, batch_size: int = 100, mini_batch_size: int = 500):
    """
    1) Initializes a new empty FAISS index.
    2) Reads text_chunks and image_embeddings collections in batches (batch_size).
    3) Accumulates embeddings + docs in lists.
    4) Every time (len(text_embs) + len(img_embs)) >= mini_batch_size, calls _flush_to_faiss_and_mongo.
    5) After all documents, does one final flush and saves FAISS index.
    """
    # 1) Initialize new FAISS index
    rag.faiss_index = init_faiss_index(dim=FAISS_DIM)

    # 2) Connect to Mongo collections
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    text_col = db[TEXT_COLLECTION_NAME]
    img_col  = db[IMAGE_COLLECTION_NAME]

    # Initialize accumulators
    all_text_docs = []
    all_text_embs = []
    all_img_docs  = []
    all_img_embs  = []

    # 3) Iterate text_docs in pages
    text_cursor = get_mongo_cursor(text_col, batch_size=batch_size)
    for text_doc in text_cursor:
        all_text_docs.append(text_doc)
        # We'll embed in mini‐batches below

        # When we reach mini_batch size, embed & flush
        if len(all_text_docs) >= mini_batch_size:
            # Embed this mini‐batch of text  
            text_emb_batch = embed_text_chunks(rag, all_text_docs)
            # Clear the accumulator after copying to local
            tdocs = all_text_docs.copy()
            all_text_docs.clear()

            # No images yet, so img_embs and img_docs empty for this flush
            _flush_to_faiss_and_mongo(rag, text_emb_batch, tdocs, np.zeros((0, FAISS_DIM), dtype=np.float32), [])

    # After text cursor, if any remain
    if all_text_docs:
        text_emb_batch = embed_text_chunks(rag, all_text_docs)
        tdocs = all_text_docs.copy()
        all_text_docs.clear()
        _flush_to_faiss_and_mongo(rag, text_emb_batch, tdocs, np.zeros((0, FAISS_DIM), dtype=np.float32), [])

    # 4) Iterate image_docs in pages
    img_cursor = get_mongo_cursor(img_col, batch_size=batch_size)
    for img_doc in img_cursor:
        all_img_docs.append(img_doc)

        if len(all_img_docs) >= mini_batch_size:
            # Embed this mini-batch of images
            img_emb_batch = embed_image_docs(rag, all_img_docs)
            idocs = all_img_docs.copy()
            all_img_docs.clear()

            # No text in this flush; pass empty arrays for text
            _flush_to_faiss_and_mongo(rag, np.zeros((0, FAISS_DIM), dtype=np.float32), [],
                                      img_emb_batch, idocs)

    # Final flush for any remaining images
    if all_img_docs:
        img_emb_batch = embed_image_docs(rag, all_img_docs)
        idocs = all_img_docs.copy()
        all_img_docs.clear()
        _flush_to_faiss_and_mongo(rag, np.zeros((0, FAISS_DIM), dtype=np.float32), [],
                                  img_emb_batch, idocs)

    # 5) Save the rebuilt FAISS index once at the end
    save_faiss_index(rag.faiss_index, FAISS_INDEX_PATH)
    logger.info("Reindexing complete. FAISS index saved to %s", FAISS_INDEX_PATH)
