from google import genai
from backend.rag_system.config import GEMINI_API_KEY
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

def init_gemini_client(api_key: str = GEMINI_API_KEY):
    logger.info("Configuring Google Gemini client")
    client = genai.Client(api_key=api_key)
    return client

def build_gemini_prompt(text_chunks: List[Dict], image_docs: Optional[List[Dict]] = None, user_query: str = ""):
    """
    Builds a prompt for Gemini LLM that includes text chunks and optionally image metadata.
    
    Args:
        text_chunks: List of text document dictionaries 
        image_docs: Optional list of image document dictionaries
        user_query: The user's original query
        
    Returns:
        str: The formatted prompt
    """
    prompt = (
        "You are a knowledgeable assistant. "
        "Below are excerpts from news articles and their associated images/OCR. "
        "Use them to answer the user's query.\n\n"
        f"User Query: {user_query}\n\n"
    )
    
    # Process text chunks
    for idx, chunk_doc in enumerate(text_chunks, start=1):
        txt = chunk_doc.get("text_chunk", "").strip()
        src = chunk_doc.get("source_url", "")
        title = chunk_doc.get("title", chunk_doc.get("article_title", "")).strip()
        
        if title:
            prompt += f"[Article Title: {title}]\n"
            
        prompt += f"[Excerpt {idx}]\n{txt}\n(Source: {src})\n\n"
    
    # Process image documents if provided
    if image_docs:
        prompt += "Below are images (with OCR captions) related to those excerpts:\n"
        for idx, img_doc in enumerate(image_docs, start=1):
            url = img_doc.get("image_url", "")
            ocr = img_doc.get("ocr_text", "").strip()
            desc = img_doc.get("desc_text", "").strip()
            alt_text = img_doc.get("alt_text", "").strip()
            
            prompt += f"[Image {idx}]: {url}\n"
            if alt_text:
                prompt += f"Alt-text: {alt_text}\n"
            if ocr:
                prompt += f"OCR: {ocr}\n"
            if desc:
                prompt += f"Description: {desc}\n"
            prompt += "\n"
    
    prompt += (
        "Using the above information, answer the user's query accurately and concisely. "
        "If referencing an article, always mention its title AND URL in your answer. "
        "If referencing an image, include its URL. "
        "After the answer, list all sources used. "
        "If you cannot find relevant information, say 'Unfortunately, I cannot find relevant information.'\n\n"
    )
    
    return prompt
