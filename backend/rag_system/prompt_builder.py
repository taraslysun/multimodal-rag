from google import genai
from .config import GEMINI_API_KEY
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

def init_gemini_client(api_key: str = GEMINI_API_KEY):
    logger.info("Configuring Google Gemini client")
    client = genai.Client(api_key=api_key)
    return client

def build_gemini_prompt(text_chunks: List[Dict], image_docs: Optional[List[Dict]] = None, user_query: str = "", chat_history: Optional[List[Dict]] = None):
    """
    Builds a prompt for Gemini LLM that includes text chunks, image metadata, and chat history.
    
    Args:
        text_chunks: List of text document dictionaries 
        image_docs: Optional list of image document dictionaries
        user_query: The user's original query
        chat_history: Optional list of previous conversation messages
        
    Returns:
        str: The formatted prompt
    """
    prompt = (
        "You are a knowledgeable assistant in a chat conversation. "
        "Below are excerpts from news articles and their associated images/OCR. "
        "Use them to answer the user's latest query in context of the conversation.\n\n"
    )
    
    # Add conversation history if available
    if chat_history and len(chat_history) > 0:
        prompt += "### Previous Conversation:\n"
        for msg in chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        prompt += "### End of Previous Conversation\n\n"
    
    # Add current query
    prompt += f"User's Current Query: {user_query}\n\n"
    
    # Process text chunks
    prompt += "### Retrieved Information:\n"
    for idx, chunk_doc in enumerate(text_chunks, start=1):
        txt = chunk_doc.get("text_chunk", "")
        src = chunk_doc.get("source_url", "")
        title = chunk_doc.get("title", chunk_doc.get("article_title", ""))
        
        if title:
            prompt += f"[Article Title: {title}]\n"
            
        prompt += f"[Excerpt {idx}]\n{txt}\n(Source: {src})\n\n"
    
    # Process image documents if provided
    if image_docs:
        prompt += "Below are images (with OCR captions) related to those excerpts:\n"
        for idx, img_doc in enumerate(image_docs, start=1):
            url = img_doc.get("image_url", "")
            ocr = img_doc.get("ocr_text", "")
            desc = img_doc.get("desc_text", "")
            alt_text = img_doc.get("alt_text", "")
            
            prompt += f"[Image {idx}]: {url}\n"
            if alt_text:
                prompt += f"Alt-text: {alt_text}\n"
            if ocr:
                prompt += f"OCR: {ocr}\n"
            if desc:
                prompt += f"Description: {desc}\n"
            prompt += "\n"
    
    prompt += (
        "You are a helpful assistant working for Deeeplearning.ai at The Batch."
        "Using the above information and conversation history, answer the user's current query accurately and conversationally. "
        "Maintain a helpful and friendly tone consistent with a chat assistant. "
        "If referencing an article, mention its title AND URL. "
        "If referencing an image, include its URL in markdown format (i.e. ![Image](image_url)). "
        "After the answer, list all sources used. If used images, include article URLs as well. "
        "If you cannot find relevant information, say 'Unfortunately, I don't have enough information to answer that question.'\n\n"
    )
    
    return prompt
