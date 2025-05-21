from google import genai
from config import GEMINI_API_KEY
import logging

logger = logging.getLogger(__name__)

def init_gemini_client(api_key: str = GEMINI_API_KEY):
    logger.info("Configuring Google Gemini client")
    client = genai.Client(api_key=api_key)
    return client

def build_gemini_prompt(text_chunks, image_docs, user_query):
    prompt = (
        "You are a knowledgeable assistant. "
        "Below are excerpts from news articles and their associated images/OCR. "
        "Use them to answer the user’s query.\n\n"
        f"User Query: {user_query}\n\n"
    )
    for idx, chunk_doc in enumerate(text_chunks, start=1):
        txt = chunk_doc.get("text_chunk", "").strip()
        src = chunk_doc.get("source_url", "")
        article_id = chunk_doc.get("article_id", "")
        if article_id:
            article_docs = [doc for doc in text_chunks if doc.get("article_id") == article_id]
            if len(article_docs) > 1:
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
        "If you cannot find relevant information, say 'Unfortunately, I cannot find relevant information.'\n\n"
    )
    return prompt
