# Multimodal RAG Demo (Text + Image) with Streamlit

## Overview

This repository demonstrates how to build a **Retrieval-Augmented Generation (RAG)** system that:
1. Ingests news articles (text + images) into FAISS + MongoDB (already done separately).
2. Exposes a **backend** module (`backend/rag.py`) that can:
   - Accept **text** and/or **image** queries,
   - Retrieve relevant excerpts (via FAISS + MongoDB),
   - Build a multimodal prompt,
   - Send everything to **Google Gemini** for a final answer.
3. Wraps it all in a **Streamlit UI** (`frontend/app.py`), letting users:
   - Enter a text query,
   - Upload an image,
   - Click “Submit,”
   - See the model’s text answer and any image URLs it included.

## Project Structure
```plaintext
multimodal_rag_app/
├── backend/
│ ├── config.py
│ ├── rag.py
│ └── utils.py
├── frontend/
│ └── app.py
├── requirements.txt
└── README.md
```
- **backend/**: All the retrieval/embedding/Gemini logic.  
- **frontend/**: The Streamlit application.

## API Usage

After building and running the Docker container, the API will be available at `http://localhost:8000`.

### POST /generate

Request JSON:
```
{
  "text_query": "Your question here",         // optional
  "image_base64": "...base64 string...",      // optional
  "chat_history": [                            // optional
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Response JSON:
```
{
  "answer": "...model answer..."
}
```

### Example curl
```
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text_query": "What is RAG?"}'
```

## Docker Build & Run

```
docker build -t multimodal-rag-api .
docker run -p 8000:8000 multimodal-rag-api
```