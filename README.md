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