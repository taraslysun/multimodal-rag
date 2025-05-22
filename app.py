# ----------------------------------------------
# frontend/app.py
# ----------------------------------------------

import re
import streamlit as st
from PIL import Image
import io

from backend.rag_system.embedding import load_sentence_embedding_model, load_clip_model
from backend.rag_system.faiss_utils import load_faiss_index
from backend.rag_system.mongo_utils import init_mongo_collections
from backend.rag_system.prompt_builder import init_gemini_client
from backend.rag_system.multimodal_rag import MultimodalRAG
from backend.rag_system.tools.ingest import batch_ingest_all
from backend.rag_system.config import (
    FAISS_INDEX_PATH,
    DB_NAME,
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    TOP_K,
    DEVICE,
)

# ----------------------------------------------
# Streamlit UI
# ----------------------------------------------

st.set_page_config(
    page_title="Multimodal RAG Demo",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("📚 Multimodal RAG (Text + Image) Demo")
st.markdown(
    """
Enter a **text query** (e.g. “Recent AI ethics news”) and/or upload an **image** 
(e.g. a chart or screenshot). When you click **Submit**, 
our backend will retrieve relevant excerpts (from The Batch) and any associated images, 
then send everything to Google Gemini for a multimodal answer.
"""
)

# Sidebar: show version info / quick instructions
with st.sidebar:
    st.header("Instructions")
    st.markdown(
        """
1. Enter a text query (optional if you upload an image).  
2. Upload an image (optional if you typed a query).  
3. Click **Submit** to run retrieval + Gemini.  
4. The answer will appear below, along with any image URLs extracted.  
"""
    )
    st.markdown("---")
    st.write(f"**Device**: {DEVICE}")

# ---------------------------------------------------------------------
# 1. Initialize & cache backend models/index/collections (expensive)
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_backend():
    # 1a. Initialize Gemini client
    gemini_client = init_gemini_client()

    # 1b. Load SentenceTransformer
    text_model = load_sentence_embedding_model()

    # 1c. Load CLIP model & processor
    clip_model, clip_processor = load_clip_model()

    # 1d. Load FAISS index from disk
    faiss_index = load_faiss_index()

    # 1e. Connect to MongoDB
    mongo_client, text_col, image_col = init_mongo_collections()

    # 1f. Instantiate the RAG pipeline with all required parameters
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

    return rag

rag_system = load_backend()

# ---------------------------------------------------------------------
# 2. User Inputs: text_input + file_uploader
# ---------------------------------------------------------------------
with st.form(key="query_form"):
    user_text_query = st.text_input("Enter your text query:", placeholder="e.g. Latest developments in AI ethics…")
    uploaded_file = st.file_uploader(
        "Or upload an image to run an image-based (or multimodal) query:",
        type=["png", "jpg", "jpeg"],
        help="If you upload an image, the pipeline will embed it with CLIP and combine with any text query."
    )
    submit_button = st.form_submit_button(label="Submit")

# ---------------------------------------------------------------------
# 3. On Submit: call rag_system.query_and_generate(...)
# ---------------------------------------------------------------------

if submit_button:
    if not user_text_query and not uploaded_file:
        st.warning("Please enter a text query and/or upload an image.")
    else:
        # If an image is uploaded, read bytes
        image_bytes = None
        if uploaded_file:
            image_bytes = uploaded_file.read()

            # Show a preview in the UI
            st.image(image_bytes, caption="🖼️ Uploaded image")
            
            # Get and display image description
            with st.spinner("Analyzing image..."):
                image_description = rag_system.get_image_description(image_bytes)
                if image_description:
                    st.info(f"**Image description:** {image_description}")

        with st.spinner("Running retrieval + Gemini (this can take 5–10 seconds)…"):
            answer = rag_system.query_and_generate(
                text_query=user_text_query or None,
                image_file=image_bytes,
            )
        
        st.markdown("### 💡 Model Answer:")
        
        # A simple regex to find URLs ending in .jpg/.jpeg/.png/.gif
        url_pattern = re.compile(r'(https?://\S+\.(?:jpg|jpeg|png|gif))', re.IGNORECASE)
        found_urls = url_pattern.findall(answer)

        # incorporate the URLs into the answer text
        for url in found_urls:
            answer = answer.replace(url, f"[Image URL]({url})")
        st.markdown(answer)
