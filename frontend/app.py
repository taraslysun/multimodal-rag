import re
import streamlit as st
from PIL import Image
import io
import requests
import base64


st.set_page_config(
    page_title="Multimodal RAG Chat",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
# Multimodal RAG Chat Demo

Type a **message** (e.g., “What's the latest news on the AI ethics?”) and/or upload an **image** (e.g., a chart).  
When you click **Send**, our backend queries a joint FAISS index of text & image embeddings, retrieves the most relevant pieces, and calls Google Gemini to generate a multimodal response.

The conversation will appear below in a chat format. Any inline Markdown image syntax in the model’s response will render the image.
"""
)

# Sidebar: instructions and settings
with st.sidebar:
    st.header("Instructions")
    st.markdown(
        """
1. Optionally upload an image.  
2. Type your query in the chat box.  
3. Click **Send** (or press Enter).  
4. The answer appears as chat bubbles below, and any inline Markdown images (`![...]()`) will render automatically.  
"""
    )


BACKEND_URL = "http://multimodal-backend:8000/generate"  # Docker Compose service name

# ---------------------------------------------------------------------
# 1. Initialize session state for chat history
# ---------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []  # Each entry: {"sender": "user"|"assistant", "message": str, "image": bytes or None}

# ---------------------------------------------------------------------
# 2. Render entire chat history (persisted across reruns)
# ---------------------------------------------------------------------

for entry in st.session_state.history:
    sender = entry["sender"]
    msg    = entry["message"]
    img    = entry["image"]

    if sender == "user":
        with st.chat_message("user"):
            if img:
                st.image(img, caption="You uploaded this image")
            st.markdown(msg)
    else:  # "assistant"
        with st.chat_message("assistant"):
            st.markdown(msg)

# ---------------------------------------------------------------------
# 3. Input area: image uploader + chat input
# ---------------------------------------------------------------------

# 3a. Image uploader (give it a key so we can clear it later)
uploaded_file = st.file_uploader(
    "Upload an image (optional)",
    type=["png", "jpg", "jpeg"],
    key="uploaded_file",
    help="If you upload an image, it will be embedded with CLIP and combined with your text query."
)

if uploaded_file:
    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption="Uploaded image preview")
else:
    image_bytes = None

user_message = st.chat_input("Type your message here...")

if user_message:
    st.session_state.history.append({
        "sender": "user",
        "message": user_message,
        "image": image_bytes
    })

    # Display user bubble immediately
    with st.chat_message("user"):
        if image_bytes:
            st.image(image_bytes, caption="You uploaded this image")
        st.markdown(user_message)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")

        chat_history = [
            {"role": h["sender"], "content": h["message"]}
            for h in st.session_state.history if h["sender"] in ("user", "assistant")
        ]
        # Encode image as base64 if present
        image_base64 = base64.b64encode(image_bytes).decode() if image_bytes else None
        payload = {
            "text_query": user_message,
            "image_base64": image_base64,
            "chat_history": chat_history
        }
        try:
            response = requests.post(BACKEND_URL, json=payload, timeout=60)
            response.raise_for_status()
            answer = response.json().get("answer", "(No answer)")
        except Exception as e:
            answer = f"Error: {e}"

        placeholder.markdown(answer)

    # 3c(iii). Append assistant response to history
    st.session_state.history.append({
        "sender": "assistant",
        "message": answer,
        "image": None
    })

    # Optionally clear uploader (Streamlit limitation: can't programmatically clear file_uploader)
