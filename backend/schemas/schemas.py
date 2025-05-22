from pydantic import BaseModel
from typing import Optional, List

class ChatHistoryItem(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    text_query: Optional[str] = None
    image_base64: Optional[str] = None  # base64-encoded image
    chat_history: Optional[List[ChatHistoryItem]] = None

class GenerateResponse(BaseModel):
    answer: str

class IngestRequest(BaseModel):
    db_name: str
    collection_name: str
    batch_size: int = 50
    mini_batch_size: int = 50

class IngestCollectionRequest(BaseModel):
    source_db_url: str
    source_db_name: str
    source_collection_name: str


class IngestSingleRequest(BaseModel):
    url: str