from pymongo import MongoClient
from config import MONGO_URI, DB_NAME, TEXT_COLLECTION_NAME, IMAGE_COLLECTION_NAME
import logging

logger = logging.getLogger(__name__)

def init_mongo_collections(uri: str = MONGO_URI):
    """
    Connect to MongoDB and return (client, text_collection, image_collection).
    """
    client = MongoClient(uri)
    # check if the connection is successful
    try:
        client.admin.command('ping')
        logger.info("MongoDB connection successful.")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise
    db = client[DB_NAME]
    text_col = db[TEXT_COLLECTION_NAME]
    image_col = db[IMAGE_COLLECTION_NAME]
    text_col.create_index("vector_id")
    image_col.create_index("vector_id")
    return client, text_col, image_col
