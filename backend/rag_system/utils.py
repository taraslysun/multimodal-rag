import numpy as np

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize vector to unit length
    """
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding

