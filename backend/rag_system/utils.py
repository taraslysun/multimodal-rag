import numpy as np

def normalize_embedding(np_emb: np.ndarray) -> np.ndarray:
    """
    L2-normalize a numpy embedding array. If norm=0, returns the original vector.
    """
    norm = np.linalg.norm(np_emb, axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = np_emb / norm
    normalized = np.where(np.isnan(normalized), np_emb, normalized)
    return normalized.astype(np.float32)

