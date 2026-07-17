import numpy as np

def to_float32(arr: np.ndarray) -> np.ndarray:
    """
    Ensure embeddings are float32 for FAISS compatibility.
    """
    return arr.astype("float32")


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """
    Ensure embeddings are 2D. If a single vector is passed, reshape it.
    """
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def validate_dim(arr: np.ndarray, dim: int):
    """
    Validate that embeddings match the expected dimensionality.
    """
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"Expected shape (n, {dim}), got {arr.shape}")
