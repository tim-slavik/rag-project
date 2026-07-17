import faiss
import numpy as np

class FaissStore:
    def __init__(self, dim, index=None):
        self.dim = dim
        self.index = index if index is not None else faiss.IndexFlatL2(dim)
        self._count = self.index.ntotal

    @property
    def size(self):
        return self._count
    
    def add(self, embeddings: np.ndarray):
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise(ValueError(f"Expected shape (n, {self.dim}), got {embeddings.shape}"))
        
        self.index.add(embeddings.astype("float32"))
        self._count += embeddings.shape[0]

    def search(self, query_embeddings: np.ndarray, k: int = 5):
        if query_embeddings.ndim != 2 or query_embeddings.shape[1] != self.dim:
            raise(ValueError(f"Expected shape (n, {self.dim}), got {query_embeddings.shape}"))

        distances, indicies = self.index.search(query_embeddings.astype("float32"), k)
        return distances, indicies
    
    def save(self, path: str):
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str):
        index = faiss.read_index(path)
        dim = index.d
        store = cls(dim=dim, index=index)
        store._count = index.ntotal
        return store