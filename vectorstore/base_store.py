from abc import ABC, abstractmethod
import numpy as np


class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.
    Defines the interface used by retrieval and hybrid search modules.
    """

    @abstractmethod
    def add(self, embeddings: np.ndarray):
        pass

    @abstractmethod
    def search(self, query_embeddings: np.ndarray, k: int = 5):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @classmethod
    @abstractmethod
    def load(self, path: str):
        pass

    


                  