from abc import ABC, abstractmethod
from typing import List

class BaseReranker(ABC):
    """
    Abstract base class for rerankers.
    Every reranker must implement:
        - score(query, document)
        - rerank(query, documents)
    """


    @abstractmethod
    def score(self, query: str, document:str) -> float:
        """
        Compute a relevance score for (query, document).
        Higher score = more relevant
        """
        pass

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """
        Score a list of documents and return a list of scores
        aligned with the input order.
        """
        return [self.score(query, doc) for doc in documents]

