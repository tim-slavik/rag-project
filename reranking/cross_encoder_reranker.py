import numpy as np
from typing import Optional
from reranking.base_reranker import BaseReranker

class CrossEncoderReranker(BaseReranker):
    """
    Lightweight placeholder cross-encoder reranker.
    This is NOT a real transformer model — it provides a deterministic,
    dependency-free scoring mechanism so the pipeline can be tested
    and swapped later for a real cross-encoder.

    Real implementations will override `score()` with an actual
    model forward pass.
    """

    def __init__(self, seed: Optional[int] = 58):
        # Deterministic scoring for tests
        self.rng = np.random.default_rng(seed)

    def score(self, query: str, document: str) -> float:
        q_tokens = set(query.lower().split())
        d_tokens = set(document.lower().split())
        overlap = len(q_tokens.intersection(d_tokens))
        return min(1.0, overlap * 0.2)
