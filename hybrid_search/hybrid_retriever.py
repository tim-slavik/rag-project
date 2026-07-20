import numpy as np
from typing import Dict, List, Tuple

from hybrid_search.bm_25 import BM25
from hybrid_search.fusion import rrf_fusion, weighted_fusion
from vectorstore.faiss_store import FaissStore
from reranking.base_reranker import BaseReranker


class HybridRetriever:
    """
    Hybrid retriever combining:
    - BM25 lexical search
    - FAISS vector search
    - Fusion ranking (RRF or weighted)
    - Optional reanking layer (cross-encoder or heuristic)

    This class plugs directly into the RAG pipeline and provides
    a unified retrieval interface.
    """

    def __init__ (
            self,
            documents: List[str],
            vector_store: FaissStore,
            fusion_method: str = "rrf",
            rrf_k: int = 60,
            w_bm25: float = 0.5,
            w_vector: float = 0.5,
    ):
        
        self.documents = documents
        self.vector_store = vector_store

        # BM25 index
        self.bm25 = BM25(documents)

        # Fusion settings
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.w_bm25 = w_bm25
        self.w_vector = w_vector


    def _vector_search(self, query_embeddings: np.ndarray, k: int) -> Dict[int, float]:
        """
        Run FAISS vector search and convert distances to scores.
        Lower distance = higher score, so invert.
        """

        distances, indicies = self.vector_store.search(query_embeddings, k)

        scores = {}
        for dist, idx in zip(distances[0], indicies[0]):
            # Convert L2 distance to similarity score
            scores[idx] = 1 / (1 + dist)
        
        return scores
    
    def _bm25_search(self, query: str, k:int) -> Dict[int, float]:
        """
        Run BM25 lexical search
        """
        top = self.bm25.top_k(query, k)
        
        return {doc_id: score for doc_id, score in top}
    
    def _fuse(self, bm25_results: Dict[int, float], vector_results: Dict[int, float]):
        """
        Apply fusion method (RRF or weighted).
        """
        if self.fusion_method == "rrf":
            return rrf_fusion(
                bm25_results,
                vector_results,
                k=self.rrf_k,
            )
        else:
            return weighted_fusion(
                bm25_results,
                vector_results,
                w_bm25=self.w_bm25,
                w_vector=self.w_vector,
            )
    
    def retrieve(self, query: str, query_embeddings: np.ndarray, k: int = 5):
        """
        Unified hybrid retrieval
        -- BM25 lexical search
        -- FAISS vector search
        -- Fusion ranking 
        """

        bm25_results = self._bm25_search(query, k)
        vector_results = self._vector_search(query_embeddings, k)
        fused = self._fuse(bm25_results, vector_results)
        return fused
    
    def retrieve_with_rerank(
            self,
            query: str,
            query_embeddings: np.ndarray,
            k: int,
            reranker: BaseReranker,
            final_k: int = 5,
            ):
        """
        Full retrieval pipeline:
        1.  Hybrid search (BM25 + FAISS + fusion)
        2.  Reranking using any BaseReranker implementation
        3.  Return top-n reranked results

        final_k: number of final reranked chunks to return
        """

        # Step 1: hybrid retrieval
        fused = self.retrieve(query, query_embeddings, k)

        # Extract candidate documents
        candidate_ids = [doc_id for doc_id, _ in fused]
        candidate_docs = [self.documents[i] for i in candidate_ids]

        # Step 2: reranking
        scores = reranker.rerank(query, candidate_docs)

        # Step 3: sort by reranker score
        reranked = sorted(
            zip(candidate_ids, scores),
            key = lambda x: x[1],
            reverse=True,
        )

        return reranked[:final_k]