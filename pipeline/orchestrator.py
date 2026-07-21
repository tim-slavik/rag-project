import numpy as np
from typing import List, Dict, Any, Optional

from hybrid_search.hybrid_retriever import HybridRetriever
from reranking.base_reranker import BaseReranker

class PipelineOrchestrator:
    """
    End to end RAG pipeline orchestrator.

    Responsibilities:
    - embed query
    - run hybrid retrieval
    - optionally rerank results
    - return final ranked chunks with text and scores

    This class is the single entry point for the entire retrieval pipeline.
    """

    def __init__(
            self,
            embed_fn,
            retriever: HybridRetriever,
            reranker: Optional[BaseReranker] = None,
            bm25_top_k: int = 20,
            vector_top_k: int = 20,
            final_k: int = 5,
            use_reranker: bool = True,
    ):
        """
        embed_fn: function(query:str) -> np.ndarray
        retriever: HybridRetriever instance
        reranker: optional reranker implementing BaseReranker
        """

        self.embed_fn = embed_fn
        self.retriever = retriever
        self.reranker = reranker

        self.bm25_top_k = bm25_top_k
        self.vector_top_k = vector_top_k
        self.final_k = final_k
        self.use_reranker = use_reranker

    def run(self,query: str) -> List[Dict[str, Any]]:
        """
        Full pipeline:
        1.  Embed query
        2.  Hybrid retrieval
        3.  Optional Reranking
        4.  Return final ranked chunks
        """

        # Step 1: Embed query
        query_emb = self.embed_fn(query)
        if query_emb.ndim == 1:
            query_emb = np.expand_dims(query_emb, axis=0)
        
        # Step 2: hybrid retrieval
        fused_results = self.retriever.retrieve(
            query=query,
            query_embeddings=query_emb,
            k=max(self.bm25_top_k, self.vector_top_k),
        )

        # Step 3: optional reranking
        if self.use_reranker and self.reranker is not None:
            reranked = self.retriever.retrieve_with_rerank(
                query=query,
                query_embeddings=query_emb,
                k=max(self.bm25_top_k, self.vector_top_k),
                reranker=self.reranker,
                final_k=self.final_k
            )
        else:
            # No reranking - just take the fused results
            reranked = fused_results[: self.final_k]

        # Step 4: format output
        output = []
        for doc_id, score in reranked:
            output.append(
                {
                    "doc_id": doc_id,
                    "score": float(score),
                    "text": self.retriever.documents[doc_id]
                }
            )

        return output


