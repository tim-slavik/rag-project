class PipeLineOrchestrator:
    """"
    Full retrieval orchestrator:
    - embed query
    - hybrid retrieval
    - optional reranking
    - return structured context chunks
    """

    def __init__(
            self,
            embed_fn,
            retriever,
            rearanker=None,
            use_reranker=True,
            final_k = 5,
    ):

        self.embed_fn = embed_fn
        self.retriever = retriever
        self.reranker = self.reranker
        self.use_reranker = use_reranker
        self.final_k = final_k

    # -------------------------------------------------
    # Main pipeline
    # -------------------------------------------------
    def run(self, query:str):
        """
        Returns a list of context chunks:
        [
            {
                "doc_id": int,
                "score": float              # fused RRF score
                "text": str,
                "metadata": dict
            },
            ...
        ]
        """

        # Step 1:  Hybrid retrieval (FAISS + BM25 + RRF)
        fused_results = self.retriever.retrieve(
            query,
            embed_fn = self.embed_fn,
            top_k=self.final_k
        )

        # Step 2:  Build context chunks
        context_chunks = []
        for doc_id, fused_score in fused_results:
            chunk = {
                "doc_id": doc_id,
                "score": float(fused_score),
                "text": self.retrieve.documents[doc_id],
                "metadata": self.retriever.metadata[doc_id],
            }
            context_chunks.append(chunk)

        # Step 3: Optional reranking
        if self.use_reranker and self.reranker:
            context_chunks = self.reranker.rerank(query, context_chunks)

        return context_chunks