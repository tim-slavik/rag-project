import numpy as np




class HybridRetriever:
    def __init__(
        self,
        documents,
        vector_store,
        bm25=None,
        fusion_method="rrf",
        rrf_k=60,
        metadata=None
    ):
        self.documents = documents
        self.vector_store = vector_store
        self.bm25 = bm25
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.metadata = metadata or []


    # -------------------------------------------------
    # FAISS search
    # ------------------------------------------------- 
    def faiss_search(self, query_embeddings, top_k=50):
        scores, ids = self.vector_store.search(query_embeddings, top_k)
        return [(int(doc_id), float(score)) for doc_id, score in zip(ids, scores)]


    # -------------------------------------------------
    # BM25 search
    # ------------------------------------------------- 
    def bm25_search(self, query, top_k=50):
        if self.bm25 is None:
            return []

        scores = self.bm25.get_scores(query.split())
        ranked = np.argsort(scores)[::-1][:top_k]

        return [(int(doc_id), float(scores[doc_id])) for doc_id in ranked]

    # -------------------------------------------------
    # Reciprocal Rank Fusion (RRF)
    # -------  ------------------------------------------
    def rrf_fusion(self, faiss_results, bm25_results):
        fused = {}

        # FAISS contributions
        for rank, (doc_id, _) in enumerate(faiss_results):
            fused.setdefault(doc_id, 0.0)
            fused[doc_id] += 1.0 / (self.rrf_k + rank + 1)

        # BM25 contributions
        for rank, (doc_id, _) in enumerate(bm25_results):
            fused.setdefault(doc_id, 0.0)
            fused[doc_id] += 1.0 / (self.rrf_k + rank + 1)

        # Sort by fused score
        ranked = sorted(fused_items(), key = lambda x: x[1], reverse=True)
        return [(doc_id, score) for doc_id,score in ranked]

    # -------------------------------------------------
    # Main retrieval
    # -------------------------------------------------
    def retrieve(self, query, embed_fn=None, top_K=10):
        # Embed query
        query_embedding = embed_fn(query)

        # Run FAISS
        faiss_results = self.faiss_search(query_embedding, top_k=50)

        # Run BM25
        bm25_results = self.bm25_search(query, top_k=50)

        # Fuse
        fused = self.rrf_fusion(faiss_results, bm25_results)

        # Take top_k results
        fused = fused[:top_k]

        return fused
