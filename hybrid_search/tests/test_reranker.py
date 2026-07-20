import numpy as np

from reranking.base_reranker import BaseReranker
from reranking.cross_encoder_reranker import CrossEncoderReranker
from hybrid_search.hybrid_retriever import HybridRetriever
from vectorstore.faiss_store import FaissStore

def test_base_reranker_interface():
    """
    Ensure BaseReranker enforces the required interface
    """
    class DummyReranker(BaseReranker):
        def score(self, query, document):
            return 0.5
        
    r = DummyReranker()
    scores = r.rerank("test query", ["doc1", "doc2"])

    assert scores == [0.5, 0.5]

def test_cross_encoder_reranker_scoring():
    """
    Validate dterministic scoring behavior of the placeholder cross-encoder
    """

    reranker = CrossEncoderReranker(seed=3126)
    score1 = reranker.score("brown fox", "The quick brown fox jumps")
    score2 = reranker.score("brown fox", "Deep learning models use neural networks")

    # Fox=related foc should score higher
    assert score1 > score2

    # Scores should be normalized to [0,1]
    assert 0 <= score1 <= 1
    assert 0 <= score2 <= 1

    def test_hybrid_retriever_with_rerank():
        """
        Full integration test:
        - hybrid retrieval (BM25 + FAISS + fusion)
        - reranking using CrossEncoderReranker
        """

        documents = [
            "The quick brown fox jumps over the lazy dog",
            "A fast brown fox leaps over sleeping dogs",
            "Neural networks are used for machine learning",
            "Foxes are wild animals found in forests"
        ]

        # Fake embeddings for FAISS
        embeddings = np.array([
            np.random.rand(8),
            np.random.rand(8),
            np.random.rand(8),
            np.random.rand(8),
        ])

        vector_store = FaissStore(dim=8)
        vector_store.add(embeddings)

        retriever = HybridRetriever(
            documents=documents
            vector_store=vector_store,
            fusion_method="rrf",
            rrf_k=60,
        )

        reranker = CrossEncoderReranker(seed=729)

        query = "brown fox"
        query_emb = np.random.rand(1,8)

        results = test_hybrid_retriever_with_rerank(
            query=query,
            query_embeddings=query_emb,
            k=3,
            reranker=reranker,
            final_k=2,
        )

        # Basic structure checks
        assert isinstance(results, list)
        assert len(results) == 2

        # Top reranked result should be fox-related
        top_doc_id, score = results[0]
        assert top_doc_id in [0,1,3]
        assert 0 <= score <= 1

