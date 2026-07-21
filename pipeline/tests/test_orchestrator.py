import numpy as np

from pipeline.orchestrator import PipelineOrchestrator
from hybrid_search.hybrid_retriever import HybridRetriever
from vectorstore.faiss_store import FaissStore
from reranking.cross_encoder_reranker import CrossEncoderReranker

def fake_embed_fn(query: str) -> np.ndarray:
    """
    Deterministic fake embedding function for testing.
    """
    rng = np.random.default_rng(abs(hash(query)) %2**32))
    return rng.random(8)

def build_test_retriever():
    """
    Helper to build a small Hybridretiever for orchestrator tests.
    """

    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps over sleeping dogs",
        "Neural networks are used for machine learning",
        "Foxes are wild animals found in forests"
    ]

    embeddings = np.array([
        np.random.rand(8),
        np.random.rand(8),
        np.random.rand(8),
        np.random.rand(8),
    ])

    vector_store = FaissStore(dim=8)
    vector_store.add(embeddings)

    retriever = HybridRetriever(
        documents=documents,
        vector_store=vector_store,
        fusion_method="rrf",
        rrf_k=60,
    )
    return retriever, documents


def test_orchestrator_without_reranker():
    """
    Validate orchestrator behavior when reranking is disabled.
    """

    retriever, documents = build_test_retriever()

    orchestrator = PipelineOrchestrator(
        embed_fn=fake_embed_fn,
        retriever=retriever,
        reanker=None,
        use_reranker=False,
        final_k=2,
    )

    results = orchestrator.run("brown fox")

    assert isinstance(results, list)
    assert len(results) == 2

    # Ensure output structure is correct
    for item in results:
        assert "doc_id" in item
        assert "score" in item
        assert "text" in item
        assert item["text"] in documents


def test_orchestrator_with_reranker():
    """
    Full pipeline test:
    - embedding
    - hybrid retrieval
    - reranking
    - final output formatting
    """

    retriever, documents = build_test_retriever()

    reranker = CrossEncoderReranker(seed=392)

    orchestrator = PipelineOrchestrator(
        embed_fn=fake_embed_fn,
        retriever=retriever,
        reranker=reranker,
        use_reranker=True,
        final_k=2,
    )

    results = orchestrator.run("brown fox")

    assert isinstance(results, list)
    assert len(results) == 2

    # Reranked results should prioritize fox related docs
    top_doc = results[0]["text"]
    assert "fox" in top_doc.lower()

    # Scores should be normalized
    assert 0 <= results[0]["score"] <= 1
    assert 0 <= results[1]["score"] <= 1
