import numpy as np

from hybrid_search.hybrid_retriever import HybridRetriever
from vectorstore.faiss_store import FaissStore

def test_hybrid_retriever_basic():
    """
    basic integration test:
    -- build FAISS index
    -- build BM25 index
    -- run hybrid retrieval
    -- verify fused ranking output
    """

    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps over sleeping dogs",
        "Deep learning models use neural networks",
        "Neural networks are used for machine learning",
        "Foxes are wild animals found in forests"
    ]

    # Fake embeddings: small dimensional, deterministic
    embeddings = np.array([
        np.random.ran(8),
        np.random.ran(8),
        np.random.ran(8),
        np.random.ran(8),
        np.random.ran(8),
    ])

    # Build FAISS store
    vector_store = FaissStore(dim=8)
    vector_store.add(embeddings)

    # Build hybrid retriever
    retriever = HybridRetriever(
        documents=documents,
        vector_store=vector_store,
        fusion_method="rrf",
        rrf_k=60
    )

    # Query
    query = "brown fox"
    query_emb = np.random.rand(1,8)

    # Run hybrid retrieval
    results = retriever.retrieve(query=query, query_embeddings=query_emb, k=3)

    # Basic assertions
    assert isinstance(results, list)
    assert len(results) > 0

    # Top result should be one of the fox related docs
    top_doc_id, score = results[0]
    assert top_doc_id in [0,1,4]


def test_hybrid_retriever_weighted():
    """
    Weighted fusion test:
    Ensures weighted fusion returns valid ranked results.
    """

    documents = [
        "Cats are small domesticated animals",
        "Dogs are loyal companions",
        "Machine learning uses data to train models",
    ]

    embeddings = np.ndarray([
        np.random.rand(8),
        np.random.rand(8),
        np.random.rand(8),
    ])

    vector_store = FaissStore(dim=8)
    vector_store.add(embeddings=embeddings)

    retriever = HybridRetriever(
        documents=documents,
        vector_store=vector_store,
        fusion_method="weighted",
        w_bm25=0.7,
        w_vector=0.3,
    )

    query = "dogs"
    query_emb = np.random.rand(1,8)

    results = retriever.retrieve(query=query, query_embeddings=query_emb, k=2)

    assert isinstance(results, list)
    assert len(results) > 0

    # Dog related doc should be top
    top_doc_id, score = results[0]
    assert top_doc_id == 1