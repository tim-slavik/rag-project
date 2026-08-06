# tests/test_retrieval.py

import numpy as np
import pytest

# Adjust these imports to match your project structure
from hybrid_search.bm25 import BM25
from vectorstore.faiss_store import FaissStore
from hybrid_search.hybrid_retriever import HybridRetriever
from hybrid_search.fusion import rrf_fusion


# ---------- Test data & fixtures ----------

TEST_DOCS = [
    "The quick brown fox jumps over the lazy dog.",
    "Neural networks are used for machine learning.",
    "Foxes are wild animals found in forests.",
    "Dogs are common domestic animals.",
]

@pytest.fixture
def bm25():
    return BM25(TEST_DOCS)


@pytest.fixture
def faiss_store():
    dim = 8
    store = FaissStore(dim=dim)

    # Deterministic embeddings for tests
    embeddings = np.array([
        np.ones(dim),              # doc 0
        np.zeros(dim),             # doc 1
        np.full(dim, 0.5),         # doc 2
        np.linspace(0, 1, dim),    # doc 3
    ])

    store.add(embeddings)
    return store


@pytest.fixture
def hybrid_retriever(bm25, faiss_store):
    return HybridRetriever(
        documents=TEST_DOCS,
        bm25=bm25,
        vector_store=faiss_store,
        fusion_method="rrf",
        rrf_k=60,
    )


# ---------- BM25 tests ----------

def test_bm25_returns_scores(bm25):
    scores = bm25.search("fox")
    assert len(scores) == len(TEST_DOCS)
    # At least one doc should have non-zero score
    assert any(s > 0 for s in scores)


def test_bm25_ranks_relevant_doc_higher(bm25):
    scores = bm25.search("fox")
    # doc 0 and 2 mention fox/foxes; they should rank above unrelated docs
    top_indices = np.argsort(scores)[::-1][:2]
    assert set(top_indices).issubset({0, 2})


# ---------- FAISS tests ----------

def test_faiss_returns_neighbors(faiss_store):
    dim = faiss_store.dim
    query_vec = np.ones(dim)  # closest to doc 0
    indices, distances = faiss_store.search(query_vec, k=2)

    assert len(indices) == 2
    assert indices[0] == 0  # doc 0 should be closest


def test_faiss_distance_ordering(faiss_store):
    dim = faiss_store.dim
    query_vec = np.full(dim, 0.5)
    indices, distances = faiss_store.search(query_vec, k=4)

    # Distances should be non-decreasing
    assert all(
        distances[i] <= distances[i + 1]
        for i in range(len(distances) - 1)
    )


# ---------- Fusion (RRF) tests ----------

def test_rrf_fusion_basic():
    ranks = [1, 5, 10]
    scores = rrf_fusion(ranks)

    assert len(scores) == len(ranks)
    # Better rank (smaller number) should have higher score
    assert scores[0] > scores[1] > scores[2]


def test_rrf_fusion_handles_empty():
    scores = rrf_fusion([])
    assert scores == []


# ---------- Hybrid retriever tests ----------

def test_hybrid_retriever_returns_results(hybrid_retriever):
    dim = hybrid_retriever.vector_store.dim
    query_vec = np.ones(dim)

    results = hybrid_retriever.retrieve("fox", query_vec, k=3)

    assert len(results) > 0
    # results: list of (doc_index, score, metadata)
    doc_indices = [r[0] for r in results]
    assert any(i in {0, 2} for i in doc_indices)


def test_hybrid_retriever_metadata(hybrid_retriever):
    dim = hybrid_retriever.vector_store.dim
    query_vec = np.ones(dim)

    results = hybrid_retriever.retrieve("fox", query_vec, k=3)

    for _, _, meta in results:
        assert "source" in meta
        assert "rank_bm25" in meta
        assert "rank_faiss" in meta


def test_hybrid_retriever_regression(hybrid_retriever):
    """
    Regression test: ensure 'fox' still retrieves at least one of the expected docs.
    """
    dim = hybrid_retriever.vector_store.dim
    query_vec = np.ones(dim)

    results = hybrid_retriever.retrieve("fox", query_vec, k=5)
    doc_indices = {r[0] for r in results}

    expected_docs = {0, 2}
    assert len(doc_indices.intersection(expected_docs)) > 0
