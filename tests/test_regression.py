import pytest
from rag.engine import RAGEngine

# -------------------------------------------------
# Shared tiny test corpus
# -------------------------------------------------

TEST_DOCS = [
    "Foxes are wild animals found in forests.",
    "Dogs are common domestic animals.",
    "Neural networks are used for machine learning.",
    "Forests contain many wild animals including foxes.",
    "Machine learning models can classify images."
]


@pytest.fixture
def rag_engine():
    """
    Build a small, deterministic RAG engine using from_documents().
    Regression tests rely on stable behavior.
    """
    return RAGEngine.from_documents(TEST_DOCS)


# -------------------------------------------------
# Regression Tests
# -------------------------------------------------

def test_regression_retrieval_consistency(rag_engine):
    """
    Ensure retrieval for a known query returns expected top documents.
    This protects against accidental changes in FAISS, BM25, or RRF.
    """

    result = rag_engine.answer("foxes")
    chunks = result["chunks_used"]

    # Expected docs that should always appear for "foxes"
    expected_substrings = ["fox", "forests"]

    assert any(any(sub in c["text"].lower() for sub in expected_substrings)
               for c in chunks)


def test_regression_machine_learning_query(rag_engine):
    """
    Ensure ML-related queries consistently retrieve ML-related documents.
    """

    result = rag_engine.answer("machine learning")
    chunks = result["chunks_used"]

    assert any("machine learning" in c["text"].lower() for c in chunks)


def test_regression_answer_structure(rag_engine):
    """
    Ensure the RAG engine's output structure never changes.
    This protects API clients and downstream consumers.
    """

    result = rag_engine.answer("foxes")

    assert isinstance(result, dict)
    assert set(result.keys()) == {"query", "answer", "chunks_used"}

    assert isinstance(result["query"], str)
    assert isinstance(result["answer"], str)
    assert isinstance(result["chunks_used"], list)


def test_regression_no_hallucination(rag_engine):
    """
    Ensure the LLM continues to avoid hallucinations for unknown topics.
    """

    result = rag_engine.answer("unicorns")
    answer = result["answer"]

    hallucination_markers = [
        "not mentioned",
        "no information",
        "not in the context",
        "the documents do not say",
        "not found"
    ]

    assert any(marker in answer.lower() for marker in hallucination_markers)


def test_regression_prompt_grounding(rag_engine):
    """
    Ensure the LLM continues grounding answers in retrieved context.
    """

    result = rag_engine.answer("Tell me about forests.")
    answer = result["answer"]
    chunks = result["chunks_used"]

    # Context must contain forest-related text
    assert any("forest" in c["text"].lower() for c in chunks)

    # Answer must reference forests
    assert "forest" in answer.lower()

    # Answer must NOT reference unrelated topics
    assert "machine learning" not in answer.lower()
    assert "dog" not in answer.lower()
