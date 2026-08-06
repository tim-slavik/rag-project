import pytest
from fastapi.testclient import TestClient

# Import both API apps
from api.main import app as main_app
from api.server import app as server_app

# Import RAGEngine so we can override it
from rag.engine import RAGEngine


# -------------------------------------------------
# Shared tiny test corpus
# -------------------------------------------------

TEST_DOCS = [
    "Foxes are wild animals found in forests.",
    "Dogs are common domestic animals.",
    "Neural networks are used for machine learning.",
    "Forests contain many wild animals including foxes."
]


# -------------------------------------------------
# Override engines in BOTH API apps
# -------------------------------------------------

@pytest.fixture(autouse=True)
def override_engines(monkeypatch):
    """
    Replace the production AMAQA-loaded RAGEngine with a tiny test engine
    for BOTH api/main.py and api/server.py.
    """

    test_engine = RAGEngine.from_documents(TEST_DOCS)

    # Override engine in api/main.py
    monkeypatch.setattr("api.main.engine", test_engine)

    # Override engine in api/server.py
    monkeypatch.setattr("api.server.engine", test_engine)

    return test_engine


@pytest.fixture
def main_client():
    return TestClient(main_app)


@pytest.fixture
def server_client():
    return TestClient(server_app)


# -------------------------------------------------
# Tests for api/main.py
# -------------------------------------------------

def test_main_health(main_client):
    response = main_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_main_query_endpoint(main_client):
    response = main_client.post("/query", json={"query": "foxes", "k": 5})
    assert response.status_code == 200

    data = response.json()
    assert "query" in data
    assert "answer" in data
    assert "chunks_used" in data
    assert any("fox" in c["text"].lower() for c in data["chunks_used"])


def test_main_search_endpoint(main_client):
    response = main_client.post("/search", json={"query": "machine learning", "k": 5})
    assert response.status_code == 200

    data = response.json()
    assert "results" in data
    assert any("machine learning" in c["text"].lower() for c in data["results"])


# -------------------------------------------------
# Tests for api/server.py
# -------------------------------------------------

def test_server_health(server_client):
    response = server_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_server_rag_endpoint(server_client):
    response = server_client.post("/rag", json={"query": "foxes"})
    assert response.status_code == 200

    data = response.json()
    assert "query" in data