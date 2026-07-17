import numpy as np
from vectorstore.faiss_store import FaissStore


def test_add_and_search():
    dim = 8
    store = FaissStore(dim=dim)

    embeddings = np.random.rand(5, dim).astype("float32")
    store.add(embeddings)

    assert store.size == 5

    query = np.random.rand(1, dim).astype("float32")
    distances, indices = store.search(query, k=2)

    assert distances.shape == (1, 2)
    assert indices.shape == (1, 2)



def test_save_load(tmp_path):
    dim = 8
    store = FaissStore(dim=dim)

    embeddings = np.random.rand(5, dim).astype("float32")
    store.add(embeddings)

    path = tmp_path / "faiss_store"
    store.save(str(path))

    loaded = FaissStore.load(str(path))

    assert loaded.size == 5

    query = np.random.rand(1, dim).astype("float32")
    distances, indices = loaded.search(query, k=2)

    assert distances.shape == (1, 2)
    assert indices.shape == (1, 2)

