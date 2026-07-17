# Vectorstore Module (Section 4)

This module provides the vector storage layer for the RAG pipeline.  
It handles embedding storage, similarity search, and index persistence.

## Files

- **faiss_store.py**  
  FAISS-based vector store implementation using exact L2 search.

- **base_store.py**  
  Abstract interface for all vectorstores. Allows future upgrades
  (Chroma, Milvus, Pinecone) without changing retrieval code.

- **utils.py**  
  Small helpers for validation and array formatting.

- **examples/**
  - `sample_embeddings.npy` — example embedding matrix
  - `sample_index.faiss` — example FAISS index
  - `generate_sample_index.py` — script to generate the example index

- **tests/**
  - `test_faiss_store.py` — basic unit tests for add/search/save/load

## Purpose

The vectorstore provides:
- fast similarity search
- efficient embedding storage
- clean interface for retrieval
- foundation for hybrid search and reranking

This module is designed to be simple, modular, and easy to extend.

## Usage

```python
from 04_vectorstore.faiss_store import FaissStore
import numpy as np

store = FaissStore(dim=384)
store.add(np.random.rand(10, 384).astype("float32"))

query = np.random.rand(1, 384).astype("float32")
distances, indices = store.search(query, k=3)