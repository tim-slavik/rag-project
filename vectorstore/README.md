# Vectorstore Module

The `vectorstore/` module provides the storage and retrieval layer for dense embeddings used by the RAG system. It wraps FAISS functionality behind a clean, modular interface and ensures that vector search is fast, reliable, and easy to integrate with the rest of the pipeline.

This module is responsible for loading, querying, and managing the FAISS index and associated embedding metadata.

## Overview

The vectorstore layer provides:
- A base interface for vector storage (`base_store.py`)
- A FAISS-backed implementation (`faiss_store.py`)
- Utility functions for loading embeddings and indexes (`utils.py`)
- Example artifacts for testing and demonstration (`examples/`)

It is designed to be modular so additional vectorstore backends (e.g., Milvus, Pinecone, Elasticsearch) can be added later without changing the pipeline.

## Files

### `base_store.py`
Defines the abstract interface for vectorstores.

Responsibilities:
- Define required methods (`search`, `add`, `load`, etc.)
- Provide a consistent API for all vectorstore implementations
- Allow the pipeline to remain backend-agnostic

This abstraction makes the system extensible and testable.

### `faiss_store.py`
Implements the FAISS-based vectorstore.

Key responsibilities:
- Load FAISS index files
- Load embedding matrices
- Run similarity search
- Return ranked document IDs and scores
- Provide metadata lookup for retrieved chunks

This is the primary vectorstore used by the hybrid search module.

### `utils.py`
Utility functions for:
- Loading embeddings from `.npy` files
- Loading FAISS index files
- Validating index/embedding compatibility
- Normalizing vectors if needed

These helpers keep the main FAISS store implementation clean.

### `examples/`
Contains small example FAISS indexes and embedding files for demonstration and testing.

Useful for:
- sanity checks
- development without loading full AMAQA data
- verifying FAISS behavior in isolation

## How It Fits in the System

The vectorstore sits at the foundation of the retrieval pipeline:
Embeddings → FAISS Index → Vectorstore → Hybrid Retriever → Pipeline → RAG Engine → LLM
It provides the dense retrieval signal used by hybrid search.

## Responsibilities

The vectorstore module is responsible for:
- Managing FAISS index files
- Running dense similarity search
- Returning ranked document IDs and scores
- Providing metadata lookup for retrieved chunks
- Ensuring fast, efficient vector retrieval

## Not Responsible For

The vectorstore does NOT:
- Generate embeddings
- Build FAISS indexes
- Perform BM25 search
- Fuse retrieval results
- Call the LLM
- Handle synthesis or prompting

Those responsibilities belong to other modules.

## Extensibility

The vectorstore layer is designed to support:
- Additional FAISS index types
- GPU-backed FAISS
- Alternative vectorstores (Milvus, Pinecone, Qdrant)
- Sharded or distributed indexes
- On-the-fly index updates

Because the pipeline uses the base interface, new backends can be added without modifying the orchestrator.

## Testing

Tests (indirectly) for the vectorstore are located in:
tests/test_retrieval.py
tests/test_rag_pipeline.py
tests/test_grounding.py
tests/test_regression.py