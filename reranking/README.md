# Reranking Module

The `reranking/` module provides an optional reranking stage for the RAG system.  
Reranking refines the results produced by hybrid retrieval (FAISS + BM25 + fusion) by applying a more precise relevance scoring model, typically a cross‑encoder.

This module is intentionally lightweight: it preserves the architectural hook for reranking without requiring heavy dependencies or transformer models. A real cross‑encoder can be plugged in later without changing the pipeline.

## Overview

The reranking layer provides:
- A base interface for rerankers (`base_reranker.py`)
- A deterministic placeholder cross‑encoder reranker (`cross_encoder_reranker.py`)
- A clean integration point for the hybrid retriever and pipeline orchestrator

The goal is to maintain modularity and future extensibility while keeping the current system dependency‑free.

## Files

### `base_reranker.py`
Defines the abstract interface for reranking models.

Responsibilities:
- Specify required methods (`score`, `rerank`)
- Provide a consistent API for all reranker implementations
- Allow the pipeline and hybrid retriever to remain backend‑agnostic

This abstraction ensures that future rerankers (transformers, domain‑specific models, etc.) can be added without modifying the pipeline.

### `cross_encoder_reranker.py`
Implements a deterministic placeholder cross‑encoder reranker.

Key characteristics:
- Dependency‑free (no transformers or PyTorch)
- Deterministic scoring for stable behavior
- Lexical‑overlap‑based scoring with optional noise removed or minimized
- Serves as a drop‑in replacement for a real cross‑encoder model

This reranker exists to:
- Demonstrate the reranking architecture
- Allow the pipeline to exercise reranking logic
- Provide a clean hook for future model upgrades

It is not intended to be a production‑quality relevance model.

## How It Fits in the System

Reranking sits between hybrid retrieval and the pipeline orchestrator:
Hybrid Retrieval (FAISS + BM25 + Fusion)
↓
Optional Reranker (Cross Encoder)
↓
Pipeline Orchestrator
↓
RAG Engine
↓
LLM
↓
Final Answer


If reranking is disabled, the pipeline simply uses the fused retrieval results.

## Responsibilities

The reranking module is responsible for:
- Scoring query–chunk pairs
- Reordering retrieved chunks based on relevance
- Providing a clean, modular interface for the pipeline
- Supporting deterministic behavior for evaluation and debugging

## Not Responsible For

The reranking module does NOT:
- Perform FAISS or BM25 search
- Fuse retrieval results
- Build prompts
- Call the LLM
- Manage embeddings or indexes

Those responsibilities belong to other modules.

## Extensibility

This module is designed to support:
- Real transformer‑based cross‑encoders
- Domain‑specific scoring models
- Custom ranking heuristics
- Late‑interaction models (e.g., ColBERT)
- Weighted reranking strategies

Because the pipeline uses the base interface, new rerankers can be added without modifying the orchestrator or hybrid retriever.

## Testing

The previous test file:
hybrid_search/tests/test_reranker.py

was removed because it relied on random values and did not provide meaningful coverage.

Reranking behavior is now validated indirectly through:
- `tests/test_retrieval.py`
- `tests/test_rag_pipeline.py`
- `tests/test_grounding.py`
- The evaluation harness (LLM‑as‑judge scoring)

This ensures reranking integrates cleanly without relying on synthetic or unstable tests.