# Pipeline Module

The `pipeline/` module contains the high‑level orchestration layer that coordinates the full Retrieval‑Augmented Generation (RAG) workflow. It acts as the central controller that ties together hybrid retrieval, reranking, metadata handling, and the RAG engine.

This module ensures that all components work together in a clean, predictable, and testable sequence.

## Overview

The pipeline orchestrator is responsible for:
- Executing hybrid retrieval (FAISS + BM25 + fusion)
- Applying optional reranking
- Preparing context for the RAG engine
- Passing structured inputs to the RAG engine
- Returning the final grounded answer
- Supporting debug modes for evaluation and testing

It is the “glue” that binds the system together.

## Files

### `orchestrator.py`
This file implements the main pipeline class that coordinates the full RAG workflow.

Key responsibilities:
- Accept a user query
- Run hybrid retrieval
- Merge and normalize retrieved chunks
- Apply reranking if enabled
- Prepare metadata and context blocks
- Call the RAG engine for synthesis
- Return the final answer and optional debug information

The orchestrator does not perform retrieval or synthesis itself. Instead, it delegates to:
- `hybrid_search/hybrid_retriever.py`
- `reranking/` modules
- `rag/engine.py`

This separation keeps the architecture modular and easy to test.

## How It Fits in the System

The orchestrator sits at the center of the RAG pipeline:

User Query
↓
Hybrid Retriever (FAISS + BM25 + Fusion)
↓
Optional Reranker (Cross Encoder)
↓
Pipeline Orchestrator
↓
RAG Engine (Prompt + LLM)
↓
Final Answer


The API layer (`api/main.py`) calls the orchestrator directly for all RAG endpoints.

## Responsibilities

The pipeline orchestrator is responsible for:
- Coordinating retrieval, reranking, and synthesis
- Ensuring consistent data flow between modules
- Preparing clean, structured context for the RAG engine
- Supporting debug modes for evaluation harness and tests
- Providing a single entry point for the API

## Not Responsible For

The orchestrator does NOT:
- Perform FAISS or BM25 search
- Build prompts
- Call the LLM directly
- Generate embeddings
- Manage indexes or metadata files

Those responsibilities belong to other modules.

## Extensibility

The pipeline is designed to support:
- Additional retrieval backends
- Custom fusion strategies
- Multiple reranking models
- Multi-step reasoning pipelines
- Domain-specific metadata injection

Because the orchestrator is modular, new components can be added without changing the API layer.

## Testing

Tests for the pipeline are located in:
pipeline/tests/test_orchestrator.py
tests/test_rag_pipeline.py

These ensure:
- retrieval and synthesis are correctly integrated
- context is passed cleanly to the RAG engine
- reranking hooks behave as expected
- pipeline outputs remain stable across changes
