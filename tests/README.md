# Tests Module

The `tests/` module contains the automated test suite for the RAG system.  
It ensures that ingestion, retrieval, reranking, pipeline orchestration, grounding, and evaluation behave consistently across changes. The test suite is intentionally lightweight and deterministic, suitable for a small‑scale RAG system while still enforcing professional engineering discipline.

## Overview

The test layer provides:
- deterministic unit tests
- integration tests for the full pipeline
- regression checks for retrieval and grounding
- evaluation consistency tests

The goal is to detect regressions early and guarantee that changes to chunking, retrieval, or synthesis do not silently degrade system quality.

## Test Philosophy

The test suite follows these principles:

### **Deterministic**
No random embeddings, random FAISS vectors, or nondeterministic scoring.  
All tests must produce identical results across runs.

### **Integration‑focused**
Because RAG systems are highly interconnected, tests emphasize:
- ingestion → embeddings → FAISS → retrieval → pipeline → LLM
- retrieval correctness
- grounding behavior
- metadata consistency

### **Minimal but meaningful**
Only tests that validate real behavior are included.  
Trivial or synthetic tests (e.g., random FAISS smoke tests) are intentionally removed.

### **Evaluation‑driven**
The evaluation harness provides structured metrics that serve as regression checks.

## Folder Structure

Typical contents include:

### `test_ingestion.py`
Validates:
- chunking behavior
- metadata correctness
- deterministic boundaries
- ingestion pipeline stability

### `test_retrieval.py`
Validates:
- FAISS + BM25 hybrid retrieval
- fusion logic
- metadata lookup
- top‑k consistency

### `test_rag_pipeline.py`
Validates:
- orchestrator behavior
- end‑to‑end RAG flow
- prompt construction
- LLM wrapper integration

### `test_grounding.py`
Validates:
- grounding rules
- evidence usage
- hallucination checks

### `test_evaluation.py`
Validates:
- relevance scoring
- grounding scoring
- evaluation report structure

### `test_api.py`
Validates:
- FastAPI request/response models
- `/rag` endpoint behavior
- integration with the pipeline

## Removed Tests

The following tests were intentionally removed:

### `hybrid_search/tests/test_reranker.py`
Removed because:
- it relied on random scoring
- it did not validate meaningful behavior
- reranking is optional and deterministic in the current system

### `vectorstore/tests/*`
Removed because:
- random FAISS tests do not protect real behavior
- retrieval correctness is validated through integration tests

## How Tests Fit in the System

Tests validate the entire RAG stack:

Chunking  
↓  
Embeddings  
↓  
FAISS Index  
↓  
Hybrid Retrieval  
↓  
Optional Reranking  
↓  
Pipeline Orchestrator  
↓  
RAG Engine  
↓  
LLM Wrapper  
↓  
Evaluation

This ensures that every layer behaves consistently and integrates correctly.

## Running Tests

Tests are designed to run with:

- deterministic chunking
- deterministic retrieval
- deterministic evaluation
- stable FAISS index + embeddings

This guarantees reproducible results across machines and environments.

## Future Improvements

Potential enhancements include:
- adding deterministic chunking tests
- adding ingestion snapshot tests
- adding regression snapshots for retrieval quality
- adding AMAQA‑specific evaluation tests

These can be added as the system grows.