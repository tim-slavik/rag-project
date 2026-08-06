# API Module

The `api/` module provides the HTTP interface for interacting with the RAG system.  
It exposes a clean, minimal FastAPI server that forwards incoming requests to the pipeline orchestrator and returns grounded, retrieval‑augmented answers.

This module is intentionally lightweight and dependency‑free beyond FastAPI.  
It keeps the API surface simple while allowing the underlying RAG engine to evolve independently.

## Overview

The API layer provides:
- a FastAPI application (`main.py`)
- request/response models (`schemas.py`)
- a single RAG endpoint for querying the system
- optional debug output for development and evaluation

The goal is to make the RAG system easy to call from:
- CLI tools  
- frontend applications  
- notebooks  
- external services  

## Files

### `main.py`
Defines the FastAPI server and the RAG endpoint.

Responsibilities:
- initialize the FastAPI app
- load or initialize the pipeline orchestrator
- expose a `/rag` POST endpoint
- validate incoming requests
- return structured responses containing:
  - the final answer
  - retrieved chunks (optional)
  - metadata (optional)
  - debug information (optional)

The API layer does not perform retrieval or synthesis itself — it delegates all logic to the pipeline.

### `schemas.py`
Defines Pydantic models for request and response validation.

Typical models include:
- `RAGRequest`
  - `query: str`
  - optional flags (debug, top_k, etc.)
- `RAGResponse`
  - `answer: str`
  - `chunks: List[dict]` (optional)
  - `metadata: dict` (optional)
  - `debug: dict` (optional)

These schemas ensure consistent, predictable API behavior.

## How It Fits in the System

The API sits at the top of the RAG stack:
Client → FastAPI → Pipeline Orchestrator → RAG Engine → LLM → Answer


It is the primary entry point for external consumers.

## Responsibilities

The API module is responsible for:
- exposing a clean HTTP interface
- validating input and output
- forwarding queries to the pipeline
- returning structured, grounded responses
- supporting optional debug modes

## Not Responsible For

The API module does NOT:
- perform retrieval
- fuse FAISS/BM25 results
- build prompts
- call the LLM directly
- chunk documents
- embed text
- manage indexes

Those responsibilities belong to other modules.

## Extensibility

This module is designed to support:
- additional endpoints (health checks, batch queries, ingestion triggers)
- authentication or API keys
- streaming responses (SSE or WebSockets)
- frontend integration
- logging and monitoring hooks

Because the API layer is thin, new features can be added without modifying the pipeline.

## Testing

API behavior is validated through:
- `tests/test_api.py`
- integration tests that call the `/rag` endpoint
- regression tests that compare API outputs across versions

These ensure:
- request validation works
- responses are structured correctly
- the pipeline integrates cleanly with FastAPI