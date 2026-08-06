# RAG Module

The `rag/` module contains the core Retrieval-Augmented Generation (RAG) engine responsible for transforming retrieved context into grounded, hallucination-resistant answers. It acts as the central logic layer between the retrieval pipeline and the LLM.

## Overview

This module provides:
- The main RAG engine (`engine.py`)
- The grounded synthesis component (`synthesis.py`)
- Interfaces used by the pipeline orchestrator
- Logic for combining retrieved chunks, metadata, and prompts
- Hallucination prevention through grounding checks

The RAG engine does not perform retrieval itself. Instead, it receives:
- Retrieved chunks
- Metadata
- Query text
- Optional reranking results

It then builds a prompt, sends it to the LLM, and produces a final answer.

## Files

### `engine.py`
Implements the core RAG workflow:
- Accepts a query and retrieved context
- Builds a structured prompt using the prompt builder
- Calls the LLM wrapper
- Applies grounding logic
- Returns the final answer and optional debug information

Key responsibilities:
- Prompt assembly
- Context formatting
- Metadata injection
- Hallucination mitigation
- Output normalization

### `synthesis.py`
Handles the final answer synthesis step:
- Merges retrieved chunks into a coherent context block
- Applies formatting rules
- Ensures the LLM receives grounded evidence
- Supports optional answer post-processing

This module is intentionally lightweight so it can be replaced or extended with:
- chain-of-thought synthesis
- multi-step reasoning
- summarization-based grounding

## How It Fits in the System

The RAG engine sits between retrieval and the LLM:

Hybrid Retriever  
↓  
Reranker  
↓  
RAG Engine  
↓  
LLM  
↓  
Final Answer

It is called by the pipeline orchestrator (`pipeline/orchestrator.py`) and used by the API layer (`api/main.py`).

## Responsibilities

The RAG engine is responsible for:
- Ensuring the LLM only sees relevant, grounded context
- Preventing hallucinations by enforcing evidence-based answers
- Producing consistent, structured outputs
- Supporting debug modes for evaluation and testing
- Integrating cleanly with the evaluation harness

## Not Responsible For

The RAG engine does NOT:
- Perform FAISS or BM25 retrieval
- Handle fusion or reranking
- Manage embeddings or indexes
- Serve API requests
- Run evaluation or metrics

Those responsibilities belong to other modules.

## Extensibility

This module is designed to support:
- Custom prompt templates
- Multiple LLM backends
- Additional synthesis strategies
- Multi-document reasoning
- Metadata-aware answer generation

## Testing

Tests for the RAG engine are located in:
tests/test_rag_engine.py
tests/test_rag_pipeline.py

These ensure:
- prompt construction is correct
- grounding logic works
- synthesis produces stable outputs
- pipeline integration behaves as expected