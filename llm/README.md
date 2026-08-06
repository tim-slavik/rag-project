# LLM Module

The `llm/` module provides the language model interface and prompt‑building utilities used by the RAG system. It acts as the final stage of the pipeline, transforming structured context and user queries into grounded, hallucination‑resistant answers.

This module abstracts away model details so the rest of the system can remain backend‑agnostic.

## Overview

The LLM layer provides:
- A unified wrapper for calling the language model (`llm_wrapper.py`)
- A structured prompt builder for grounded RAG prompts (`prompt_builder.py`)
- Optional hooks for debugging, logging, and evaluation
- A clean interface for the pipeline and RAG engine

The goal is to ensure consistent, predictable LLM behavior regardless of the underlying model.

## Files

### `llm_wrapper.py`
Implements the main interface for interacting with the language model.

Responsibilities:
- Accept a prompt and optional parameters
- Call the underlying LLM backend
- Normalize and clean the output
- Provide a consistent API for the RAG engine
- Support debug modes for evaluation and testing

The wrapper is intentionally simple so it can be replaced with:
- OpenAI models
- Azure OpenAI
- Local models (e.g., llama.cpp, vLLM)
- Custom inference servers

### `prompt_builder.py`
Constructs structured prompts for grounded RAG synthesis.

Key responsibilities:
- Format retrieved context into a clean evidence block
- Insert metadata when available
- Build the final instruction + context + query prompt
- Enforce grounding rules to reduce hallucinations
- Support optional debug formatting

The prompt builder ensures that the LLM receives:
- relevant context
- consistent structure
- clear instructions
- no noisy or irrelevant text

## How It Fits in the System

The LLM module sits at the end of the RAG pipeline:

Hybrid Retrieval  
↓  
Reranking  
↓  
RAG Engine  
↓  
Prompt Builder  
↓  
LLM Wrapper  
↓  
Final Answer

It is called by:
- `rag/engine.py`
- `pipeline/orchestrator.py`
- the evaluation harness (for LLM-as-judge scoring)

## Responsibilities

The LLM module is responsible for:
- Building grounded prompts
- Calling the language model
- Normalizing outputs
- Supporting debug and evaluation modes
- Providing a stable interface for the rest of the system

## Not Responsible For

The LLM module does NOT:
- Perform retrieval
- Fuse FAISS/BM25 results
- Handle metadata lookup
- Manage indexes or embeddings
- Run the full pipeline
- Serve API requests

Those responsibilities belong to other modules.

## Extensibility

This module is designed to support:
- Multiple LLM backends
- Custom prompt templates
- Chain-of-thought prompting
- Multi-step reasoning
- Domain-specific prompt formats
- Model-specific optimizations

Because the RAG engine uses the wrapper interface, new models can be added without modifying the pipeline.

## Testing

Tests for the LLM layer are located in:
tests/test_rag_engine.py
tests/test_rag_pipeline.py
tests/test_grounding.py

These ensure:
- prompts are constructed correctly
- grounding logic is enforced
- LLM outputs integrate cleanly with the pipeline

