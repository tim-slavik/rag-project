# Hybrid Search Module

The `hybrid_search/` module implements the full hybrid retrieval stack used by the RAG system. It combines dense vector search (FAISS), lexical search (BM25), and fusion strategies (RRF) to produce high‑quality, contextually relevant retrieval results.

This module is one of the core strengths of the system: it ensures that retrieval is robust, accurate, and resistant to embedding failures or lexical gaps.

## Overview

Hybrid search integrates multiple retrieval signals:
- Dense similarity (FAISS)
- Lexical relevance (BM25)
- Rank fusion (RRF)
- Optional reranking hooks
- Metadata-aware chunk selection

The goal is to produce a ranked list of chunks that best match the user query, regardless of whether the query is semantic, keyword-based, or mixed.

## Files

### `bm25.py`
Implements BM25 lexical search over the AMAQA corpus.

Key responsibilities:
- Tokenize documents
- Build BM25 index
- Compute lexical relevance scores
- Return ranked document IDs and scores

BM25 is especially strong for:
- keyword-heavy queries
- short queries
- queries with rare terms
- exact phrase matching

### `fusion.py`
Implements Reciprocal Rank Fusion (RRF), a simple but powerful method for combining multiple ranked lists.

RRF advantages:
- Works well with heterogeneous retrieval signals
- Resistant to noise in any single retriever
- Easy to tune
- Produces stable rankings

This module merges FAISS and BM25 results into a single fused ranking.

### `hybrid_retriever.py`
The main hybrid retrieval interface.

Key responsibilities:
- Run FAISS search
- Run BM25 search
- Normalize and merge results
- Apply RRF fusion
- Return final ranked chunks with metadata
- Provide hooks for optional reranking

This is the component used by:
- the pipeline orchestrator
- the API layer
- the evaluation harness
- the test suite

## How It Fits in the System

Hybrid search sits at the front of the RAG pipeline:

Query
↓
FAISS Vector Search
↓
BM25 Lexical Search
↓
Reciprocal Rank Fusion (RRF)
↓
Optional Reranking
↓
Pipeline Orchestrator
↓
RAG Engine
↓
LLM
↓
Final Answer


It ensures that the RAG engine receives the best possible context.

## Responsibilities

The hybrid search module is responsible for:
- Running dense and lexical retrieval
- Combining results using fusion
- Returning ranked chunks with metadata
- Supporting reranking integration
- Providing consistent retrieval behavior across the system

## Not Responsible For

Hybrid search does NOT:
- Build FAISS indexes
- Generate embeddings
- Build BM25 indexes
- Call the LLM
- Perform synthesis
- Handle API requests

Those responsibilities belong to other modules.

## Extensibility

This module is designed to support:
- Additional retrievers (e.g., Elasticsearch, ColBERT)
- Custom fusion strategies
- Weighted fusion
- Domain-specific retrieval logic
- Query rewriting or expansion

## Testing

Tests for hybrid search are located in:
tests/test_retrieval.py


These ensure:
- FAISS and BM25 return correct results
- fusion behaves predictably
- hybrid retrieval is stable across changes
- reranking hooks integrate cleanly