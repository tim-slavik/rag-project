# Evaluation Module

The `evaluation/` module provides the tools used to measure retrieval quality, grounding strength, and overall RAG system performance. It acts as the quality‑control layer of the project, ensuring that changes to retrieval, chunking, or synthesis do not silently degrade results.

This module is intentionally lightweight and designed for small‑scale RAG experimentation. It provides deterministic evaluation logic and integrates cleanly with the pipeline and test suite.

## Overview

The evaluation layer provides:
- A simple evaluation harness (`evaluator.py`)
- Deterministic scoring functions
- Grounding and relevance checks
- Regression‑style comparisons across pipeline versions

The goal is to make it easy to:
- test retrieval quality
- validate grounding behavior
- detect regressions early
- compare different chunking or retrieval strategies

## Files

### `evaluator.py`
The main evaluation interface.

Responsibilities:
- Accept a query and the pipeline output
- Inspect retrieved chunks
- Score relevance based on lexical and semantic signals
- Score grounding based on evidence usage
- Produce a structured evaluation report

The evaluator is intentionally model‑agnostic and dependency‑free.  
It does not require an LLM‑as‑judge or external scoring models.

### `metrics.py`
Provides deterministic scoring functions.

Includes:
- lexical relevance scoring
- simple semantic similarity heuristics
- grounding checks (does the answer use retrieved evidence?)
- stability checks for regression testing

These metrics are designed to be:
- fast
- deterministic
- easy to interpret
- suitable for small‑scale RAG systems

### `reports.py`
Utility functions for formatting evaluation results.

Responsibilities:
- produce human‑readable summaries
- generate structured dictionaries for programmatic use
- highlight failures or regressions

This keeps evaluation output consistent across tests and experiments.

## How It Fits in the System

Evaluation sits outside the main RAG pipeline:

Query  
↓  
Pipeline (retrieval → reranking → RAG engine)  
↓  
Answer + Retrieved Chunks  
↓  
Evaluator  
↓  
Metrics + Report

It is used during development, testing, and regression checks.

## Responsibilities

The evaluation module is responsible for:
- scoring retrieval quality
- checking grounding behavior
- detecting regressions
- providing structured evaluation reports
- supporting deterministic, repeatable experiments

## Not Responsible For

The evaluation module does NOT:
- perform retrieval
- build prompts
- call the LLM
- chunk documents
- embed text
- manage indexes

Those responsibilities belong to other modules.

## Extensibility

This module is designed to support:
- additional metrics (semantic similarity, coverage, novelty)
- LLM‑as‑judge scoring (optional future enhancement)
- dataset‑based evaluation (AMAQA, custom corpora)
- multi‑query batch evaluation
- regression dashboards

Because the evaluator is modular, new metrics can be added without modifying the pipeline.

## Testing

Evaluation behavior is validated through:
- `tests/test_evaluation.py`
- `tests/test_retrieval.py`
- `tests/test_rag_pipeline.py`

These ensure:
- relevance scoring is deterministic
- grounding checks behave consistently
- evaluation reports remain stable across changes