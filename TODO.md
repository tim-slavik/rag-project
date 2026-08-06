# TODO — Future Improvements

This document tracks planned enhancements for the small‑scale RAG system.  
Items are grouped by subsystem to keep the roadmap organized and easy to maintain.

---

## 🧩 Chunking Module

- **Add sliding‑window chunking**
  - Improve recall for long documents
  - Reduce boundary‑related retrieval failures
  - Provide a third strategy for comparison

- **Add deterministic chunking tests**
  - Validate fixed‑size boundaries
  - Validate semantic chunk grouping
  - Validate metadata correctness
  - Validate `clean_text` normalization

- **Add example corpus for debugging**
  - Tiny text samples to demonstrate chunking behavior
  - Useful for development and documentation

- **Enhance metadata**
  - Add `document_id`, `section`, `version`, `source`
  - Improve traceability for larger ingestion pipelines

- **Add adaptive or model‑based semantic chunking**
  - Use embeddings or sentence similarity
  - Produce more coherent chunks for complex documents

- **Add sliding‑window semantic chunking**
  - Combine sentence boundaries with overlap
  - Hybrid approach for high‑quality chunking

---

## 🔍 Retrieval & Reranking

- **Add transformer‑based cross‑encoder reranker**
  - Replace placeholder reranker with a real model
  - Improve precision for top‑k ranking

- **Add reranking evaluation tests**
  - Validate reranker impact on retrieval quality
  - Compare hybrid‑only vs hybrid+rereanking performance

- **Add retrieval regression snapshots**
  - Store expected top‑k results for key queries
  - Detect silent degradations in FAISS/BM25 behavior

---

## 🧠 RAG Engine & Pipeline

- **Add ingestion snapshots**
  - Store chunking + embedding outputs per version
  - Enable reproducible ingestion runs

- **Add prompt‑template versioning**
  - Track changes to prompt formats
  - Prevent regressions in synthesis behavior

- **Add multi‑query batch pipeline**
  - Support batch evaluation and batch inference

---

## 📊 Evaluation

- **Add AMAQA‑specific evaluation sets**
  - Curated queries for relevance + grounding checks
  - Regression tests tied to AMAQA metadata

- **Add semantic similarity metrics**
  - Embedding‑based relevance scoring
  - Optional upgrade beyond lexical heuristics

- **Add grounding coverage metrics**
  - Measure how much retrieved evidence is used
  - Detect hallucinations more reliably

---

## 🌐 API

- **Add streaming responses (SSE/WebSockets)**
  - Stream token‑by‑token LLM output
  - Improve UX for long answers

- **Add batch `/rag` endpoint**
  - Accept multiple queries in a single request

- **Add health + metadata endpoints**
  - `/health`
  - `/stats`
  - `/version`

---

## 🧪 Tests

- **Add deterministic ingestion tests**
  - Validate chunking + embedding consistency

- **Add FAISS dimension + metadata tests**
  - Ensure index and embeddings remain aligned

- **Add pipeline end‑to‑end regression tests**
  - Validate full RAG flow across versions

---

## 📚 Documentation

- **Add architecture diagrams**
  - High‑level RAG flow
  - Retrieval + reranking pipeline
  - Ingestion → FAISS → pipeline

- **Add developer guide**
  - How to extend chunking
  - How to add new rerankers
  - How to update ingestion

- **Add dataset usage notes**
  - AMAQA license summary
  - Citation requirements
  - Non‑commercial usage reminders

---

## 🧱 Infrastructure (Optional Future Work)

- **Add sharded FAISS indexes**
  - Support larger datasets
  - Enable distributed retrieval

- **Add embedding model versioning**
  - Track embedding changes across ingestion runs

- **Add ingestion caching**
  - Avoid recomputing embeddings unnecessarily

---

This list will evolve as the project grows.  
Each item is scoped to be achievable without compromising the simplicity and clarity of the small‑scale RAG system.