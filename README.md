# Small‑Scale RAG System (AMAQA)

This repository contains a fully functional, small‑scale Retrieval‑Augmented Generation (RAG) system built around the AMAQA dataset. The project is designed to be clean, modular, deterministic, and easy to understand—while still demonstrating the architectural patterns used in production‑grade RAG systems.

The system includes:
- ingestion + chunking
- embeddings + FAISS vectorstore
- hybrid retrieval (FAISS + BM25 + fusion)
- optional reranking
- a pipeline orchestrator
- a lightweight RAG engine
- an evaluation harness
- a FastAPI interface

Everything is dependency‑minimal and intentionally transparent, making this an ideal reference implementation for learning, experimentation, or interviews.

---

## 📐 Architecture Overview

The system follows a standard RAG flow:
Raw Documents (AMAQA + custom)
↓
Chunking (fixed / semantic)
↓
Embeddings (dense vectors)
↓
FAISS Index
↓
Hybrid Retrieval (FAISS + BM25 + fusion)
↓
Optional Reranking (cross‑encoder placeholder)
↓
Pipeline Orchestrator
↓
RAG Engine (prompting + synthesis)
↓
LLM Wrapper
↓
Final Answer


All components are modular and can be swapped or extended without changing the rest of the system.

---

## 📁 Repository Structure
data/               # raw AMAQA data, chunks, embeddings, FAISS index, metadata
chunking/           # chunking strategies + orchestrator
vectorstore/        # FAISS loading, search, and metadata lookup
hybrid_search/      # FAISS + BM25 + fusion retrieval
reranking/          # optional cross‑encoder reranker
pipeline/           # orchestrator that ties retrieval + RAG engine together
rag/                # RAG engine (prompt building + synthesis)
llm/                # LLM wrapper (OpenAI, local models, etc.)
evaluation/         # relevance + grounding metrics, reports
api/                # FastAPI server + request/response schemas
tests/              # deterministic unit + integration tests
docs/               # optional diagrams or notes


Each folder contains its own README describing responsibilities, interfaces, and extensibility.

---

## 📚 Dataset: AMAQA

This project uses the **AMAQA: A Metadata-based QA Dataset for RAG Systems** dataset.

The official AMAQA license and README are included in:
data/amaqa_raw/LICENSE
data/amaqa_raw/README.md

### License  
AMAQA is licensed under **CC BY‑NC‑ND 4.0**  
https://creativecommons.org/licenses/by-nc-nd/4.0/

### Required Citation
```bibtex
@misc{bruni2026amaqametadatabasedqadataset,
      title={AMAQA: A Metadata-based QA Dataset for RAG Systems}, 
      author={Davide Bruni and Marco Avvenuti and Nicola Tonellotto and Maurizio Tesconi},
      year={2026},
      eprint={2505.13557},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2505.13557}, 
}
```


---

## 🧩 Key Features

### 🔹 Deterministic Chunking
- Fixed‑size chunking  
- Semantic chunking (NLTK sentence boundaries)  
- Clean metadata for traceability  

### 🔹 Hybrid Retrieval
- FAISS dense vector search  
- BM25 lexical search  
- Fusion scoring  
- Top‑k selection  

### 🔹 Optional Reranking
- Lightweight cross‑encoder placeholder  
- Deterministic scoring  
- Clean interface for future transformer‑based rerankers  

### 🔹 Pipeline Orchestrator
- Single entry point for RAG queries  
- Retrieval → reranking → prompting → synthesis  
- Debug mode for inspection  

### 🔹 RAG Engine
- Prompt construction  
- Evidence grounding  
- Answer synthesis  

### 🔹 Evaluation Harness
- Relevance scoring  
- Grounding checks  
- Regression‑style comparisons  
- Deterministic metrics  

### 🔹 FastAPI Interface
- `/rag` endpoint  
- Pydantic request/response models  
- Optional debug output  

### 🔹 Test Suite
- deterministic unit tests  
- integration tests  
- grounding + retrieval validation  

---

## 🚀 Running the API

Start the FastAPI server:

```bash
uvicorn api.main:app --reload
Query the RAG endpoint:
curl -X POST http://localhost:8000/rag \
     -H "Content-Type: application/json" \
     -d '{"query": "What is metadata in AMAQA?"}'

