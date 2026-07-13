
Each module contains its own README explaining the purpose, design, and implementation details for that stage.

---

## 🚀 Project Goals

- Build a clear, educational progression of RAG concepts  
- Demonstrate real‑world ingestion (PDF/DOCX scraping, cleaning, metadata)  
- Explore multiple retrieval strategies (FAISS, hybrid search, reranking)  
- Show how to evolve a simple RAG into a robust, production‑ready system  
- Provide a clean GitHub structure that highlights growth and engineering discipline  

---

## 🧩 Module Overview

### **01_basic_rag/**
Minimal RAG pipeline: scraping, extraction, cleaning, embeddings, retrieval.

### **02_chunking/**
Chunking strategies, windowing, overlap, and text segmentation.

### **03_metadata/**
Document metadata, source tracking, timestamps, semantic tags.

### **04_vectorstore/**
Vector database implementations (FAISS, Chroma, etc.).

### **05_hybrid_search**
BM25 + embeddings, weighted scoring, fusion retrieval.

### **06_reranking**
Cross‑encoder reranking, relevance scoring, LLM‑based rerankers.

### **07_full_rag_pipeline**
Integrated RAG system with orchestration, guardrails, and evaluation.

---

## 🔧 Installation
# Update when files and structure is finalized/complete.
```bash
pip install --upgrade pip
pipenv install
pipenv shell
