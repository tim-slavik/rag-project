from fastapi import FastAPI
from pydantic import BaseModel
from rag.engine import RAGEngine

app = FastAPI(title="AMAQA RAG Engine", version="1.0.0")

# Load engine once at startup
engine = RAGEngine()

# -----------------------------
# Request models
# -----------------------------
class QueryRequest(BaseModel):
    query: str
    k: int = 5

class SearchRequest(BaseModel):
    query: str
    k: int = 10

# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "engine": "ready"}

# -----------------------------
# Query endpoint (full RAG answer)
# -----------------------------
@app.post("/query")
def query(req: QueryRequest):
    result = engine.answer(req.query, req.k)
    return result

# -----------------------------
# Search endpoint (raw material)
# -----------------------------
@app.post("/search")
def search(req: SearchRequest):
    docs = engine.retrieve(req.query, req.k)
    return {"query": req.query, "results":docs}