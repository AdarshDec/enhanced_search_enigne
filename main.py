"""
Enhanced FastAPI Backend with Search + Retrieval + LLM
======================================================
REST API with hybrid search (keyword + semantic) and LLM-powered Q&A
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os

from hybrid_search_engine import HybridSearchEngine

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Search Engine with LLM",
    description="Hybrid search engine with semantic search and LLM-powered question answering",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize hybrid search engine
search_engine = HybridSearchEngine()


# Request/Response Models

class DocumentRequest(BaseModel):
    """Request model for adding a document"""
    title: str
    content: str


class QuestionRequest(BaseModel):
    """Request model for asking questions"""
    question: str
    top_k_docs: Optional[int] = 5
    use_hybrid: Optional[bool] = True


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint - serves the frontend"""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Enhanced Search Engine API", "docs": "/docs"}


@app.post("/documents", response_model=dict)
async def add_document(doc: DocumentRequest):
    """
    Add a new document to the search index
    
    Indexes the document in both:
    - Keyword search index (TF-IDF)
    - Vector store (semantic embeddings)
    """
    doc_id = search_engine.add_document(doc.title, doc.content)
    return {
        "id": doc_id,
        "message": "Document indexed successfully",
        "title": doc.title
    }


@app.get("/search")
async def search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, description="Number of results to return"),
    mode: str = Query("hybrid", description="Search mode: 'hybrid', 'keyword', or 'semantic'")
):
    """
    Search for documents matching a query
    
    Modes:
    - 'hybrid': Combines keyword (TF-IDF) and semantic (vector) search (default)
    - 'keyword': Only keyword search
    - 'semantic': Only semantic search (requires vector store)
    
    Returns results with both keyword and semantic scores.
    """
    if not q.strip():
        return {"results": [], "query": q, "message": "Empty query"}
    
    # Determine search mode
    keyword_only = (mode == "keyword")
    semantic_only = (mode == "semantic")
    use_hybrid = (mode == "hybrid")
    
    results = search_engine.search(
        q,
        top_k=top_k,
        use_hybrid=use_hybrid,
        keyword_only=keyword_only,
        semantic_only=semantic_only
    )
    
    return {
        "results": results,
        "query": q,
        "count": len(results),
        "mode": mode
    }


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Answer a question using RAG (Retrieval Augmented Generation)
    
    Process:
    1. Retrieves relevant documents using hybrid search
    2. Uses LLM to generate answer based on retrieved context
    
    Returns the answer along with source documents.
    """
    if not request.question.strip():
        return {
            "answer": "Please provide a question.",
            "sources": [],
            "model": None
        }
    
    result = search_engine.answer_question(
        question=request.question,
        top_k_docs=request.top_k_docs,
        use_hybrid=request.use_hybrid
    )
    
    return result


@app.get("/documents")
async def get_all_documents():
    """Get all indexed documents"""
    documents = search_engine.get_all_documents()
    return {
        "documents": documents,
        "count": len(documents)
    }


@app.get("/stats")
async def get_stats():
    """
    Get search engine statistics
    
    Returns:
    - Document counts
    - Search capabilities (keyword, semantic, LLM)
    - Model information
    """
    stats = search_engine.get_stats()
    return stats


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    stats = search_engine.get_stats()
    return {
        "status": "healthy",
        "documents_indexed": stats.get("total_documents", 0),
        "vector_search": stats.get("vector_search_available", False),
        "llm_available": stats.get("llm_available", False),
        "search_mode": stats.get("search_mode", "unknown")
    }


# Load sample documents on startup
@app.on_event("startup")
async def load_sample_documents():
    """Load sample documents when the server starts"""
    try:
        from sample_documents import SAMPLE_DOCUMENTS
        
        print("=" * 60)
        print("Loading sample documents...")
        print("=" * 60)
        
        for doc in SAMPLE_DOCUMENTS:
            search_engine.add_document(doc["title"], doc["content"])
        
        stats = search_engine.get_stats()
        print(f"✓ Loaded {stats['total_documents']} sample documents")
        print(f"✓ Indexed {stats['total_words_indexed']} words")
        print(f"✓ Found {stats['unique_words']} unique words")
        
        if stats.get("vector_search_available"):
            print(f"✓ Vector search enabled with model: {stats.get('vector_model')}")
        else:
            print("⚠ Vector search not available (install sentence-transformers)")
        
        if stats.get("llm_available"):
            print(f"✓ LLM enabled: {stats.get('llm_provider')} ({stats.get('llm_model')})")
        if stats.get("llm_available"):
            print(f"✓ LLM enabled: {stats.get('llm_provider')} ({stats.get('llm_model')})")
        else:
            print("⚠ LLM not available (set OPENAI_API_KEY or run Ollama)")
            
        kg_entities = stats.get("knowledge_graph_entities", 0)
        print(f"✓ Knowledge Graph active: {kg_entities} entities")
        
        print("=" * 60)
        print("Server ready! Visit http://localhost:8000")
        print("API docs: http://localhost:8000/docs")
        print("=" * 60)
        
    except ImportError:
        print("No sample documents found. Start indexing documents via API!")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)



