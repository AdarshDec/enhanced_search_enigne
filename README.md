# Enhanced Search Engine: Search + Retrieval + LLM

A comprehensive search engine with hybrid search (keyword + semantic), vector embeddings, and LLM-powered question answering using RAG.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python main.py
```

3. Open browser:
```
http://localhost:8000
```

## Features

- **Hybrid Search**: Combines keyword (TF-IDF) and semantic (vector) search
- **Question Answering**: LLM-powered Q&A using RAG
- **Multiple LLM Providers**: OpenAI, Ollama, or Mock mode

## Optional LLM Setup

### OpenAI:
```bash
export OPENAI_API_KEY="your-key-here"
```

### Ollama:
```bash
ollama pull llama2
```

## Files

- `main.py` - FastAPI server
- `hybrid_search_engine.py` - Main search engine
- `vector_store.py` - Semantic search
- `llm_service.py` - LLM integration
- `search_engine.py` - Keyword search (TF-IDF)


