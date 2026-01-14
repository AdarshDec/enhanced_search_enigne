"""
Hybrid Search Engine
====================
Combines keyword search (TF-IDF) with semantic search (vector embeddings)
for improved retrieval accuracy.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from search_engine import SearchEngine
from vector_store import VectorStore
from llm_service import LLMService
from knowledge_graph import KnowledgeGraph


@dataclass
class HybridSearchResult:
    """Result from hybrid search"""
    id: int
    title: str
    content: str
    snippet: str
    keyword_score: float
    semantic_score: float
    combined_score: float
    search_type: str  # "keyword", "semantic", or "hybrid"


class HybridSearchEngine:
    """
    Hybrid Search Engine
    
    Combines:
    1. Keyword search (TF-IDF) - exact matches, handles specific terms
    2. Semantic search (Vector embeddings) - conceptual similarity
    
    Benefits:
    - Better recall: finds documents even without exact keyword matches
    - Better precision: keyword matches help rank exact matches higher
    - Handles synonyms and related concepts naturally
    """
    
    def __init__(
        self,
        keyword_weight: float = 0.4,
        semantic_weight: float = 0.6,
        vector_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize hybrid search engine
        
        Args:
            keyword_weight: Weight for keyword search scores (0-1)
            semantic_weight: Weight for semantic search scores (0-1)
            vector_model: Sentence transformer model name
        """
        self.keyword_search = SearchEngine()
        self.vector_store = VectorStore(model_name=vector_model)
        self.llm_service = LLMService()
        self.knowledge_graph = KnowledgeGraph()
        
        # Normalize weights
        total = keyword_weight + semantic_weight
        self.keyword_weight = keyword_weight / total
        self.semantic_weight = semantic_weight / total
        
        self._synced_doc_ids = set()
    
    def add_document(self, title: str, content: str) -> int:
        """
        Add a document to both keyword and vector indexes
        
        Args:
            title: Document title
            content: Document content
            
        Returns:
            Document ID
        """
        # Add to keyword search
        doc_id = self.keyword_search.add_document(title, content)
        
        # Add to vector store (if available)
        if self.vector_store.is_available():
            self.vector_store.add_document(doc_id, title, content)
            
        # Add to Knowledge Graph (extract entities & relations)
        if self.llm_service.is_available():
            print(f"Extracting knowledge from document {doc_id}...")
            triplets = self.llm_service.extract_entities(content)
            if triplets:
                print(f"Found {len(triplets)} knowledge triplets")
                self.knowledge_graph.add_triplets(triplets)
        
        self._synced_doc_ids.add(doc_id)
        return doc_id
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        use_hybrid: bool = True,
        keyword_only: bool = False,
        semantic_only: bool = False
    ) -> List[Dict]:
        """
        Perform hybrid search
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_hybrid: Use both keyword and semantic search (default)
            keyword_only: Use only keyword search
            semantic_only: Use only semantic search
            
        Returns:
            List of search results with scores
        """
        if not query.strip():
            return []
        
        # Determine search mode
        if keyword_only:
            return self._keyword_search(query, top_k)
        elif semantic_only:
            return self._semantic_search(query, top_k)
        elif use_hybrid and self.vector_store.is_available():
            return self._hybrid_search(query, top_k)
        else:
            # Fallback to keyword search
            return self._keyword_search(query, top_k)
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform keyword search only"""
        results = self.keyword_search.search(query, top_k=top_k)
        for result in results:
            result["keyword_score"] = result.pop("score", 0.0)
            result["semantic_score"] = 0.0
            result["combined_score"] = result["keyword_score"]
            result["search_type"] = "keyword"
        return results
    
    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform semantic search only"""
        if not self.vector_store.is_available():
            return []
        
        vector_results = self.vector_store.search(query, top_k=top_k)
        
        results = []
        for doc_id, semantic_score, doc_dict in vector_results:
            # Get full document from keyword search
            doc = self.keyword_search.documents.get(doc_id)
            if not doc:
                continue
            
            snippet = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            
            results.append({
                "id": doc_id,
                "title": doc.title,
                "content": doc.content,
                "snippet": snippet,
                "keyword_score": 0.0,
                "semantic_score": semantic_score,
                "combined_score": semantic_score,
                "search_type": "semantic"
            })
        
        return results
    
    def _hybrid_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform hybrid search combining keyword and semantic"""
        # Get keyword results
        keyword_results = self.keyword_search.search(query, top_k=top_k * 2)
        keyword_scores = {r["id"]: r["score"] for r in keyword_results}
        
        # Get semantic results
        vector_results = self.vector_store.search(query, top_k=top_k * 2)
        semantic_scores = {doc_id: score for doc_id, score, _ in vector_results}
        
        # Normalize scores to [0, 1] range
        keyword_max = max(keyword_scores.values()) if keyword_scores else 1.0
        semantic_max = max(semantic_scores.values()) if semantic_scores else 1.0
        
        if keyword_max > 0:
            keyword_scores = {k: v / keyword_max for k, v in keyword_scores.items()}
        if semantic_max > 0:
            semantic_scores = {k: v / semantic_max for k, v in semantic_scores.items()}
        
        # Combine results
        all_doc_ids = set(keyword_scores.keys()) | set(semantic_scores.keys())
        
        combined_results = []
        for doc_id in all_doc_ids:
            kw_score = keyword_scores.get(doc_id, 0.0)
            sem_score = semantic_scores.get(doc_id, 0.0)
            
            # Weighted combination
            combined_score = (
                self.keyword_weight * kw_score +
                self.semantic_weight * sem_score
            )
            
            # Get document details
            doc = self.keyword_search.documents.get(doc_id)
            if not doc:
                continue
            
            snippet = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            
            combined_results.append({
                "id": doc_id,
                "title": doc.title,
                "content": doc.content,
                "snippet": snippet,
                "keyword_score": round(kw_score, 4),
                "semantic_score": round(sem_score, 4),
                "combined_score": round(combined_score, 4),
                "search_type": "hybrid"
            })
        
        # Sort by combined score and return top K
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return combined_results[:top_k]
    
    def answer_question(
        self,
        question: str,
        top_k_docs: int = 5,
        use_hybrid: bool = True
    ) -> Dict:
        """
        Answer a question using RAG (Retrieval Augmented Generation)
        
        Process:
        1. Retrieve relevant documents using hybrid search
        2. Pass documents as context to LLM
        3. LLM generates answer based on retrieved context
        
        Args:
            question: The question to answer
            top_k_docs: Number of documents to retrieve as context
            use_hybrid: Use hybrid search for retrieval
            
        Returns:
            Dict with 'answer', 'sources', 'model', and 'retrieved_docs'
        """
        # Retrieve relevant documents
        retrieved_docs = self.search(
            question,
            top_k=top_k_docs,
            use_hybrid=use_hybrid
        )
        
        # Knowledge Graph Retrieval (GraphRAG)
        graph_context = []
        try:
            # Extract main entities from the question to query the graph
            query_entities = self.llm_service.extract_key_terms(question)
            if query_entities:
                # Get relevant facts (subgraph)
                facts = self.knowledge_graph.get_subgraph(query_entities)
                if facts:
                    print(f"Found {len(facts)} relevant facts in Knowledge Graph")
                    graph_context = [{
                        "title": "Knowledge Graph Facts",
                        "content": "Known Facts:\n" + "\n".join(facts),
                        "id": "KG"
                    }]
        except Exception as e:
            print(f"⚠ Graph retrieval failed: {e}")
        
        # Combine contexts (Graph facts + Retrieved Docs)
        all_context = graph_context + retrieved_docs

        # Generate answer using LLM
        answer_result = self.llm_service.generate_answer(
            question=question,
            context_documents=all_context
        )
        
        # Add retrieval metadata
        answer_result["retrieved_docs"] = [
            {
                "id": doc["id"],
                "title": doc["title"],
                "score": doc.get("combined_score", doc.get("score", 0))
            }
            for doc in retrieved_docs
        ]
        
        return answer_result
    
    def get_all_documents(self) -> List[Dict]:
        """Get all indexed documents"""
        return self.keyword_search.get_all_documents()
    
    def get_stats(self) -> Dict:
        """Get statistics about the search engine"""
        keyword_stats = self.keyword_search.get_stats()
        vector_stats = self.vector_store.get_stats()
        llm_info = self.llm_service.get_info()
        
        return {
            **keyword_stats,
            "vector_search_available": vector_stats["available"],
            "vector_model": vector_stats.get("model_name"),
            "llm_provider": llm_info["provider"],
            "llm_model": llm_info["model"],
            "llm_available": llm_info["available"],
            "llm_available": llm_info["available"],
            "search_mode": "hybrid" if vector_stats["available"] else "keyword_only",
            "knowledge_graph_entities": self.knowledge_graph.get_stats()["entities"]
        }
    
    def remove_document(self, doc_id: int):
        """Remove a document from all indexes"""
        if doc_id in self.keyword_search.documents:
            del self.keyword_search.documents[doc_id]
        
        if doc_id in self.keyword_search.doc_word_counts:
            del self.keyword_search.doc_word_counts[doc_id]
        
        # Remove from vector store
        self.vector_store.remove_document(doc_id)
        
        # Remove from inverted index (simplified - would need full reindex in production)
        # For now, just mark as removed
        if doc_id in self._synced_doc_ids:
            self._synced_doc_ids.remove(doc_id)



