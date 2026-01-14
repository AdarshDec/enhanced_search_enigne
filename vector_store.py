"""
Vector Store for Semantic Search
=================================
Handles vector embeddings and similarity search using sentence transformers.
Provides semantic search capabilities alongside keyword search.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pickle
import os


@dataclass
class VectorDocument:
    """Document with vector embedding"""
    id: int
    title: str
    content: str
    embedding: np.ndarray


class VectorStore:
    """
    Vector Store for Semantic Search
    
    Uses sentence transformers to create embeddings and perform similarity search.
    This enables semantic search - finding documents that are conceptually similar
    even if they don't share exact keywords.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_dir: str = "."):
        """
        Initialize vector store
        
        Args:
            model_name: Name of sentence transformer model to use
            persist_dir: Directory to save/load vector store data
        """
        self.model_name = model_name
        self.persist_dir = persist_dir
        self.store_file = f"{persist_dir}/vector_store.pkl"
        
        self.model = None
        self.documents: Dict[int, VectorDocument] = {}
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.doc_ids: List[int] = []
        
        self._initialize_model()
        
        # Try to load existing data
        if self.is_available():
            self.load()
    
    def _initialize_model(self):
        """Load sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("✓ Model loaded successfully")
        except ImportError:
            print("⚠ Warning: sentence-transformers not installed.")
            print("  Install with: pip install sentence-transformers")
            print("  Falling back to keyword-only search.")
        except Exception as e:
            print(f"⚠ Warning: Could not load model: {e}")
            print("  Falling back to keyword-only search.")
    
    def save(self):
        """Save vector store to disk"""
        if not self.documents:
            return
            
        try:
            # We save the documents dict which contains embeddings
            data = {
                "documents": self.documents,
                "model_name": self.model_name
            }
            
            with open(self.store_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Saved vector store to {self.store_file}")
        except Exception as e:
            print(f"⚠ Error saving vector store: {e}")
            
    def load(self):
        """Load vector store from disk"""
        if not os.path.exists(self.store_file):
            return
            
        try:
            with open(self.store_file, 'rb') as f:
                data = pickle.load(f)
            
            # Verify model match
            if data.get("model_name") != self.model_name:
                print(f"⚠ Warning: Vector store model ({data.get('model_name')}) differs from current ({self.model_name}). Ignoring saved data.")
                return
                
            self.documents = data["documents"]
            self._rebuild_matrix()
            print(f"✓ Loaded {len(self.documents)} vectors from {self.store_file}")
        except Exception as e:
            print(f"⚠ Error loading vector store: {e}")
    
    def is_available(self) -> bool:
        """Check if vector search is available"""
        return self.model is not None
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to vectors
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            numpy array of embeddings (shape: [len(texts), embedding_dim])
        """
        if not self.is_available():
            raise RuntimeError("Vector model not available")
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        return embeddings
    
    def add_document(self, doc_id: int, title: str, content: str) -> bool:
        """
        Add a document to the vector store
        
        Args:
            doc_id: Unique document identifier
            title: Document title
            content: Document content
            
        Returns:
            True if successfully added, False otherwise
        """
        if not self.is_available():
            return False
        
        # Combine title and content for better context
        text = f"{title}\n{content}"
        
        # Encode document
        embedding = self.encode([text])[0]
        
        # Store document
        self.documents[doc_id] = VectorDocument(
            id=doc_id,
            title=title,
            content=content,
            embedding=embedding
        )
        
        # Rebuild embeddings matrix
        self._rebuild_matrix()
        
        # SAVE STATE
        self.save()
        
        return True
    
    def _rebuild_matrix(self):
        """Rebuild the embeddings matrix for efficient similarity search"""
        if not self.documents:
            self.embeddings_matrix = None
            self.doc_ids = []
            return
        
        self.doc_ids = sorted(self.documents.keys())
        self.embeddings_matrix = np.array([
            self.documents[doc_id].embedding for doc_id in self.doc_ids
        ])
    
    def remove_document(self, doc_id: int):
        """Remove a document from the vector store"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self._rebuild_matrix()
            self.save()
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0
    ) -> List[Tuple[int, float, Dict]]:
        """
        Perform semantic search using vector similarity
        
        Args:
            query: Search query string
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1) to include in results
            
        Returns:
            List of tuples: (doc_id, score, document_dict)
            Sorted by score (highest first)
        """
        if not self.is_available() or not self.documents:
            return []
        
        # Encode query
        query_embedding = self.encode([query])[0]
        
        # Calculate cosine similarity with all documents
        # Since embeddings are normalized, cosine similarity = dot product
        similarities = np.dot(self.embeddings_matrix, query_embedding)
        
        # Get top K results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            score = float(similarities[idx])
            
            # Filter by minimum score
            if score < min_score:
                continue
            
            doc = self.documents[doc_id]
            results.append((
                doc_id,
                score,
                {
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content,
                    "score": round(score, 4),
                    "type": "semantic"
                }
            ))
        
        return results
    
    def get_document(self, doc_id: int) -> Optional[VectorDocument]:
        """Get a document by ID"""
        return self.documents.get(doc_id)
    
    def clear(self):
        """Clear all documents from the vector store"""
        self.documents.clear()
        self.embeddings_matrix = None
        self.doc_ids = []
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            "total_documents": len(self.documents),
            "model_name": self.model_name if self.is_available() else None,
            "embedding_dim": (
                self.embeddings_matrix.shape[1] 
                if self.embeddings_matrix is not None 
                else None
            ),
            "available": self.is_available()
        }



