"""
Search Engine Core Module
=========================
This module implements the fundamental concepts of search engines:
1. Inverted Index - Maps words to documents
2. TF-IDF Ranking - Calculates relevance scores
3. Document Indexing - Processes and stores documents
"""

import re
import math
from collections import defaultdict
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Document:
    """Represents a document in the search engine"""
    id: int
    title: str
    content: str


class SearchEngine:
    """
    Core Search Engine Implementation
    
    Key Concepts:
    ------------
    1. INVERTED INDEX: 
       - Instead of: Document → Words
       - We store: Word → Documents containing it
       - Example: "python" → [doc1, doc3, doc5]
       - This makes searching MUCH faster!
    
    2. TF-IDF (Term Frequency - Inverse Document Frequency):
       - TF: How often a word appears in a document (normalized)
       - IDF: How rare/common a word is across all documents
       - Score = TF × IDF
       - Higher score = More relevant document
    """
    
    
    def __init__(self, persist_dir: str = "."):
        self.persist_dir = persist_dir
        self.index_file = f"{persist_dir}/search_index.json"
        
        # Inverted Index: word → list of [document_id, positions]
        self.inverted_index: Dict[str, List[List]] = defaultdict(list)
        
        # Document storage: document_id → Document object
        self.documents: Dict[int, Document] = {}
        
        # Document word counts: document_id → total word count
        self.doc_word_counts: Dict[int, int] = {}
        
        # Word document frequency: word → number of documents containing it
        self.word_doc_frequency: Dict[str, int] = defaultdict(int)
        
        self.next_doc_id = 1
        
        # Try to load existing data
        self.load()
    
    def save(self):
        """Save search engine state to disk"""
        import json
        import os
        
        # Convert documents to dict for serialization
        docs_dict = {
            doc_id: {"id": d.id, "title": d.title, "content": d.content}
            for doc_id, d in self.documents.items()
        }
        
        state = {
            "documents": docs_dict,
            "inverted_index": self.inverted_index,
            "doc_word_counts": self.doc_word_counts,
            "word_doc_frequency": self.word_doc_frequency,
            "next_doc_id": self.next_doc_id
        }
        
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            print(f"✓ Saved search index to {self.index_file}")
        except Exception as e:
            print(f"⚠ Error saving search index: {e}")
            
    def load(self):
        """Load search engine state from disk"""
        import json
        import os
        
        if not os.path.exists(self.index_file):
            return
            
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Restore documents
            self.documents = {}
            for doc_id_str, doc_data in state["documents"].items():
                doc_id = int(doc_id_str)
                self.documents[doc_id] = Document(
                    id=doc_data["id"],
                    title=doc_data["title"],
                    content=doc_data["content"]
                )
            
            # Restore other structures
            # Note: JSON converts int keys to strings, so we need to convert back where necessary
            self.inverted_index = defaultdict(list, state["inverted_index"])
            
            self.doc_word_counts = {
                int(k): v for k, v in state["doc_word_counts"].items()
            }
            
            self.word_doc_frequency = defaultdict(int, state["word_doc_frequency"])
            self.next_doc_id = state["next_doc_id"]
            
            print(f"✓ Loaded {len(self.documents)} documents from {self.index_file}")
            
        except Exception as e:
            print(f"⚠ Error loading search index: {e}")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenization: Break text into individual words
        
        Steps:
        1. Convert to lowercase (case-insensitive search)
        2. Remove punctuation
        3. Split into words
        4. Filter out empty strings
        """
        # Convert to lowercase and remove punctuation
        text = text.lower()
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Split into words and filter empty strings
        words = [word for word in text.split() if word]
        return words
    
    def _calculate_tf(self, word: str, doc_id: int) -> float:
        """
        Term Frequency (TF): How often a word appears in a document
        
        Formula: count(word in document) / total words in document
        
        Why normalize? Longer documents would naturally have more word occurrences.
        Normalizing makes scores comparable across documents of different lengths.
        """
        # Count how many times word appears in this document
        word_count = 0
        if word in self.inverted_index:
            for entry in self.inverted_index[word]:
                doc_id_in_index, positions = entry[0], entry[1]
                if doc_id_in_index == doc_id:
                    word_count = len(positions)
                    break
        
        # Normalize by document length
        total_words = self.doc_word_counts.get(doc_id, 1)
        return word_count / total_words if total_words > 0 else 0
    
    def _calculate_idf(self, word: str) -> float:
        """
        Inverse Document Frequency (IDF): How rare/common a word is
        
        Formula: log(total_documents / documents_containing_word)
        
        Why log? To dampen the effect - common words shouldn't be weighted too heavily.
        
        Examples:
        - "the" appears in all documents → IDF = log(10/10) = 0 (not important)
        - "python" appears in 2/10 documents → IDF = log(10/2) = 1.61 (important!)
        """
        total_docs = len(self.documents)
        if total_docs == 0:
            return 0
        
        docs_with_word = self.word_doc_frequency.get(word, 0)
        if docs_with_word == 0:
            return 0
        
        # Add 1 to avoid division by zero
        return math.log(total_docs / docs_with_word)
    
    def _calculate_tfidf(self, word: str, doc_id: int) -> float:
        """
        TF-IDF Score: Combines TF and IDF
        
        Score = TF(word, document) × IDF(word)
        
        High score means:
        - Word appears frequently in the document (high TF)
        - Word is relatively rare across all documents (high IDF)
        - This document is likely very relevant!
        """
        tf = self._calculate_tf(word, doc_id)
        idf = self._calculate_idf(word)
        return tf * idf
    
    def add_document(self, title: str, content: str) -> int:
        """
        Index a new document
        
        Process:
        1. Create Document object
        2. Tokenize content into words
        3. Build inverted index (word → document mappings)
        4. Update statistics
        """
        doc_id = self.next_doc_id
        self.next_doc_id += 1
        
        # Create document
        doc = Document(id=doc_id, title=title, content=content)
        self.documents[doc_id] = doc
        
        # Tokenize content
        words = self._tokenize(content)
        self.doc_word_counts[doc_id] = len(words)
        
        # Track which documents contain each word (for IDF calculation)
        unique_words_in_doc = set()
        
        # Build inverted index
        # For each word, record which document it appears in and at what positions
        for position, word in enumerate(words):
            # Track document frequency (for IDF)
            if word not in unique_words_in_doc:
                unique_words_in_doc.add(word)
                self.word_doc_frequency[word] += 1
            
            # Add to inverted index
            # Check if this document is already in the list for this word
            found = False
            for i, (existing_doc_id, positions) in enumerate(self.inverted_index[word]):
                if existing_doc_id == doc_id:
                    # Update the positions list
                    positions.append(position)
                    found = True
                    break
            
            if not found:
                self.inverted_index[word].append([doc_id, [position]])
        
        # SAVE STATE
        self.save()
        
        return doc_id
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search for documents matching a query
        
        Process:
        1. Tokenize query
        2. Find documents containing query words (using inverted index)
        3. Calculate TF-IDF score for each document
        4. Rank by score (highest first)
        5. Return top results
        
        Returns:
        List of dictionaries with document info and relevance score
        """
        # Tokenize query
        query_words = self._tokenize(query)
        
        if not query_words:
            return []
        
        # Find candidate documents (documents containing at least one query word)
        candidate_docs = set()
        for word in query_words:
            if word in self.inverted_index:
                for entry in self.inverted_index[word]:
                    doc_id = entry[0]
                    candidate_docs.add(doc_id)
        
        if not candidate_docs:
            return []
        
        # Calculate TF-IDF scores for each candidate document
        doc_scores = {}
        for doc_id in candidate_docs:
            score = 0.0
            # Sum TF-IDF scores for all query words
            for word in query_words:
                score += self._calculate_tfidf(word, doc_id)
            doc_scores[doc_id] = score
        
        # Sort by score (descending) and get top K
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = sorted_docs[:top_k]
        
        # Build results
        results = []
        for doc_id, score in top_docs:
            doc = self.documents[doc_id]
            
            # Create snippet (first 200 characters of content)
            snippet = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            
            results.append({
                "id": doc_id,
                "title": doc.title,
                "content": doc.content,
                "snippet": snippet,
                "score": round(score, 4)
            })
        
        return results
    
    def get_stats(self) -> Dict:
        """Get statistics about the search engine"""
        total_words = sum(self.doc_word_counts.values())
        unique_words = len(self.inverted_index)
        
        return {
            "total_documents": len(self.documents),
            "total_words_indexed": total_words,
            "unique_words": unique_words,
            "average_words_per_document": round(total_words / len(self.documents), 2) if self.documents else 0
        }
    
    def get_all_documents(self) -> List[Dict]:
        """Get all indexed documents"""
        return [
            {
                "id": doc.id,
                "title": doc.title,
                "content": doc.content
            }
            for doc in self.documents.values()
        ]


