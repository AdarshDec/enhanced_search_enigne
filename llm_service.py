"""
LLM Service for Question Answering and Text Generation
======================================================
Integrates with language models for RAG (Retrieval Augmented Generation)
and question answering capabilities.
"""

import os
import json
import re
from typing import List, Dict, Optional, Tuple
from enum import Enum

# Optional imports for LLM providers
try:
    import requests
except ImportError:
    requests = None


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    OLLAMA = "ollama"  # Local LLM via Ollama
    MOCK = "mock"  # For testing without API keys


class LLMService:
    """
    LLM Service for generating answers and text
    
    Supports multiple providers:
    - OpenAI GPT models (requires API key)
    - Ollama for local models (requires Ollama installed)
    - Mock mode for testing
    """
    
    def __init__(self, provider: str = None):
        """
        Initialize LLM service
        
        Args:
            provider: LLM provider to use. Auto-detects if None.
        """
        self.provider = provider or self._detect_provider()
        self.client = None
        self.model_name = None
        self._initialize_client()
    
    def _detect_provider(self) -> str:
        """Auto-detect which LLM provider to use based on environment"""
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            return LLMProvider.OPENAI
        
        # Check if Ollama is available (default localhost)
        if requests:
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    return LLMProvider.OLLAMA
            except:
                pass
        
        # Fall back to mock
        return LLMProvider.MOCK
    
    def _initialize_client(self):
        """Initialize the LLM client based on provider"""
        if self.provider == LLMProvider.OPENAI:
            self._init_openai()
        elif self.provider == LLMProvider.OLLAMA:
            self._init_ollama()
        else:
            self._init_mock()
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            
            self.client = OpenAI(api_key=api_key)
            self.model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            print(f"✓ Initialized OpenAI with model: {self.model_name}")
        except ImportError:
            print("⚠ Warning: openai package not installed.")
            print("  Install with: pip install openai")
            print("  Falling back to mock mode.")
            self._init_mock()
        except Exception as e:
            print(f"⚠ Warning: Could not initialize OpenAI: {e}")
            print("  Falling back to mock mode.")
            self._init_mock()
    
    def _init_ollama(self):
        """Initialize Ollama client (local LLM)"""
        if not requests:
            raise ImportError("requests package required for Ollama")
        
        try:
            self.client = requests
            self.model_name = os.getenv("OLLAMA_MODEL", "llama2")
            
            # Test connection
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError("Ollama not accessible")
            
            print(f"✓ Initialized Ollama with model: {self.model_name}")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize Ollama: {e}")
            print("  Make sure Ollama is running: https://ollama.ai")
            print("  Falling back to mock mode.")
            self._init_mock()
    
    def _init_mock(self):
        """Initialize mock client (for testing without API)"""
        self.client = None
        self.model_name = "mock"
        print("ℹ Using mock LLM (no API calls)")
    
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.client is not None or self.provider == LLMProvider.MOCK
    
    def generate_answer(
        self,
        question: str,
        context_documents: List[Dict],
        max_tokens: int = 500
    ) -> Dict:
        """
        Generate an answer to a question using retrieved documents (RAG)
        
        Args:
            question: The question to answer
            context_documents: List of relevant documents (dicts with 'title' and 'content')
            max_tokens: Maximum tokens in response
            
        Returns:
            Dict with 'answer', 'sources', and 'model' keys
        """
        if not context_documents:
            return {
                "answer": "I couldn't find any relevant documents to answer this question.",
                "sources": [],
                "model": self.model_name
            }
        
        # Build context from documents
        context = self._build_context(context_documents)
        
        # Build prompt
        prompt = self._build_rag_prompt(question, context)
        
        # Generate response
        if self.provider == LLMProvider.OPENAI:
            return self._generate_openai(prompt, max_tokens, context_documents)
        elif self.provider == LLMProvider.OLLAMA:
            return self._generate_ollama(prompt, max_tokens, context_documents)
        else:
            return self._generate_mock(question, context_documents)
    
    def _build_context(self, documents: List[Dict]) -> str:
        """Build context string from documents"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(
                f"Document {i}:\n"
                f"Title: {doc.get('title', 'Untitled')}\n"
                f"Content: {doc.get('content', '')[:500]}\n"
            )
        return "\n".join(context_parts)
    
    def _build_rag_prompt(self, question: str, context: str) -> str:
        """Build RAG prompt for question answering"""
        return f"""You are a helpful assistant that answers questions based on provided documents.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer the question using only the information from the provided documents.
- If the documents don't contain enough information, say so.
- Be concise and accurate.
- Cite which document(s) you used for your answer.

Answer:"""
    
    def _generate_openai(self, prompt: str, max_tokens: int, sources: List[Dict]) -> Dict:
        """Generate answer using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "answer": answer,
                "sources": [{"title": doc.get("title"), "id": doc.get("id")} for doc in sources[:5]],
                "model": self.model_name
            }
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "model": self.model_name
            }
    
    def _generate_ollama(self, prompt: str, max_tokens: int, sources: List[Dict]) -> Dict:
        """Generate answer using Ollama"""
        try:
            response = self.client.post(
                f"http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                
                return {
                    "answer": answer,
                    "sources": [{"title": doc.get("title"), "id": doc.get("id")} for doc in sources[:5]],
                    "model": self.model_name
                }
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "model": self.model_name
            }
    
    def _generate_mock(self, question: str, sources: List[Dict]) -> Dict:
        """Generate mock answer (for testing)"""
        titles = [doc.get("title", "Untitled") for doc in sources[:3]]
        return {
            "answer": f"[Mock Response] Based on the retrieved documents ({', '.join(titles)}), here's a sample answer to your question: '{question}'. This is a mock response since no LLM API is configured. To get real answers, please set up OpenAI API key or Ollama.",
            "sources": [{"title": doc.get("title"), "id": doc.get("id")} for doc in sources[:5]],
            "model": "mock"
        }
    
    def summarize(self, text: str, max_length: int = 150) -> str:
        """
        Generate a summary of the given text
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summarized text
        """
        prompt = f"Summarize the following text in {max_length} words or less:\n\n{text}\n\nSummary:"
        
        if self.provider == LLMProvider.OPENAI:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_length,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except:
                return text[:max_length] + "..." if len(text) > max_length else text
        elif self.provider == LLMProvider.OLLAMA:
            try:
                response = self.client.post(
                    f"http://localhost:11434/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    return response.json().get("response", text[:max_length]).strip()
            except:
                pass
        
        # Fallback: simple truncation
        return text[:max_length] + "..." if len(text) > max_length else text
    
    def get_info(self) -> Dict:
        """Get information about the LLM service"""
        return {
            "provider": self.provider,
            "model": self.model_name,
            "available": self.is_available()
        }

    def extract_entities(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract knowledge triplets from text using LLM
        
        Args:
            text: Text to process
            
        Returns:
            List of (source, relation, target) triplets
        """
        prompt = f"""Extract knowledge triplets from the following text.
Format each triplet as: (Subject, Relation, Object)
- Subject and Object should be entities (people, organizations, events, concepts)
- Relation should be a simple verb or phrase (e.g., worked_at, part_of, caused)
- Output ONLY the triplets, one per line.
- Do not output headers or numbering.

Text:
{text[:2000]}  # Limit text length

Triplets:"""

        triplets = []
        
        try:
            # Generate response using appropriate provider
            response_text = ""
            if self.provider == LLMProvider.OPENAI:
                # Reuse _generate_openai logic but simplified
                result = self._generate_openai(prompt, 500, [])
                response_text = result["answer"]
            elif self.provider == LLMProvider.OLLAMA:
                result = self._generate_ollama(prompt, 500, [])
                response_text = result["answer"]
            else:
                # Mock response
                return [("MockSubject", "related_to", "MockObject")]
            
            # Parse response
            # Expected format: (Subject, Relation, Object)
            for line in response_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Regex to match (A, B, C)
                match = re.search(r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)', line)
                if match:
                    s, r, t = match.groups()
                    triplets.append((s.strip(), r.strip(), t.strip()))
                    
        except Exception as e:
            print(f"⚠ Error extracting entities: {e}")
            
        return triplets

    def extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key entities/terms from a query
        
        Args:
            text: Query text
            
        Returns:
            List of key terms (e.g., ["Alice", "TechCorp"])
        """
        prompt = f"""Extract the main entities (people, organizations, places, concepts) from the following query.
Output ONLY the entities, one per line. No bullets or numbering.

Query: {text}

Entities:"""

        terms = []
        try:
            response_text = ""
            if self.provider == LLMProvider.OPENAI:
                result = self._generate_openai(prompt, 200, [])
                response_text = result["answer"]
            elif self.provider == LLMProvider.OLLAMA:
                result = self._generate_ollama(prompt, 200, [])
                response_text = result["answer"]
            else:
                return ["MockTerm"]
            
            for line in response_text.split('\n'):
                line = line.strip()
                if line:
                    terms.append(line)
        except:
            pass
            
        return terms



