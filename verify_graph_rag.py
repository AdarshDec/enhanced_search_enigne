
import sys
import os
import shutil
from unittest.mock import MagicMock

# Ensure we can import from current directory
sys.path.append(os.getcwd())

from hybrid_search_engine import HybridSearchEngine
from llm_service import LLMService

def test_graph_rag():
    print("Testing GraphRAG Flow...")
    
    # 1. Setup clean environment
    if os.path.exists("test_data"):
        shutil.rmtree("test_data")
    os.makedirs("test_data")
    
    # 2. Initialize Engine
    engine = HybridSearchEngine(vector_model="all-MiniLM-L6-v2")
    # Override persistence to use test_data
    engine.keyword_search.persist_dir = "test_data"
    engine.keyword_search.index_file = "test_data/search_index.json"
    engine.knowledge_graph.persist_dir = "test_data"
    engine.knowledge_graph.graph_file = "test_data/knowledge_graph.json"
    
    # 3. Mock LLM Service to return predictable extractions
    # We want to test that if the LLM extracts 'Alice', the graph stores it.
    original_extract_entities = engine.llm_service.extract_entities
    original_extract_keys = engine.llm_service.extract_key_terms
    
    def mock_extract_entities(text):
        if "Alice" in text:
            return [("Alice", "works_at", "TechCorp")]
        if "TechCorp" in text:
            return [("TechCorp", "located_in", "Seattle")]
        return []
    
    def mock_extract_keys(text):
        if "Alice" in text:
            return ["Alice"]
        return []

    engine.llm_service.extract_entities = mock_extract_entities
    engine.llm_service.extract_key_terms = mock_extract_keys
    
    # 4. Add Documents
    print("Adding documents...")
    engine.add_document("Doc 1", "Alice is a software engineer.")
    engine.add_document("Doc 2", "TechCorp has big offices.")
    
    # 5. Verify Graph Content
    print("\nVerifying Graph Storage...")
    stats = engine.knowledge_graph.get_stats()
    print(f"Graph Stats: {stats}")
    
    # We expect nodes: Alice, TechCorp, Seattle (3 nodes)
    # Edges: Alice->TechCorp, TechCorp->Seattle (2 edges)
    assert stats["entities"] >= 3, f"Expected at least 3 entities, got {stats['entities']}"
    assert stats["relationships"] >= 2, f"Expected at least 2 relationships, got {stats['relationships']}"
    
    # 6. Verify Search (Graph Retrieval)
    print("\nVerifying Graph Retrieval during Search...")
    # We ask about Alice. We expect the system to retrieve 'Alice works_at TechCorp' from the graph.
    # We'll inspect the context passed to generate_answer (by mocking it too or checking the return)
    
    # We cannot easily check the internal variable 'all_context' without mocking generate_answer
    # so let's mock generate_answer to print the context
    
    original_generate = engine.llm_service.generate_answer
    
    captured_context = []
    def mock_generate(question, context_documents, max_tokens=500):
        captured_context.extend(context_documents)
        return {
            "answer": "Mock Answer",
            "sources": [],
            "model": "mock"
        }
    
    engine.llm_service.generate_answer = mock_generate
    
    engine.answer_question("Where does Alice work?")
    
    # Check if Graph Facts are in the context
    found_graph_facts = False
    for doc in captured_context:
        if doc.get("id") == "KG":
            print(f"FAILED TO FIND KG CONTEXT. Context: {doc}")
            if "Alice works_at TechCorp" in doc["content"]:
                print("✓ Found expected graph fact: Alice works_at TechCorp")
                found_graph_facts = True
                
    if found_graph_facts:
        print("\nSUCCESS: GraphRAG is working!")
    else:
        print("\nFAILURE: Did not find graph facts in context.")
        
    # Cleanup
    if os.path.exists("test_data"):
        shutil.rmtree("test_data")

if __name__ == "__main__":
    test_graph_rag()
