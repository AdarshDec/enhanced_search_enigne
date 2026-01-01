
import os
from search_engine import SearchEngine
from vector_store import VectorStore

def test_persistence():
    print("Test: Creating first engine instance...")
    engine1 = SearchEngine()
    
    test_title = "Persistence Test Document"
    test_content = "This document should survive a restart."
    
    print(f"Test: Adding document: '{test_title}'")
    doc_id = engine1.add_document(test_title, test_content)
    
    print("Test: Document added. Destroying first engine instance...")
    del engine1
    
    print("Test: Creating second engine instance (simulating restart)...")
    engine2 = SearchEngine()
    
    # Check if document exists
    print(f"Test: Checking if document {doc_id} exists...")
    retrieved_doc = engine2.documents.get(doc_id)
    
    if retrieved_doc:
        print(f"✓ SUCCESS: Document found: {retrieved_doc.title}")
        if retrieved_doc.content == test_content:
             print("✓ SUCCESS: Content matches.")
        else:
             print("❌ FAIL: Content corrupted.")
    else:
        print("❌ FAIL: Document lost!")

    # Cleanup
    if os.path.exists("search_index.json"):
        os.remove("search_index.json")
    if os.path.exists("vector_store.pkl"):
        os.remove("vector_store.pkl")

if __name__ == "__main__":
    test_persistence()
