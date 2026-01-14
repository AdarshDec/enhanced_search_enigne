"""
Knowledge Graph Component
========================
Manage entities and relationships for GraphRAG.
Uses NetworkX for graph operations and JSON for persistence.
"""

import networkx as nx
import json
import os
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass

@dataclass
class Triplet:
    source: str
    relation: str
    target: str

class KnowledgeGraph:
    def __init__(self, persist_dir: str = "."):
        self.persist_dir = persist_dir
        self.graph_file = f"{persist_dir}/knowledge_graph.json"
        self.graph = nx.DiGraph()
        
        # Load existing graph if available
        self.load()

    def add_triplet(self, source: str, relation: str, target: str):
        """Add a subject-predicate-object triplet to the graph"""
        # Add nodes (entities)
        self.graph.add_node(source, type="entity")
        self.graph.add_node(target, type="entity")
        
        # Add edge (relationship)
        self.graph.add_edge(source, target, relation=relation)

    def add_triplets(self, triplets: List[Tuple[str, str, str]]):
        """Batch add triplets"""
        for source, relation, target in triplets:
            self.add_triplet(source, relation, target)
        self.save()

    def get_subgraph(self, entities: List[str], depth: int = 1) -> List[str]:
        """
        Get relevant knowledge as text (triplets) for a list of starting entities.
        Explores neighbors up to `depth` hops.
        """
        relevant_triplets = set()
        
        for entity in entities:
            if entity not in self.graph:
                continue
                
            # Find neighbors within depth
            # For depth=1, we just look at immediate incoming and outgoing edges
            # For now, simple 1-hop traversal
            
            # Outgoing: entity -> target
            if self.graph.has_node(entity):
                for neighbor in self.graph.successors(entity):
                    edge_data = self.graph.get_edge_data(entity, neighbor)
                    if edge_data:
                        relation = edge_data.get("relation", "related_to")
                        relevant_triplets.add(f"{entity} {relation} {neighbor}")
                
            # Incoming: source -> entity
            if self.graph.has_node(entity):
                for predecessor in self.graph.predecessors(entity):
                    edge_data = self.graph.get_edge_data(predecessor, entity)
                    if edge_data:
                        relation = edge_data.get("relation", "related_to")
                        relevant_triplets.add(f"{predecessor} {relation} {entity}")

        return list(relevant_triplets)

    def save(self):
        """Save graph to disk as adjacency list"""
        data = nx.node_link_data(self.graph)
        try:
            with open(self.graph_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✓ Saved knowledge graph to {self.graph_file}")
        except Exception as e:
            print(f"⚠ Error saving knowledge graph: {e}")

    def load(self):
        """Load graph from disk"""
        if not os.path.exists(self.graph_file):
            return
            
        try:
            with open(self.graph_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data)
            print(f"✓ Loaded knowledge graph with {self.graph.number_of_nodes()} entities")
        except Exception as e:
            print(f"⚠ Error loading knowledge graph: {e}")

    def get_stats(self) -> Dict:
        return {
            "entities": self.graph.number_of_nodes(),
            "relationships": self.graph.number_of_edges()
        }
