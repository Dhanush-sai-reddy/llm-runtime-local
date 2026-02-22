"""
LangChain Graph RAG — Local Knowledge Graph with Ollama
Replaces the brittle langextract approach with a robust graph transformer.
"""

import os
import json
import networkx as nx
from typing import List, Dict, Any

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Config ─────────────────────────────────────────────────
OLLAMA_MODEL = "gemma3:1b"
OLLAMA_EMBED = "qwen3-embedding:latest"  # Or your preferred local embedding model
INPUT_FILE = "sample_data.txt"
GRAPH_FILE = "langchain_kg.json"

# Allowed Node and Edge types to constrain the LLM
ALLOWED_NODES = [
    "Patient", "Provider", "Diagnosis", "Medication", 
    "Procedure", "Facility", "Device", "Trial"
]

ALLOWED_RELATIONSHIPS = [
    "HAS_DIAGNOSIS", "PRESCRIBES", "TAKES_MEDICATION", 
    "UNDERWENT_PROCEDURE", "ADMITTED_TO", "WORKS_AT",
    "PARTICIPATES_IN"
]

def load_and_split_data(filepath: str) -> List[Document]:
    """Load text and chunk it realistically to avoid OOM."""
    print(f"Loading {filepath}...")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

    # Better to split on logical boundaries, but standard chunking works too
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "\n\n", "\n", " "]
    )
    docs = text_splitter.create_documents([text])
    print(f"Split into {len(docs)} document chunks.")
    return docs

def extract_graph_data(docs: List[Document]) -> List[Any]:
    """Use LangChain's LLMGraphTransformer to build nodes and edges."""
    print(f"Initializing LLMGraphTransformer with {OLLAMA_MODEL}...")
    llm = Ollama(model=OLLAMA_MODEL, temperature=0.1)
    
    # The transformer handles the complex prompting for us
    graph_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=ALLOWED_NODES,
        allowed_relationships=ALLOWED_RELATIONSHIPS
    )
    
    print("Extracting graph documents (this will take time)...")
    try:
        # Pass docs to be transformed into node/edge objects
        graph_documents = graph_transformer.convert_to_graph_documents(docs)
        print(f"Successfully extracted {len(graph_documents)} graph documents.")
        return graph_documents
    except Exception as e:
        print(f"Graph extraction failed: {e}")
        return []

def save_to_networkx(graph_documents: List[Any], fp=GRAPH_FILE):
    """Convert LangChain GraphDocuments to NetworkX and save."""
    G = nx.DiGraph()
    
    for g_doc in graph_documents:
        # Add Nodes
        for node in g_doc.nodes:
            G.add_node(node.id, type=node.type)
            
        # Add Edges
        for edge in g_doc.relationships:
            G.add_edge(edge.source.id, edge.target.id, type=edge.type)

    print(f"Built NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    # Save for later visualization/querying
    data = nx.node_link_data(G)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved local graph to {fp}")
    return G

def main():
    print("--- Local LangChain Graph RAG Build ---")
    docs = load_and_split_data(INPUT_FILE)
    if not docs:
        return
        
    # For testing, just take the first few chunks to avoid massive inference times
    # Omit the slice to process the whole document.
    test_docs = docs[:2] 
    
    graph_docs = extract_graph_data(test_docs)
    if graph_docs:
        save_to_networkx(graph_docs)
    print("--- Pipeline Complete ---")

if __name__ == "__main__":
    main()
