import os
import json
import logging
import asyncio
from typing import List, Dict, Any
import pandas as pd
import uuid

# LangExtract 
from langextract import extract

# MS GraphRAG schema imports
# GraphRAG nodes require 'id', 'title', 'type', 'description', 'source_id'
# GraphRAG edges require 'source', 'target', 'weight', 'description', 'source_id', 'id'

from schemas import GraphData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VLLM_URL = "http://localhost:8000/v1"
MODEL_NAME = "google/gemma-3-1b-it" 

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """Simple character-based chunking."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

async def extract_knowledge(text_chunks: List[str]) -> List[GraphData]:
    """Extract entities and relationships from chunks using LangExtract."""
    # We use the langextract function
    # We need to construct examples for langextract.
    # Note: langextract requires 'examples' (a list of ExampleData objects).
    from langextract.core.data import ExampleData
    
    # Minimal dummy examples
    examples = [
        ExampleData(
            text="John works at Microsoft.",
            extractions=GraphData(
                entities=[
                    {"name": "John", "type": "PERSON", "description": "Employee at Microsoft", "attributes": []},
                    {"name": "Microsoft", "type": "ORGANIZATION", "description": "A tech company", "attributes": []}
                ],
                relationships=[
                    {"source_entity": "John", "target_entity": "Microsoft", "relationship_type": "WORKS_AT", "description": "John is an employee at Microsoft"}
                ]
            )
        )
    ]
    
    tasks = []
    for chunk in text_chunks:
        tasks.append(
            extract(
                text_or_documents=chunk,
                prompt_description="Identify all key entities (people, places, concepts) and their relationships mentioned in the text.",
                examples=examples,
                model_id=MODEL_NAME,
                api_key="empty",
                model_url=VLLM_URL,
            )
        )
        
    logger.info(f"Extracting graph data from {len(text_chunks)} chunks...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    valid_results = []
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"Extraction failed for a chunk: {r}")
        elif r:
             # Depending on LangExtract version, it might return the Pydantic object directly or a dict.
            if hasattr(r, 'entities'):
                valid_results.append(r)
            elif isinstance(r, dict):
                 valid_results.append(GraphData(**r))
                 
    return valid_results

def consolidate_graph(graph_results: List[GraphData]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge the chunks into a unified set of entities and relationships
    formatted for Microsoft GraphRAG downstream tasks.
    """
    nodes_dict = {}
    edges_dict = {}
    
    for doc_idx, graph in enumerate(graph_results):
        source_id = f"chunk_{doc_idx}"
        
        # 1. Consolidate Entities (Nodes)
        for ent in graph.entities:
            # We use uppercase 'name' as a simple normalization key
            ent_key = ent.name.strip().upper()
            if not ent_key: continue
            
            if ent_key not in nodes_dict:
                nodes_dict[ent_key] = {
                    "id": str(uuid.uuid4()),
                    "title": ent.name,
                    "type": ent.type.upper(),
                    "description": ent.description,
                    "source_id": source_id,
                    "degree": 0 # We'll calculate this later
                }
            else:
                # Merge descriptions if seen again
                nodes_dict[ent_key]["description"] += f" | {ent.description}"
                nodes_dict[ent_key]["source_id"] += f",{source_id}"

        # 2. Consolidate Relationships (Edges)
        for rel in graph.relationships:
            src_key = rel.source_entity.strip().upper()
            tgt_key = rel.target_entity.strip().upper()
            
            if not src_key or not tgt_key: continue
            
            # Create a unique key for the undirected edge
            pair_key = tuple(sorted([src_key, tgt_key]))
            edge_key = f"{pair_key[0]}_{pair_key[1]}_{rel.relationship_type.upper()}"
            
            if edge_key not in edges_dict:
                edges_dict[edge_key] = {
                    "id": str(uuid.uuid4()),
                    "source": rel.source_entity, # Display name
                    "target": rel.target_entity, # Display name
                    "weight": 1.0,
                    "description": rel.description,
                    "source_id": source_id
                }
            else:
                edges_dict[edge_key]["weight"] += 1.0
                edges_dict[edge_key]["description"] += f" | {rel.description}"
                edges_dict[edge_key]["source_id"] += f",{source_id}"

    # Calculate basic node degree (required by some GraphRAG visualizers)
    for edge in edges_dict.values():
        src_key = edge["source"].strip().upper()
        tgt_key = edge["target"].strip().upper()
        if src_key in nodes_dict: nodes_dict[src_key]["degree"] += int(edge["weight"])
        if tgt_key in nodes_dict: nodes_dict[tgt_key]["degree"] += int(edge["weight"])

    # Convert to DataFrames
    nodes_df = pd.DataFrame(list(nodes_dict.values()))
    edges_df = pd.DataFrame(list(edges_dict.values()))
    
    return nodes_df, edges_df

async def main():
    logger.info("Starting Custom GraphRAG Pipeline...")
    
    # 1. Read input documents
    all_text = ""
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(INPUT_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                all_text += f.read() + "\n\n"
                
    if not all_text.strip():
        logger.error(f"No text found in {INPUT_DIR}/. Add a .txt file and run again.")
        return
        
    # 2. Chunk Text
    chunks = chunk_text(all_text)
    logger.info(f"Generated {len(chunks)} text chunks.")
    
    # 3. Extract Knowledge using LangExtract + vLLM
    graph_results = await extract_knowledge(chunks)
    logger.info(f"Successfully extracted data from {len(graph_results)} chunks.")
    
    if not graph_results:
        logger.error("Failed to extract any graph data.")
        return
        
    # 4. Consolidate and format for MS GraphRAG
    nodes_df, edges_df = consolidate_graph(graph_results)
    
    logger.info(f"Consolidated Graph: {len(nodes_df)} Nodes, {len(edges_df)} Edges.")
    
    # 5. Export to Parquet
    # MS GraphRAG expects these naming conventions inside the 'output/' folder
    # specifically inside a table output path which differs by version. 
    # Usually it's output/create_final_nodes.parquet
    
    nodes_path = os.path.join(OUTPUT_DIR, "create_final_nodes.parquet")
    edges_path = os.path.join(OUTPUT_DIR, "create_final_relationships.parquet")
    
    nodes_df.to_parquet(nodes_path)
    edges_df.to_parquet(edges_path)
    
    # Save a JSON copy for easy local inspection
    nodes_df.to_json(os.path.join(OUTPUT_DIR, "nodes.json"), orient="records", indent=2)
    edges_df.to_json(os.path.join(OUTPUT_DIR, "edges.json"), orient="records", indent=2)
    
    logger.info(f"Exported Nodes and Relationships to {OUTPUT_DIR}/")
    logger.info("Extraction complete. You can now configure MS GraphRAG to run downstream clustering.")

if __name__ == "__main__":
    asyncio.run(main())
