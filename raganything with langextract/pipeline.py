"""
Pipeline — LangExtract + Graph RAG, fully local with Ollama.
"""

import json, sys
from langgraph.graph import StateGraph, END
from typing import TypedDict
from extract import extract_from_file, extractions_to_dicts, print_results
from graph_rag import KnowledgeGraph

kg = KnowledgeGraph()

class State(TypedDict):
    source_file: str; source_text: str; entities: list; error: str; query: str; answer: str

def ingest(state: State) -> State:
    try:
        text = open(state["source_file"], "r", encoding="utf-8").read()
        print(f"\n  Ingested {len(text)} chars")
        kg.ingest(text)
        return {**state, "source_text": text}
    except FileNotFoundError:
        return {**state, "error": f"File not found: {state['source_file']}"}

def extract(state: State) -> State:
    if state.get("error"): return state
    result = extract_from_file(state["source_file"])
    if not result: return {**state, "error": "Extraction failed"}
    print_results(result)
    entities = extractions_to_dicts(result)
    json.dump(entities, open("entities.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"  {len(entities)} entities → entities.json")
    return {**state, "entities": entities}

def build(state: State) -> State:
    if state.get("error") or not state.get("entities"): return state
    kg.add_entities(state["entities"])
    kg.infer_relationships()
    kg.stats()
    kg.save()
    kg.visualize()
    return state

def answer(state: State) -> State:
    if state.get("error") or not state.get("query"): return state
    a = kg.query(state["query"])
    print(f"\n  Q: {state['query']}\n  A: {a}")
    return {**state, "answer": a}

def route(state: State) -> str:
    return "error" if state.get("error") else "ok"

def build_pipeline():
    g = StateGraph(State)
    g.add_node("ingest", ingest)
    g.add_node("extract", extract)
    g.add_node("build", build)
    g.add_node("answer", answer)
    g.set_entry_point("ingest")
    g.add_conditional_edges("ingest", route, {"ok": "extract", "error": END})
    g.add_conditional_edges("extract", route, {"ok": "build", "error": END})
    g.add_conditional_edges("build", route, {"ok": "answer", "error": END})
    g.add_edge("answer", END)
    return g.compile()

def interactive():
    """Query loop over existing graph."""
    try: kg.load()
    except FileNotFoundError:
        print("No graph found. Run full pipeline first."); return

    print("\n  Interactive mode. 'quit' to exit, 'stats' for graph info.\n")
    while True:
        try: q = input("  Q: ").strip()
        except (EOFError, KeyboardInterrupt): break
        if not q: continue
        if q.lower() in ("quit", "exit", "q"): break
        if q.lower() == "stats": kg.stats(); continue
        print(f"  A: {kg.query(q)}\n")

if __name__ == "__main__":
    if "--query" in sys.argv:
        interactive()
    else:
        print("\n  LOCAL RAG: LangExtract + Graph RAG + Ollama\n")
        result = build_pipeline().invoke({
            "source_file": "sample_data.txt", "source_text": "", "entities": [],
            "error": "", "query": "Summarize all patients and their diagnoses.", "answer": ""
        })
        if result.get("error"): print(f"  Error: {result['error']}")
        else: interactive()
