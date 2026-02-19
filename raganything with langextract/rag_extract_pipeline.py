"""
RAG-Anything + LangExtract Pipeline (LangGraph orchestration)
Nodes: extract_text -> analyze_images -> rag_index -> rag_query
"""
import asyncio, base64, json, os
from pathlib import Path
from typing import TypedDict
from types import SimpleNamespace
import langextract as lx
import httpx
from langgraph.graph import StateGraph, END

# --- Config ---
TEXT_MODEL = "gemma3:1b"
VISION_MODEL = "qwen3-vl:2b"
OLLAMA = "http://localhost:11434"
DATA_FILE = "sample_data.txt"
IMAGES = ["sample_vitals_chart.png", "sample_lab_results.png"]
QUERY = "What symptoms does the patient have and what are the vitals?"

EXAMPLE = lx.data.ExampleData(
    text="Patient: Jane Smith\nDOB: 1990-01-20\nSore throat, dry cough 2 days.\nBP 120/80, HR 72.\nAssessment: Acute pharyngitis.\nPlan: Amoxicillin 500mg TID.",
    extractions=[
        lx.data.Extraction(extraction_class="patient_info", extraction_text="Jane Smith", attributes={"field": "name"}),
        lx.data.Extraction(extraction_class="symptom", extraction_text="sore throat", attributes={"duration": "2 days"}),
        lx.data.Extraction(extraction_class="vitals", extraction_text="BP 120/80, HR 72", attributes={"summary": "normal"}),
        lx.data.Extraction(extraction_class="assessment", extraction_text="Acute pharyngitis", attributes={}),
        lx.data.Extraction(extraction_class="plan", extraction_text="Amoxicillin 500mg TID", attributes={"action": "medication"}),
    ],
)

# --- State ---
class PipelineState(TypedDict):
    raw_text: str
    extractions: list
    image_descs: dict
    combined_text: str
    answer: str

# --- Nodes ---
def extract_text(state: PipelineState) -> dict:
    print(f"\n--- Extract ({TEXT_MODEL}) ---")
    try:
        result = lx.extract(
            text_or_documents=state["raw_text"],
            prompt_description="Extract medical entities: demographics, symptoms, vitals, assessments, plans.",
            examples=[EXAMPLE], model_id=TEXT_MODEL, model_url=OLLAMA,
            fence_output=False, use_schema_constraints=False,
        )
        exts = result.extractions or []
    except Exception as e:
        print(f"  Error: {e}")
        exts = []
    for e in exts:
        print(f"  [{e.extraction_class}] {e.extraction_text}")
    return {"extractions": exts}

def analyze_images(state: PipelineState) -> dict:
    print(f"\n--- Images ({VISION_MODEL}) ---")
    descs = {}
    for img in IMAGES:
        if not Path(img).exists():
            continue
        print(f"  {img}...", end=" ")
        try:
            b64 = base64.b64encode(open(img, "rb").read()).decode()
            r = httpx.post(f"{OLLAMA}/api/generate", json={
                "model": VISION_MODEL, "prompt": f"Describe this medical image ({img}) in detail.",
                "images": [b64], "stream": False
            }, timeout=300)
            descs[img] = r.json().get("response", "")
            print(f"OK ({len(descs[img])} chars)")
        except Exception as e:
            descs[img] = f"[failed: {e}]"
            print(f"Failed: {e}")
    return {"image_descs": descs}

def rag_index(state: PipelineState) -> dict:
    print("\n--- RAG Index ---")
    parts = [state["raw_text"], "\n# Extracted Entities"]
    for e in state["extractions"]:
        parts.append(f"- [{e.extraction_class}] {e.extraction_text}")
    for name, desc in state["image_descs"].items():
        parts.append(f"\n# Image: {name}\n{desc}")
    combined = "\n".join(parts)

    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import EmbeddingFunc
    from raganything import RAGAnything, RAGAnythingConfig

    async def llm(prompt, system_prompt=None, history_messages=[], **kw):
        kw.pop("response_format", None)
        return await ollama_model_complete(prompt, system_prompt=system_prompt,
            history_messages=history_messages, host=OLLAMA, model=TEXT_MODEL, **kw)

    emb = EmbeddingFunc(embedding_dim=2048, max_token_size=8192,
        func=lambda texts: ollama_embed(texts, embed_model="qwen3-embedding", host=OLLAMA))

    rag = RAGAnything(config=RAGAnythingConfig(working_dir="rag_storage"),
        llm_model_func=llm, vision_model_func=llm, embedding_func=emb)
    asyncio.get_event_loop().run_until_complete(rag.lightrag.ainsert(combined))
    print("  Indexed.")
    return {"combined_text": combined, "_rag": rag}

def rag_query(state: PipelineState) -> dict:
    print(f"\n--- Query: {QUERY} ---")
    # Re-create RAG instance to query
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import EmbeddingFunc
    from raganything import RAGAnything, RAGAnythingConfig

    async def llm(prompt, system_prompt=None, history_messages=[], **kw):
        kw.pop("response_format", None)
        return await ollama_model_complete(prompt, system_prompt=system_prompt,
            history_messages=history_messages, host=OLLAMA, model=TEXT_MODEL, **kw)

    emb = EmbeddingFunc(embedding_dim=2048, max_token_size=8192,
        func=lambda texts: ollama_embed(texts, embed_model="qwen3-embedding", host=OLLAMA))

    rag = RAGAnything(config=RAGAnythingConfig(working_dir="rag_storage"),
        llm_model_func=llm, vision_model_func=llm, embedding_func=emb)
    answer = asyncio.get_event_loop().run_until_complete(rag.aquery(QUERY, mode="hybrid"))
    print(f"\n  Answer:\n{answer}")
    return {"answer": str(answer)}

# --- Graph ---
graph = StateGraph(PipelineState)
graph.add_node("extract_text", extract_text)
graph.add_node("analyze_images", analyze_images)
graph.add_node("rag_index", rag_index)
graph.add_node("rag_query", rag_query)
graph.set_entry_point("extract_text")
graph.add_edge("extract_text", "analyze_images")
graph.add_edge("analyze_images", "rag_index")
graph.add_edge("rag_index", "rag_query")
graph.add_edge("rag_query", END)
pipeline = graph.compile()

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    raw = Path(DATA_FILE).read_text(encoding="utf-8")
    print(f"Loaded {DATA_FILE} ({len(raw)} chars)")

    result = pipeline.invoke({"raw_text": raw, "extractions": [], "image_descs": {}, "combined_text": "", "answer": ""})

    with open("output.json", "w") as f:
        json.dump({"extractions": [{"class": e.extraction_class, "text": e.extraction_text,
            "attrs": e.attributes} for e in result.get("extractions", [])],
            "image_descriptions": result.get("image_descs", {}),
            "query": QUERY, "answer": result.get("answer", "")}, f, indent=2)
    print("\nDone. Results in output.json")
