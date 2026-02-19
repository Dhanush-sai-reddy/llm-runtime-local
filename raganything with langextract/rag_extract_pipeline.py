"""LangExtract + RAG-Anything pipeline via LangGraph — Gemini API (free tier)."""
import asyncio, json, os, re, time, nest_asyncio
from pathlib import Path
from typing import TypedDict
import google.generativeai as genai
from langgraph.graph import StateGraph, END

nest_asyncio.apply()

GEMINI_MODEL = "gemini-2.0-flash"
DATA_FILE = "sample_data.txt"
IMAGES = ["sample_vitals_chart.png", "sample_lab_results.png", "sample_medication_timeline.png", "sample_pain_scores.png", "sample_spo2.png"]
QUERY = "What symptoms does the patient have and what are the vitals?"

try:
    from dotenv import load_dotenv; load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or input("Paste your Google API key: ").strip()
genai.configure(api_key=api_key)
model = genai.GenerativeModel(GEMINI_MODEL)

# Rate-limited generate with auto-retry on 429
def gen(*args, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            time.sleep(5)  # 5s gap = ~12 RPM, under 15 RPM free limit
            return model.generate_content(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = 15 * (attempt + 1)
                print(f"  [rate limited, waiting {wait}s...]", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")

EXTRACT_PROMPT = """Extract medical entities from these clinical notes as a JSON array.
Each entity: {"class": "patient_info|symptom|vitals|assessment|plan", "text": "exact text", "attributes": {}}
Return ONLY the JSON array."""

IMAGE_PROMPT = "Extract all data from this medical image as JSON. Include: image_type, patient, date, measurements/values with units, flags, trends. Return ONLY valid JSON."

def parse_json(text):
    cleaned = re.sub(r"^```\w*\n?|```$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(cleaned.strip())

# Embed with retry
def embed_with_retry(texts):
    results = []
    for t in texts:
        for attempt in range(5):
            try:
                time.sleep(2)
                results.append(genai.embed_content(model="models/text-embedding-004", content=t, task_type="retrieval_document")["embedding"])
                break
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    time.sleep(15 * (attempt + 1))
                else:
                    raise
    return results

def get_rag():
    from lightrag.utils import EmbeddingFunc
    from raganything import RAGAnything, RAGAnythingConfig
    async def llm(prompt, system_prompt=None, history_messages=[], **kw):
        kw.pop("response_format", None)
        full = (f"System: {system_prompt}\n\n" if system_prompt else "") + prompt
        return gen(full).text
    async def embed(texts):
        return embed_with_retry(texts)
    return RAGAnything(config=RAGAnythingConfig(working_dir="rag_storage_gemini"), llm_model_func=llm, vision_model_func=llm,
        embedding_func=EmbeddingFunc(embedding_dim=768, max_token_size=8192, func=embed))

class State(TypedDict):
    raw_text: str; extractions: list; image_descs: dict; combined: str; answer: str

def extract_text(s: State) -> dict:
    print(f"\n--- Extract (Gemini) ---")
    try:
        exts = parse_json(gen(f"{EXTRACT_PROMPT}\n\n{s['raw_text']}").text)
    except Exception as e:
        print(f"  Error: {e}"); exts = []
    for e in exts: print(f"  [{e.get('class','')}] {e.get('text','')}")
    return {"extractions": exts}

def analyze_images(s: State) -> dict:
    print(f"\n--- Images (Gemini) ---")
    descs = {}
    for img in IMAGES:
        if not Path(img).exists(): continue
        print(f"  {img}...", end=" ", flush=True)
        try:
            img_data = {"mime_type": "image/png", "data": Path(img).read_bytes()}
            resp = gen([IMAGE_PROMPT, img_data]).text
            try: descs[img] = json.dumps(json.loads(resp), indent=2); print("OK (JSON)")
            except: descs[img] = resp; print(f"OK ({len(resp)} chars)")
        except Exception as e:
            descs[img] = f'{{"error":"{e}"}}'; print(f"Failed: {e}")
    return {"image_descs": descs}

def rag_index(s: State) -> dict:
    print("\n--- RAG Index (this may take a few mins on free tier) ---")
    parts = [s["raw_text"], "\n# Entities"] + [f"- [{e.get('class','')}] {e.get('text','')}" for e in s["extractions"]]
    parts += [f"\n# Image: {n}\n{d}" for n,d in s["image_descs"].items()]
    combined = "\n".join(parts)
    asyncio.get_event_loop().run_until_complete(get_rag().lightrag.ainsert(combined))
    print("  Indexed.")
    return {"combined": combined}

def rag_query(s: State) -> dict:
    print(f"\n--- Query: {QUERY} ---")
    answer = asyncio.get_event_loop().run_until_complete(get_rag().aquery(QUERY, mode="hybrid"))
    print(f"\n  Answer:\n{answer}")
    return {"answer": str(answer)}

g = StateGraph(State)
for name, fn in [("extract", extract_text), ("images", analyze_images), ("index", rag_index), ("query", rag_query)]:
    g.add_node(name, fn)
g.set_entry_point("extract")
g.add_edge("extract", "images"); g.add_edge("images", "index"); g.add_edge("index", "query"); g.add_edge("query", END)
pipeline = g.compile()

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    raw = Path(DATA_FILE).read_text(encoding="utf-8")
    print(f"Loaded {DATA_FILE} ({len(raw)} chars)")
    print("Using Gemini free tier — pipeline will be slow due to rate limits (~2-5 min total)")
    result = pipeline.invoke({"raw_text": raw, "extractions": [], "image_descs": {}, "combined": "", "answer": ""})
    json.dump({"extractions": result.get("extractions",[]), "images": result.get("image_descs",{}), "query": QUERY, "answer": result.get("answer","")}, open("output.json","w"), indent=2)
    print("\nDone. Results in output.json")
