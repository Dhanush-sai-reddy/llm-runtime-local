"""
RAG-Anything + LangExtract Combined Pipeline

Multimodal pipeline flow:
  1. LangExtract (Text) -> Structured Entities
  2. Vision Model (Images) -> Text Descriptions
  3. RAG-Anything -> Knowledge Graph -> Query
"""

import argparse
import asyncio
import base64
import glob
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import langextract as lx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """Extract medical entities from clinical notes in order of appearance.
Extract patient demographics, symptoms, vitals, assessments, and plan items.
Use exact text for extractions."""

EXTRACTION_EXAMPLES = [
    lx.data.ExampleData(
        text=(
            "Patient Name: Jane Smith\nDate of Birth: 1990-01-20\n"
            "Date of Visit: 2023-09-15\n\nSubjective:\n"
            "Patient complains of sore throat and dry cough for 2 days.\n\n"
            "Objective:\nBP 120/80, HR 72.\nThroat: Erythematous.\n\n"
            "Assessment:\n1. Acute pharyngitis.\n\n"
            "Plan:\n1. Prescribe Amoxicillin 500mg three times daily."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="patient_info",
                extraction_text="Jane Smith",
                attributes={"field": "name"},
            ),
            lx.data.Extraction(
                extraction_class="patient_info",
                extraction_text="1990-01-20",
                attributes={"field": "date_of_birth"},
            ),
            lx.data.Extraction(
                extraction_class="patient_info",
                extraction_text="2023-09-15",
                attributes={"field": "visit_date"},
            ),
            lx.data.Extraction(
                extraction_class="symptom",
                extraction_text="sore throat",
                attributes={"duration": "2 days"},
            ),
            lx.data.Extraction(
                extraction_class="symptom",
                extraction_text="dry cough",
                attributes={"duration": "2 days"},
            ),
            lx.data.Extraction(
                extraction_class="vitals",
                extraction_text="BP 120/80, HR 72",
                attributes={"summary": "normal range"},
            ),
            lx.data.Extraction(
                extraction_class="assessment",
                extraction_text="Acute pharyngitis",
                attributes={"status": "acute"},
            ),
            lx.data.Extraction(
                extraction_class="plan",
                extraction_text="Prescribe Amoxicillin 500mg three times daily",
                attributes={
                    "action": "prescribe_medication",
                    "details": "Amoxicillin 500mg TID",
                },
            ),
        ],
    )
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}


# ---------------------------------------------------------------------------
# Ollama Helpers
# ---------------------------------------------------------------------------

def get_ollama_llm_func(model: str, base_url: str = "http://localhost:11434"):
    from lightrag.llm.ollama import ollama_model_complete

    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        kwargs.pop("response_format", None)
        return await ollama_model_complete(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            host=base_url,
            model=model,
            **kwargs,
        )

    return llm_func


def get_ollama_embedding_func(
    model: str = "qwen3-embedding",
    base_url: str = "http://localhost:11434",
    embedding_dim: int = 2048,
):
    from lightrag.llm.ollama import ollama_embed
    from lightrag.utils import EmbeddingFunc

    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=lambda texts: ollama_embed(
            texts,
            embed_model=model,
            host=base_url,
        ),
    )


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def describe_image_with_ollama(
    image_path: str,
    model: str = "qwen3-vl:2b",
    base_url: str = "http://localhost:11434",
) -> str:
    import httpx

    image_b64 = encode_image_base64(image_path)
    file_name = Path(image_path).name

    prompt = (
        f"Describe this medical image ({file_name}) in detail. "
        "Include all data values, labels, trends, and any clinically relevant "
        "information visible in the image. Be specific about numbers and units."
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(f"{base_url}/api/generate", json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")


# ---------------------------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------------------------

def step_langextract(text_content: str, model: str):
    print(f"\nStep 1a: Entity Extraction (Model: {model})")
    print(f"Text length: {len(text_content)} chars")

    try:
        result = lx.extract(
            text_or_documents=text_content,
            prompt_description=EXTRACTION_PROMPT,
            examples=EXTRACTION_EXAMPLES,
            model_id=model,
            model_url="http://localhost:11434",
            fence_output=False,
            use_schema_constraints=False,
        )
    except Exception as e:
        print(f"Error in LangExtract: {e}")
        return SimpleNamespace(extractions=[], document_id="unknown")

    return result


async def step_describe_images(image_paths: list[str], model: str) -> dict[str, str]:
    print(f"\nStep 1b: Image Analysis (Model: {model})")
    print(f"Images: {len(image_paths)}")

    descriptions = {}
    for img_path in image_paths:
        name = Path(img_path).name
        print(f"Analyzing {name}...")
        try:
            desc = await describe_image_with_ollama(img_path, model=model)
            descriptions[name] = desc
        except Exception as e:
            print(f"Failed to analyze {name}: {e}")
            descriptions[name] = f"[Image: {name} - analysis failed]"

    return descriptions


def format_extractions_for_rag(result, image_descriptions: dict[str, str] | None = None) -> str:
    lines = []
    lines.append("# Extracted Medical Entities\n")

    groups = {}
    for ext in (result.extractions or []):
        cls = ext.extraction_class
        if cls not in groups:
            groups[cls] = []
        groups[cls].append(ext)

    for cls, extractions in groups.items():
        lines.append(f"\n## {cls.replace('_', ' ').title()}\n")
        for ext in extractions:
            lines.append(f"- **{ext.extraction_text}**")
            if ext.attributes:
                attrs = "; ".join(f"{k}: {v}" for k, v in ext.attributes.items())
                lines.append(f"  ({attrs})")
            lines.append("")

    if image_descriptions:
        lines.append("\n## Visual Evidence\n")
        for img_name, desc in image_descriptions.items():
            lines.append(f"### {img_name}\n")
            lines.append(desc)
            lines.append("")

    lines.append("\n## Summary\n")
    entity_names = [ext.extraction_text for ext in (result.extractions or [])]
    modalities = ["text"]
    if image_descriptions:
        modalities.append(f"{len(image_descriptions)} image(s)")
    lines.append(
        f"Entities: {len(entity_names)}. "
        f"Categories: {', '.join(sorted(groups.keys()))}. "
        f"Sources: {', '.join(modalities)}."
    )

    return "\n".join(lines)


def print_extractions(result):
    print("\nExtracted Entities:")
    if result.extractions:
        for ext in result.extractions:
            print(f"[{ext.extraction_class}] {ext.extraction_text}")
            if ext.attributes:
                for k, v in ext.attributes.items():
                    print(f"  {k}: {v}")
    else:
        print("No extractions found.")


def step_save_extractions(result, image_descriptions, output_path: str):
    try:
        lx.io.save_annotated_documents(
            [result],
            output_name=os.path.basename(output_path),
            output_dir=os.path.dirname(output_path) or ".",
        )
        print(f"Entities saved: {output_path}")
    except Exception as e:
        print(f"Warning: Could not save entities to file: {e}")

    if image_descriptions:
        img_output = output_path.replace(".jsonl", "_images.json")
        with open(img_output, "w", encoding="utf-8") as f:
            json.dump(image_descriptions, f, indent=2, ensure_ascii=False)
        print(f"Image descriptions saved: {img_output}")

    print(f"Entities saved: {output_path}")


async def step_rag_ingest(structured_text: str, raw_text: str, model: str, working_dir: str):
    from raganything import RAGAnything, RAGAnythingConfig

    print(f"\nStep 2: Knowledge Graph Indexing")
    print(f"Working dir: {working_dir}")

    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    llm_func = get_ollama_llm_func(model)
    embedding_func = get_ollama_embedding_func()

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_func,
        vision_model_func=llm_func,
        embedding_func=embedding_func,
    )

    print(f"Inserting structured content ({len(structured_text)} chars)...")
    await rag.lightrag.ainsert(structured_text)

    print(f"Inserting raw text ({len(raw_text)} chars)...")
    await rag.lightrag.ainsert(raw_text)

    print("Indexing complete.")
    return rag


async def step_rag_query(rag, query: str):
    print(f"\nStep 3: RAG Query")
    print(f"Query: {query}")
    return await rag.aquery(query, mode="hybrid")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_image_paths(image_args: list[str] | None) -> list[str]:
    if not image_args:
        return []
    paths = []
    for pattern in image_args:
        expanded = glob.glob(pattern)
        if expanded:
            paths.extend(expanded)
        elif os.path.exists(pattern):
            paths.append(pattern)
    return [p for p in paths if Path(p).suffix.lower() in IMAGE_EXTENSIONS]


async def run_pipeline(
    file_path: str,
    image_paths: list[str] | None = None,
    query: str | None = None,
    text_model: str = "qwen3-vl:2b",
    vision_model: str = "qwen3-vl:2b",
    extract_only: bool = False,
    query_only: bool = False,
    output_dir: str = ".",
):
    file_path = str(Path(file_path).resolve())
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    images = resolve_image_paths(image_paths)

    file_name = Path(file_path).stem
    working_dir = os.path.join(output_dir, f"rag_storage_{file_name}")
    extraction_output = os.path.join(output_dir, f"{file_name}_entities.jsonl")

    print(f"\nPipeline: LangExtract + RAG-Anything")
    print(f"Inputs: {Path(file_path).name} + {len(images)} images")
    print(f"Models: Text={text_model}, Vision={vision_model}")

    # Read raw text
    raw_text = Path(file_path).read_text(encoding="utf-8")

    image_descriptions = {}

    # Step 1: Extraction
    if not query_only:
        # 1a: Text extraction
        result = step_langextract(raw_text, text_model)
        print_extractions(result)

        # 1b: Image analysis
        if images:
            image_descriptions = await step_describe_images(images, vision_model)

        step_save_extractions(result, image_descriptions, extraction_output)

        structured_text = format_extractions_for_rag(result, image_descriptions)
    else:
        structured_text = None
        if os.path.exists(extraction_output):
            print(f"Loading prior extractions from {extraction_output}")

    # Step 2: RAG Indexing
    rag = None
    if not extract_only:
        if structured_text:
            rag = await step_rag_ingest(structured_text, raw_text, text_model, working_dir)
        else:
            print("Indexing raw text only (no extractions)")
            from raganything import RAGAnything, RAGAnythingConfig
            config = RAGAnythingConfig(working_dir=working_dir)
            llm_func = get_ollama_llm_func(text_model)
            rag = RAGAnything(
                config=config,
                llm_model_func=llm_func,
                vision_model_func=llm_func,
                embedding_func=get_ollama_embedding_func(),
            )
            await rag.lightrag.ainsert(raw_text)

    # Step 3: Query
    if query and rag:
        answer = await step_rag_query(rag, query)
        print("\nAnswer:")
        print(answer)
    elif query and not rag:
        print("Cannot query in extract-only mode.")

    print("\nPipeline Complete.")


def main():
    parser = argparse.ArgumentParser(description="LangExtract + RAG-Anything Pipeline")
    parser.add_argument("file", help="Input text file")
    parser.add_argument("--images", "-i", nargs="+", help="Image files")
    parser.add_argument("--query", "-q", help="Query string")
    parser.add_argument("--text-model", "-tm", default="qwen3-vl:2b", help="Text LLM (default: qwen3-vl:2b)")
    parser.add_argument("--vision-model", "-vm", default="qwen3-vl:2b", help="Vision LLM (default: qwen3-vl:2b)")
    parser.add_argument("--extract-only", action="store_true", help="Skip RAG indexing")
    parser.add_argument("--query-only", action="store_true", help="Skip extraction")
    parser.add_argument("--output-dir", "-o", default=".", help="Output directory")

    args = parser.parse_args()

    if args.extract_only and args.query_only:
        print("Error: Cannot use extract-only and query-only together")
        sys.exit(1)

    if args.query_only and not args.query:
        print("Error: query-only requires a query")
        sys.exit(1)

    asyncio.run(
        run_pipeline(
            file_path=args.file,
            image_paths=args.images,
            query=args.query,
            text_model=args.text_model,
            vision_model=args.vision_model,
            extract_only=args.extract_only,
            query_only=args.query_only,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
