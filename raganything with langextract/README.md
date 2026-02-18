# RAG-Anything + LangExtract Pipeline

A multimodal pipeline that combines LangExtract (structured entity extraction) with RAG-Anything (knowledge graph + retrieval), running locally via Ollama.

Flow:
1. Raw Text -> LangExtract -> Structured Entities
2. Images -> Vision Model -> Text Descriptions
3. Structured Data -> Knowledge Graph -> Query

## Prerequisites

1. Python 3.10+
2. Ollama running locally with:
   - `mistral` (Text LLM)
   - `qwen3-vl:2b` (Vision LLM)
   - `qwen3-embedding` (Embeddings)

## Usage

### Text Only
```bash
python rag_extract_pipeline.py sample_data.txt --query "What symptoms does the patient have?"
```

### Multimodal (Text + Images)
```bash
python rag_extract_pipeline.py sample_data.txt --images sample_vitals_chart.png sample_lab_results.png --query "What are the vitals trends?"
```

### Extract Only (Step 1)
```bash
python rag_extract_pipeline.py sample_data.txt --images *.png --extract-only
```

### Custom Models
```bash
python rag_extract_pipeline.py report.txt \
  --text-model llama3 \
  --vision-model llava \
  --query "Summarize findings"
```

## Output

- `*_entities.jsonl`: Structured entities from text
- `*_entities_images.json`: Image descriptions
- `rag_storage_*/`: Knowledge graph storage
