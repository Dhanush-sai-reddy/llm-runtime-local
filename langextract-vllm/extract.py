import os
import json
import langextract as lx
import yaml
from pathlib import Path

# --- CONFIGURATION (Load from YAML) ---
with open("settings.yaml", "r") as f:
    config = yaml.safe_load(f)

VLLM_URL = config.get("vllm_url", "http://localhost:8000/v1")
VLLM_MODEL = config.get("vllm_model", "google/gemma-3-1b-it")
INPUT_DIR = config.get("input_directory", "./")
OUTPUT_FILE = config.get("output_file", "extraction_results.json")

prompt = """
Extract all entities from the text.
Find the following:
- patient: name, DOB, mrn
- diagnosis: condition, status
- medication: name, dose, frequency
- follow_up: provider, timeframe
"""

def extract_all():
    all_results = []
    
    # 1. Grab every text file you have
    text_files = list(Path(INPUT_DIR).glob("*.txt"))
    if not text_files:
        print("No .txt files found to process.")
        return

    print(f"Found {len(text_files)} files. Starting extraction with vLLM...")

    # 2. Loop through and extract from each one
    for file_path in text_files:
        print(f"Processing: {file_path.name}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            extracted = lx.extract(
                text_or_documents=text,
                prompt_description=prompt,
                model_id=VLLM_MODEL,
                model_url=VLLM_URL
            )

            if extracted and extracted.extractions:
                for ext in extracted.extractions:
                    all_results.append({
                        "file": file_path.name,
                        "type": ext.extraction_class,
                        "text": ext.extraction_text,
                        "details": ext.attributes
                    })
        except Exception as e:
            print(f"Error extracting from {file_path.name}: {e}")

    # 3. Save everything to a clean JSON file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(all_results, out, indent=2, ensure_ascii=False)
        
    print(f"Done! Extracted {len(all_results)} total entities. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_all()
