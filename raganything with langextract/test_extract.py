
import langextract as lx
from types import SimpleNamespace
import traceback

EXTRACTION_PROMPT = """Extract medical entities from clinical notes in order of appearance.
Extract patient demographics, symptoms, vitals, assessments, and plan items.
Use exact text for extractions."""

EXTRACTION_EXAMPLES = [] # Empty for simplicity or minimal example

try:
    print("Starting extraction test...")
    result = lx.extract(
        text_or_documents="Patient John Doe has a fever.",
        prompt_description=EXTRACTION_PROMPT,
        examples=EXTRACTION_EXAMPLES,
        model_id="qwen3-vl:2b",
        model_url="http://localhost:11434",
        fence_output=False,
        use_schema_constraints=False,
    )
    print("Extraction success!")
    print(result)
except Exception as e:
    print("Extraction failed!")
    traceback.print_exc()
