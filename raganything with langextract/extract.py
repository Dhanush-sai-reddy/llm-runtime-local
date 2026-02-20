import langextract as lx
import json
import os

# ── Configuration ──────────────────────────────────────────
OLLAMA_MODEL = "gemma3:1b"
OLLAMA_URL = "http://localhost:11434"
INPUT_FILE = "sample_data.txt"
OUTPUT_FILE = "extraction_results.jsonl"

# ── Extraction Prompt ──────────────────────────────────────
prompt = """Extract medical entities from clinical records in order of appearance.
Extract ALL of the following entity types:
- patient: patient demographics (name, DOB, gender, blood type, insurance, MRN)
- provider: doctors, nurses, therapists, social workers (name, role, specialty, department)
- diagnosis: medical diagnoses and conditions (name, status, severity, ICD category)
- medication: prescribed drugs (name, dose, route, frequency, indication)
- procedure: medical procedures and interventions (name, date, outcome, details)
- lab_result: laboratory tests and vital signs (test_name, value, unit, reference_range, status)
- facility: hospital units, departments, clinics (name, type)
- device: medical devices and implants (name, type, specifications)
- research: clinical trials and studies (name, role, PI)
- assessment_score: clinical scoring tools (tool_name, score, interpretation)
- follow_up: scheduled appointments and referrals (provider, timeframe, purpose)

Use exact text for extractions. Do not paraphrase.
Provide meaningful attributes for each entity to add context.
Extract entities from ALL patient records in the document."""

# ── Few-shot Examples ──────────────────────────────────────
examples = [
    lx.data.ExampleData(
        text="""PATIENT RECORD: MRN-99990101-0001
Patient Name: John Example
Date of Birth: 1975-05-15
Gender: Male
Blood Type: AB+

Referring Physician: Dr. Sarah Miller, MD (Cardiology, City Hospital)
Primary Care Provider: Dr. James Wilson, DO (City Family Practice)

Admission Date: 2024-06-01
Admitting Diagnosis: Unstable angina

PRESENTING COMPLAINT:
52-year-old male with chest pain on exertion for 2 weeks. BP 152/94, HR 88. ECG shows ST depression V3-V5. Troponin I: 0.08 ng/mL.

Procedure: Coronary angiography by Dr. Miller on 2024-06-02.
Findings: 80% stenosis mid-LAD. Drug-eluting stent (Resolute Onyx 3.0x18mm) placed.

Medications:
- Aspirin 81mg daily
- Ticagrelor 90mg BID
- Atorvastatin 40mg daily
- Metoprolol 50mg daily

PHQ-9 Score: 8 (mild depression)

Follow-up: Dr. Miller in 2 weeks, Dr. Wilson in 1 week.
Enrolled in HEART-OUTCOMES-2024 trial (PI: Dr. Miller).""",
        extractions=[
            lx.data.Extraction(
                extraction_class="patient",
                extraction_text="John Example",
                attributes={"field": "name", "mrn": "MRN-99990101-0001", "dob": "1975-05-15", "gender": "Male", "blood_type": "AB+"}
            ),
            lx.data.Extraction(
                extraction_class="provider",
                extraction_text="Dr. Sarah Miller, MD",
                attributes={"role": "Referring Physician", "specialty": "Cardiology", "facility": "City Hospital"}
            ),
            lx.data.Extraction(
                extraction_class="provider",
                extraction_text="Dr. James Wilson, DO",
                attributes={"role": "Primary Care Provider", "specialty": "Family Practice", "facility": "City Family Practice"}
            ),
            lx.data.Extraction(
                extraction_class="diagnosis",
                extraction_text="Unstable angina",
                attributes={"status": "acute", "type": "admitting_diagnosis"}
            ),
            lx.data.Extraction(
                extraction_class="lab_result",
                extraction_text="BP 152/94",
                attributes={"test_name": "blood_pressure", "status": "elevated"}
            ),
            lx.data.Extraction(
                extraction_class="lab_result",
                extraction_text="Troponin I: 0.08 ng/mL",
                attributes={"test_name": "troponin_i", "value": "0.08", "unit": "ng/mL"}
            ),
            lx.data.Extraction(
                extraction_class="procedure",
                extraction_text="Coronary angiography",
                attributes={"date": "2024-06-02", "provider": "Dr. Miller"}
            ),
            lx.data.Extraction(
                extraction_class="device",
                extraction_text="Resolute Onyx 3.0x18mm",
                attributes={"type": "drug-eluting stent", "location": "mid-LAD"}
            ),
            lx.data.Extraction(
                extraction_class="medication",
                extraction_text="Aspirin 81mg daily",
                attributes={"drug": "Aspirin", "dose": "81mg", "frequency": "daily"}
            ),
            lx.data.Extraction(
                extraction_class="medication",
                extraction_text="Ticagrelor 90mg BID",
                attributes={"drug": "Ticagrelor", "dose": "90mg", "frequency": "BID"}
            ),
            lx.data.Extraction(
                extraction_class="medication",
                extraction_text="Atorvastatin 40mg daily",
                attributes={"drug": "Atorvastatin", "dose": "40mg", "frequency": "daily"}
            ),
            lx.data.Extraction(
                extraction_class="medication",
                extraction_text="Metoprolol 50mg daily",
                attributes={"drug": "Metoprolol", "dose": "50mg", "frequency": "daily"}
            ),
            lx.data.Extraction(
                extraction_class="assessment_score",
                extraction_text="PHQ-9 Score: 8",
                attributes={"tool": "PHQ-9", "score": "8", "interpretation": "mild depression"}
            ),
            lx.data.Extraction(
                extraction_class="follow_up",
                extraction_text="Dr. Miller in 2 weeks",
                attributes={"provider": "Dr. Miller", "timeframe": "2 weeks"}
            ),
            lx.data.Extraction(
                extraction_class="follow_up",
                extraction_text="Dr. Wilson in 1 week",
                attributes={"provider": "Dr. Wilson", "timeframe": "1 week"}
            ),
            lx.data.Extraction(
                extraction_class="research",
                extraction_text="HEART-OUTCOMES-2024 trial",
                attributes={"name": "HEART-OUTCOMES-2024", "pi": "Dr. Miller"}
            ),
        ]
    )
]


def extract_from_file(filepath: str = INPUT_FILE) -> lx.data.AnnotatedDocument | None:
    """Run LangExtract on a text file and return the annotated document."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text_content = f.read()
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None

    print(f"Loaded text ({len(text_content)} chars) from {filepath}")
    print(f"Extracting entities using LangExtract + Ollama ({OLLAMA_MODEL})...\n")

    result = lx.extract(
        text_or_documents=text_content,
        prompt_description=prompt,
        examples=examples,
        model_id=OLLAMA_MODEL,
        model_url=OLLAMA_URL,
        fence_output=False,
        use_schema_constraints=False,
    )

    return result


def print_results(result: lx.data.AnnotatedDocument):
    """Pretty-print extraction results grouped by entity class."""
    if not result.extractions:
        print("  No extractions found.")
        return

    # Group by class
    grouped = {}
    for ext in result.extractions:
        cls = ext.extraction_class
        if cls not in grouped:
            grouped[cls] = []
        grouped[cls].append(ext)

    total = len(result.extractions)
    print(f"\n{'='*60}")
    print(f"  EXTRACTION RESULTS — {total} entities across {len(grouped)} classes")
    print(f"{'='*60}\n")

    for cls, exts in sorted(grouped.items()):
        print(f"  ┌─ {cls.upper()} ({len(exts)} entities)")
        for ext in exts:
            print(f"  │  • \"{ext.extraction_text}\"")
            if ext.attributes:
                attrs = ", ".join(f"{k}={v}" for k, v in ext.attributes.items())
                print(f"  │    └─ {attrs}")
        print(f"  └─{'─'*40}\n")


def save_results(result: lx.data.AnnotatedDocument, output: str = OUTPUT_FILE):
    """Save extraction results to JSONL."""
    lx.io.save_annotated_documents([result], output_name=output, output_dir=".")
    print(f"\nResults saved to {output}")


def extractions_to_dicts(result: lx.data.AnnotatedDocument) -> list[dict]:
    """Convert extractions to a list of dicts for graph_rag consumption."""
    entities = []
    for ext in result.extractions:
        entities.append({
            "class": ext.extraction_class,
            "text": ext.extraction_text,
            "attributes": ext.attributes if ext.attributes else {},
        })
    return entities


def main():
    result = extract_from_file()
    if result is None:
        return

    print_results(result)
    save_results(result)

    # Also save as JSON for graph_rag.py to consume
    entities = extractions_to_dicts(result)
    with open("entities.json", "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=2, ensure_ascii=False)
    print(f"Entities JSON saved to entities.json ({len(entities)} entities)")

    print("\n--- Extraction Complete ---")


if __name__ == "__main__":
    main()
