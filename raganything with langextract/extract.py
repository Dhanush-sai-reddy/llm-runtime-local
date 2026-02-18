import langextract as lx

# Define the prompt and extraction rules for medical records
prompt = """Extract medical entities from clinical notes in order of appearance.
Extract patient demographics, symptoms, vitals, assessments, and plan items.
Use exact text for extractions. Do not paraphrase or overlap entities.
Provide meaningful attributes for each entity to add context."""

# Provide a high-quality example to guide the model
examples = [
    lx.data.ExampleData(
        text="Patient Name: Jane Smith\nDate of Birth: 1990-01-20\nDate of Visit: 2023-09-15\n\nSubjective:\nPatient complains of sore throat and dry cough for 2 days.\n\nObjective:\nBP 120/80, HR 72.\nThroat: Erythematous.\n\nAssessment:\n1. Acute pharyngitis.\n\nPlan:\n1. Prescribe Amoxicillin 500mg three times daily.",
        extractions=[
            lx.data.Extraction(
                extraction_class="patient_info",
                extraction_text="Jane Smith",
                attributes={"field": "name"}
            ),
            lx.data.Extraction(
                extraction_class="patient_info",
                extraction_text="1990-01-20",
                attributes={"field": "date_of_birth"}
            ),
            lx.data.Extraction(
                extraction_class="patient_info",
                extraction_text="2023-09-15",
                attributes={"field": "visit_date"}
            ),
            lx.data.Extraction(
                extraction_class="symptom",
                extraction_text="sore throat",
                attributes={"duration": "2 days"}
            ),
            lx.data.Extraction(
                extraction_class="symptom",
                extraction_text="dry cough",
                attributes={"duration": "2 days"}
            ),
            lx.data.Extraction(
                extraction_class="vitals",
                extraction_text="BP 120/80, HR 72",
                attributes={"summary": "normal range"}
            ),
            lx.data.Extraction(
                extraction_class="assessment",
                extraction_text="Acute pharyngitis",
                attributes={"status": "acute"}
            ),
            lx.data.Extraction(
                extraction_class="plan",
                extraction_text="Prescribe Amoxicillin 500mg three times daily",
                attributes={"action": "prescribe_medication", "details": "Amoxicillin 500mg TID"}
            ),
        ]
    )
]


def main():
    # Load sample data
    try:
        with open("sample_data.txt", "r") as f:
            text_content = f.read()
    except FileNotFoundError:
        print("Error: sample_data.txt not found.")
        return

    print(f"Loaded medical record ({len(text_content)} chars)...")

    # Run extraction using local Ollama model (Gemma 3 1B)
    print("Extracting data using LangExtract + Ollama (gemma3:1b)...\n")

    try:
        result = lx.extract(
            text_or_documents=text_content,
            prompt_description=prompt,
            examples=examples,
            model_id="gemma3:1b",
            model_url="http://localhost:11434",
            fence_output=False,
            use_schema_constraints=False,
        )

        # Output results
        print("\n--- Extracted Entities ---\n")
        if result.extractions:
            for ext in result.extractions:
                print(f"  [{ext.extraction_class}] \"{ext.extraction_text}\"")
                if ext.attributes:
                    for k, v in ext.attributes.items():
                        print(f"    {k}: {v}")
                print()
        else:
            print("  No extractions found.")

        # Save results to JSONL
        lx.io.save_annotated_documents(
            [result], output_name="extraction_results.jsonl", output_dir="."
        )
        print("Results saved to extraction_results.jsonl")

        print("\n--- Extraction Complete ---")

    except Exception as e:
        print(f"An error occurred during extraction: {e}")


if __name__ == "__main__":
    main()
