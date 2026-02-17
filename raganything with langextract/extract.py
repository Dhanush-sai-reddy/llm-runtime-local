import os
from langextract import Extract
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List, Optional

# Define the schema for extraction
class MedicalAssessment(BaseModel):
    condition: str = Field(description="The diagnosed or suspected condition.")
    status: str = Field(description="Status of the condition (e.g., Acute, Chronic, Suspected).")

class MedicalPlan(BaseModel):
    action: str = Field(description="The action to be taken (e.g., Order labs, Prescribe medication).")
    details: Optional[str] = Field(description="Details of the action (e.g., dosage, specific labs).")

class MedicalRecord(BaseModel):
    patient_name: str = Field(description="Name of the patient.")
    dob: str = Field(description="Date of birth of the patient.")
    visit_date: str = Field(description="Date of the visit.")
    symptoms: List[str] = Field(description="List of symptoms reported by the patient.")
    vitals: str = Field(description="Vital signs summary.")
    assessments: List[MedicalAssessment] = Field(description="List of medical assessments.")
    plan: List[MedicalPlan] = Field(description="List of plan items.")

def main():
    # Load sample data
    try:
        with open("sample_data.txt", "r") as f:
            text_content = f.read()
    except FileNotFoundError:
        print("Error: sample_data.txt not found.")
        return

    print(f"Loaded medical record ({len(text_content)} chars)...")

    # Initialize Local Gemma 3 Model via Ollama
    print("Initializing Gemma 3 (1B)")
    llm = ChatOllama(model="gemma3:1b", temperature=0)

    # Initialize LangExtract
    extractor = Extract(llm=llm, schema=MedicalRecord)

    # Run extraction
    print("Extracting data using LangExtract")
    
    try:
        result = extractor.extract(text_content)
        
        # Output results
        print("\n--- Extracted Data (JSON Structure) ---\n")
        print(result.json(indent=2))
        print("\n--- Extraction Complete ---")
        
    except Exception as e:
        print(f"An error occurred during extraction: {e}")

if __name__ == "__main__":
    main()
