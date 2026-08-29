"""
Evaluation Dataset Creation Module
Creates and manages evaluation datasets in Langfuse.
"""

import os
from dotenv import load_dotenv
from langfuse import Langfuse

# Load environment variables
load_dotenv()


def get_langfuse_client() -> Langfuse:
    """Initialize Langfuse client using env vars."""
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not secret_key or not public_key:
        raise RuntimeError(
            "LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY must be set in environment."
        )

    return Langfuse(
        secret_key=secret_key,
        public_key=public_key,
        host=host,
    )


def create_dataset(langfuse: Langfuse, dataset_name: str):
    """
    Create an evaluation dataset for the RAG system.
    
    Creates:
    - A named dataset
    - With metadata for tracking
    """
    print(f"Creating dataset '{dataset_name}' if it does not exist...")

    langfuse.create_dataset(
        name=dataset_name,
        description="Course 3:Evaluation dataset for MedLearn AI Assistant RAG system",
        metadata={
            "sprint": "course 3: MedLearnAI Assistant",
            "type": "medlearn_qa",
            "version": "1.0",
            "knowledge_base": "medlearn",
        },
    )
    print("✓ Dataset created (or already exists).")


def add_items(langfuse: Langfuse, dataset_name: str):
    """
    Add test cases to the dataset based on the medlearn.pdf.
    
    Creates:
    - 10 test questions total
    - 3 easy factual questions
    - 4 medium procedural questions
    - 3 hard edge-case questions
    
    All questions are based on the 
    """
    
    test_cases = [
        # Easy factual questions
        {
            "input": "What is the normal range for systolic blood pressure?",
            "expected": "The normal range for systolic blood pressure is less than 120 mmHg.",
            "category": "vital_signs",
            "difficulty": "easy",
            "manual_section": "vital_signs_basics",
        },
        {
            "input": "What does BMI stand for?",
            "expected": "BMI stands for Body Mass Index, a measure of body fat based on height and weight.",
            "category": "measurements",
            "difficulty": "easy",
            "manual_section": "patient_assessment",
        },
        {
            "input": "What is the normal heart rate range for adults at rest?",
            "expected": "The normal heart rate range for adults at rest is 60 to 100 beats per minute.",
            "category": "vital_signs",
            "difficulty": "easy",
            "manual_section": "vital_signs_basics",
        },
        # Medium procedural questions
        {
            "input": "How should you perform a proper blood pressure measurement?",
            "expected": "To properly measure blood pressure: patient should sit with feet flat, arm at heart level, cuff placed on bare skin, take reading after 5 minutes of rest, and take 2 readings 1-2 minutes apart.",
            "category": "vital_signs",
            "difficulty": "medium",
            "manual_section": "vital_signs_procedures",
        },
        {
            "input": "What are the steps to calculate BMI using height and weight?",
            "expected": "BMI is calculated as: weight (kg) divided by height (m) squared, or BMI = weight(lb) / height(in)² × 703 for imperial units.",
            "category": "measurements",
            "difficulty": "medium",
            "manual_section": "patient_assessment",
        },
        {
            "input": "How should you document patient vital signs in the medical record?",
            "expected": "Document each vital sign with the exact time of measurement, include all four vitals (temperature, pulse, respiration, blood pressure), note any abnormalities, and record the patient's position and activity level during measurement.",
            "category": "documentation",
            "difficulty": "medium",
            "manual_section": "medical_records",
        },
        {
            "input": "What is the proper technique for taking oral temperature?",
            "expected": "Place thermometer under patient's tongue with lips closed, wait 3-5 minutes for mercury thermometers or until beep for digital, ensure mouth was closed for 5 minutes before measurement, and document temperature with time.",
            "category": "vital_signs",
            "difficulty": "medium",
            "manual_section": "vital_signs_procedures",
        },
        # Hard edge-case questions
        {
            "input": "A patient has a blood pressure of 140/90 mmHg. What should you consider before classifying this as hypertension?",
            "expected": "Consider: multiple readings over time (white coat effect), medication use, caffeine/nicotine use before measurement, proper technique was used, correct cuff size, patient position and activity, stress level, and time of day.",
            "category": "vital_signs",
            "difficulty": "hard",
            "manual_section": "vital_signs_interpretation",
            "edge_case": True,
        },
        {
            "input": "A patient reports significant variation in their heart rate measurements (55-95 bpm) throughout the day. What factors should you investigate?",
            "expected": "Investigate: physical activity level, stress and emotional state, caffeine/medication intake, time of day, measurement technique accuracy, underlying cardiac conditions, medications affecting heart rate, sleep quality, and fever/infection.",
            "category": "vital_signs",
            "difficulty": "hard",
            "manual_section": "vital_signs_interpretation",
            "edge_case": True,
        },
        {
            "input": "How would you handle a patient who refuses standard vital sign measurement methods?",
            "expected": "Document refusal with reason, offer alternative methods (different sites, positions, or equipment), explain clinical importance and risks, involve patient care team and physician, obtain written refusal if necessary, and document all communication and attempts in medical record.",
            "category": "documentation",
            "difficulty": "hard",
            "manual_section": "patient_care_ethics",
            "edge_case": True,
        },
    ]

    print(f"Adding {len(test_cases)} items to dataset '{dataset_name}'...")

    for idx, case in enumerate(test_cases, start=1):
        # Build metadata dict with all available fields
        metadata = {
            "category": case["category"],
            "difficulty": case["difficulty"],
            "manual_section": case.get("manual_section", "general"),
            "source": "medlearn",
        }
        # Add edge_case field if present
        if "edge_case" in case:
            metadata["edge_case"] = case["edge_case"]
        
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input={"question": case["input"]},
            expected_output=case["expected"],
            metadata=metadata,
        )
        print(f"  ✓ Added item {idx} ({case['difficulty']}): {case['input']}")

    print("✓ All items added to dataset.")


def main():
    """Main function to create the evaluation dataset."""
    DATASET_NAME = "medlearn"

    langfuse = get_langfuse_client()
    create_dataset(langfuse, DATASET_NAME)
    add_items(langfuse, DATASET_NAME)

    print("\n" + "="*60)
    print("Dataset setup completed.")
    print("="*60)
    print(f"Dataset: '{DATASET_NAME}'")
    print("Total questions: 10")
    print("  - Easy factual: 3")
    print("  - Medium procedural: 4")
    print("  - Hard edge-case: 3")
    print(f"\nYou can now see dataset '{DATASET_NAME}' in the Langfuse dashboard under 'Datasets'.")
    print("="*60)


if __name__ == "__main__":
    main()

