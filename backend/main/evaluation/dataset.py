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
    {
        "input": "What are the main components of the somatosensory system?",
        "expected": "The somatosensory system consists of sensors in the skin and sensors in our muscles, tendons, and joints. Cutaneous receptors in the skin tell us about temperature (thermoreceptors), pressure and surface texture (mechanoreceptors), and pain (nociceptors). Receptors in muscles and joints provide information about muscle length, muscle tension, and joint angles.",
        "category": "anatomy_basics",
        "difficult": "easy",
        "manual_section": "Anatomy of the Somatosensory System - Introduction",
        "source": "Somatosensory System Wikibook",
        "edge_case": False
    },
    {
        "input": "Explain the difference between rapidly adapting and slowly adapting cutaneous receptors.",
        "expected": "Rapidly adapting receptors like Meissner corpuscles and Pacinian corpuscles respond briefly to stimuli changes. Meissner corpuscles detect fine vibrations and are used for texture recognition. Pacinian corpuscles give a feeling of vibration. Slowly adapting receptors like Merkel's receptors and Ruffini corpuscles respond to sustained stimulation. Merkel's receptors are responsible for form and texture perception, while Ruffini corpuscles respond to lateral movement or stretching of skin.",
        "category": "receptor_classification",
        "difficult": "medium",
        "manual_section": "Cutaneous receptors",
        "source": "Somatosensory System Wikibook",
        "edge_case": False
    },
    {
        "input": "Where are Merkel's receptors found and what is their density in different body parts?",
        "expected": "Merkel's receptors are responsible for form and texture perception. They are present at high density in the digits and around the mouth (50/mm² of skin surface), at lower density in other glabrous surfaces, and at very low density in hairy skin. This innervations density shrinks progressively with age, so that by the age of 50, the density in human digits is reduced to 10/mm².",
        "category": "receptor_distribution",
        "difficult": "medium",
        "manual_section": "Cutaneous receptors - Merkel's receptors",
        "source": "Somatosensory System Wikibook",
        "edge_case": False
    },
    {
        "input": "What role do Meissner corpuscles play in motor activity?",
        "expected": "Sensory information from Meissner corpuscles and rapidly adapting afferents leads to adjustment of grip force when objects are lifted. These afferents respond with a brief burst of action potentials when objects move a small distance during the early stages of lifting. In response to rapidly adapting afferent activity, muscle force increases reflexively until the gripped object no longer moves. Such a rapid response to a tactile stimulus is a clear indication of the role played by somatosensory neurons in motor activity.",
        "category": "motor_control",
        "difficult": "medium",
        "manual_section": "Cutaneous receptors - Meissner corpuscles",
        "source": "Somatosensory System Wikibook",
        "edge_case": False
    },
    {
        "input": "How do nociceptors respond to different types of stimuli?",
        "expected": "Nociceptors have free nerve endings and can be either high-threshold mechanoreceptors or polymodal receptors. Polymodal receptors respond not only to intense mechanical stimuli, but also to heat and to noxious chemicals. These receptors respond to minute punctures of the epithelium, with a response magnitude that depends on the degree of tissue deformation. They also respond to temperatures in the range of 40–60°C, and change their response rates as a linear function of warming.",
        "category": "pain_receptors",
        "difficult": "hard",
        "manual_section": "Nociceptors",
        "source": "Somatosensory System Wikibook",
        "edge_case": False
    },
    {
        "input": "What are the three types of pain signals and their characteristics?",
        "expected": "Pain signals can be separated into three components: (1) First pain or cutaneous pricking pain - rapidly transmitted with high spatial resolution, well localized and easily tolerated; (2) Second pain or burning pain - much slower with highly affective component, poorly localized and poorly tolerated; (3) Third or deep pain - arising from viscera, musculature and joints, poorly localized, can be chronic and is often associated with referred pain.",
        "category": "pain_signals",
        "difficult": "hard",
        "manual_section": "Nociceptors - Pain signals",
        "source": "Somatosensory System Wikibook",
        "edge_case": False
    },
    {
        "input": "Describe the structure and function of muscle spindles.",
        "expected": "Muscle spindles are long, thin, stretch receptors scattered throughout virtually every striated muscle in the body. They consist of a few small muscle fibers (called intrafusal fibers) with a capsule surrounding the middle third of the fibers. The ends of the intrafusal fibers are attached to extrafusal fibers. The central region of each intrafusal fiber has few myofilaments and is non-contractile, but it does have one or more sensory endings applied to it. When the muscle is stretched, the central part of the intrafusal fiber is stretched and each sensory ending fires impulses.",
        "category": "proprioception",
        "difficult": "hard",
        "manual_section": "Muscle Spindles",
        "source": "Somatosensory System Wikibook",
        "edge_case": False
    },
    {
        "input": "What is the difference between alpha motor neurons and gamma neurons?",
        "expected": "The large motor neurons that supply extrafusal muscle fibers are called alpha motor neurons, while the smaller ones supplying the contractile portions of intrafusal fibers are called gamma neurons. Gamma motor neurons can regulate the sensitivity of the muscle spindle so that this sensitivity can be maintained at any given muscle length.",
        "category": "motor_innervation",
        "difficult": "medium",
        "manual_section": "Muscle Spindles - Motor innervation",
        "source": "Somatosensory System Wikibook",
        "edge_case": False
    },
    {
        "input": "What information do joint receptors signal about joint function?",
        "expected": "The joint receptors are low-threshold mechanoreceptors and have been divided into four groups. They signal different characteristics of joint function including position, movements, direction and speed of movements. The free receptors or type 4 joint receptors are nociceptors.",
        "category": "proprioception",
        "difficult": "medium",
        "manual_section": "Joint receptors",
        "source": "Somatosensory System Wikibook",
        "edge_case": False
    },
    {
        "input": "Which receptors detect vibration and what is the difference between them?",
        "expected": "Two types of receptors detect vibration: Pacinian corpuscles and Meissner corpuscles. Pacinian corpuscles are rapidly adapting deep receptors with large receptive fields and give a feeling of 'diffuse vibration' such as tapping with a pencil. Meissner corpuscles are rapidly adapting surface receptors with small receptive fields and are used for detecting an insect or a very fine vibration and for recognizing texture.",
        "category": "receptor_function",
        "difficult": "medium",
        "manual_section": "Cutaneous receptors - Table 1",
        "source": "Somatosensory System Wikibook",
        "edge_case": False
    },
    {
        "input": "How are intrafusal fibers different from extrafusal fibers in muscle spindles?",
        "expected": "Intrafusal fibers are small muscle fibers found inside muscle spindles, with their central region being non-contractile but containing sensory endings. Extrafusal fibers are the ordinary muscle fibers that form the bulk of the muscle and are responsible for muscle contraction. The ends of the intrafusal fibers are attached to extrafusal fibers, so whenever the muscle is stretched, the intrafusal fibers are also stretched.",
        "category": "muscle_anatomy",
        "difficult": "hard",
        "manual_section": "Muscle Spindles",
        "source": "Somatosensory System Wikibook",
        "edge_case": False
    },
    {
        "input": "What is the sensory output of primary and secondary muscle spindle afferents?",
        "expected": "Primary muscle spindle afferents (group Ia) encode length error and velocity of muscle stretch, providing information about how fast the muscle is being stretched and how much the stretch differs from expected. Secondary muscle spindle afferents (group II) primarily encode muscle length. Together with feedback from Golgi tendon organs about muscle force, these afferents provide comprehensive proprioceptive feedback for motor control.",
        "category": "proprioception",
        "difficult": "hard",
        "manual_section": "Muscle Spindles - Feedback loops",
        "source": "Somatosensory System Wikibook",
        "edge_case": True
    },
    {
        "input": "Explain the feedback loops involved in proprioceptive control of limb movements.",
        "expected": "Proprioceptive control involves multiple feedback loops: (1) Length and velocity feedback from muscle spindles (primary afferents) and length feedback from secondary spindle afferents inform the alpha motor neuron about muscle stretch and velocity; (2) Force feedback from Golgi tendon organs provides information about muscle force; (3) Gamma neurons can adjust the sensitivity of muscle spindles to maintain appropriate sensitivity at any muscle length; (4) These feedback signals are processed through interneurons to generate corrective motor commands that maintain load control and force control.",
        "category": "motor_control",
        "difficult": "hard",
        "manual_section": "Muscle Spindles - Feedback loops",
        "source": "Somatosensory System Wikibook",
        "edge_case": True
    },
    {
        "input": "What is the relationship between receptor density and spatial perception in glabrous skin?",
        "expected": "In glabrous (hairless) skin, receptor density correlates with spatial perception abilities. Merkel's receptors, which mediate form perception, are present at high density in the digits and around the mouth (50/mm²), enabling precise spatial detail discrimination like recognizing 'an X' in braille. At lower densities in other glabrous surfaces and at very low densities in hairy skin, spatial perception is correspondingly reduced. This density-perception relationship decreases with age, with digit density reducing to 10/mm² by age 50.",
        "category": "sensory_perception",
        "difficult": "hard",
        "manual_section": "Cutaneous receptors - Merkel's receptors",
        "source": "Somatosensory System Wikibook",
        "edge_case": True
    },
    {
        "input": "How do thermoreceptors respond to different temperatures compared to nociceptors?",
        "expected": "Non-noxious thermoreceptors display saturating responses at high temperatures, meaning their response rate plateaus. In contrast, nociceptors (polymodal receptors responding to noxious heat) respond to temperatures in the range of 40–60°C and change their response rates as a linear function of warming, continuing to increase their response proportionally with temperature increases rather than saturating.",
        "category": "temperature_sensing",
        "difficult": "hard",
        "manual_section": "Nociceptors",
        "source": "Somatosensory System Wikibook",
        "edge_case": True
    }
    ]


    print(f"Adding {len(test_cases)} items to dataset '{dataset_name}'...")

    for idx, case in enumerate(test_cases, start=1):
        # Build metadata dict with all available fields
        metadata = {
            "category": case["category"],
            "difficult": case["difficult"],
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
        print(f"  ✓ Added item {idx} ({case['difficult']}): {case['input']}")

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

