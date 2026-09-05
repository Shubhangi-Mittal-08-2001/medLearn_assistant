"""Metadata Service Module
Handles medical education metadata extraction from filenames and file paths.

This module provides:
- Medical metadata extraction from filenames
- File metadata construction (source, extension, size, tags)
- Support for anatomy, physiology, pathology, and clinical content
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Medical Education Metadata Categories
MEDICAL_SPECIALTIES = [
    "anatomy",
    "physiology",
    "pathology",
    "pharmacology",
    "clinical-diagnosis",
    "surgery",
    "neurology",
    "cardiology",
    "pulmonology",
    "gastroenterology",
    "nephrology",
    "endocrinology",
    "immunology",
    "microbiology",
    "oncology"
]

BODY_SYSTEMS = [
    "nervous-system",
    "cardiovascular-system",
    "respiratory-system",
    "digestive-system",
    "urinary-system",
    "musculoskeletal-system",
    "integumentary-system",
    "endocrine-system",
    "lymphatic-system",
    "reproductive-system"
]

CONTENT_TYPES = [
    "textbook",
    "research-paper",
    "case-study",
    "clinical-guideline",
    "quiz",
    "lecture-notes",
    "procedure-guide",
    "reference-table"
]

ANATOMICAL_REGIONS = [
    "head-neck",
    "thorax",
    "abdomen",
    "pelvis",
    "upper-limb",
    "lower-limb",
    "spine",
    "surface-anatomy",
    "cross-sectional-anatomy"
]

RECEPTOR_TYPES = [
    "mechanoreceptor",
    "nociceptor",
    "thermoreceptor",
    "proprioceptor",
    "chemoreceptor",
    "photoreceptor"
]

CLINICAL_TOPICS = [
    "disease-mechanism",
    "symptoms-signs",
    "diagnosis-tests",
    "treatment-options",
    "drug-interactions",
    "adverse-effects",
    "contraindications",
    "complications"
]

DIFFICULTY_LEVELS = [
    "introductory",
    "intermediate",
    "advanced",
    "expert"
]


def extract_medical_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """
    Extract medical education metadata from filename.

    Automatically assigns metadata based on filename patterns:
    - Medical specialties: anatomy, physiology, pathology, pharmacology, etc.
    - Body systems: nervous-system, cardiovascular-system, etc.
    - Content types: textbook, research-paper, case-study, clinical-guideline, quiz
    - Anatomical regions: head-neck, thorax, abdomen, etc.
    - Receptor types: mechanoreceptor, nociceptor, etc.
    - Clinical topics: disease-mechanism, symptoms-signs, diagnosis-tests, etc.
    - Difficulty levels: introductory, intermediate, advanced, expert

    Args:
        filename: Name of the file (without path)

    Returns:
        Dictionary containing extracted medical metadata
    """
    filename_lower = filename.lower()
    metadata: Dict[str, Any] = {}

    # Extract medical specialty
    for specialty in MEDICAL_SPECIALTIES:
        if specialty in filename_lower or specialty.replace("-", "_") in filename_lower:
            metadata["medical_specialty"] = specialty
            break

    # Extract body system
    for system in BODY_SYSTEMS:
        if system in filename_lower or system.replace("-", "_") in filename_lower:
            metadata["body_system"] = system
            break

    # Extract content type
    for content_type in CONTENT_TYPES:
        if content_type in filename_lower or content_type.replace("-", "_") in filename_lower:
            metadata["content_type"] = content_type
            break

    # Extract anatomical region
    for region in ANATOMICAL_REGIONS:
        if region in filename_lower or region.replace("-", "_") in filename_lower:
            metadata["anatomical_region"] = region
            break

    # Extract receptor type
    for receptor in RECEPTOR_TYPES:
        if receptor in filename_lower or receptor.replace("-", "_") in filename_lower:
            metadata["receptor_type"] = receptor
            break

    # Extract clinical topic
    for topic in CLINICAL_TOPICS:
        if topic in filename_lower or topic.replace("-", "_") in filename_lower:
            metadata["clinical_topic"] = topic
            break

    # Extract difficulty level
    for level in DIFFICULTY_LEVELS:
        if level in filename_lower or level.replace("-", "_") in filename_lower:
            metadata["difficulty_level"] = level
            break

    logger.debug(f"Extracted medical metadata from '{filename}': {metadata}")
    return metadata


def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a file including medical education metadata.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary containing file metadata and medical education tags
    """
    path = Path(file_path)
    filename = path.name

    # Base file metadata
    metadata: Dict[str, Any] = {
        "source": filename,
        "file_path": str(path),
        "file_extension": path.suffix.lower(),
        "file_size": path.stat().st_size if path.exists() else 0,
    }

    # Extract medical education metadata from filename
    medical_metadata = extract_medical_metadata_from_filename(filename)
    metadata.update(medical_metadata)

    return metadata