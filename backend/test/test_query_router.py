#!/usr/bin/env python3
"""
Test Query Router with Sample Queries

Tests the QueryRouter against the sample queries to show:
1. Which retriever(s) are selected for each query
2. The confidence and reasoning behind each decision
3. Detailed logging of the routing decision
"""

import logging
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from main.service.query_router import QueryRouter, QueryType
from main.logs.logging import get_logger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger(__name__)

# Sample queries from backend/sample_queries
VECTOR_QUERIES = [
    "How does the nervous system sense physical changes in limbs, muscles, and joints?",
    "What happens to our tactile perception of textures and fine details as we grow older?",
    "Explain the biological mechanism behind adjusting our hand grip dynamically when lifting objects.",
]

BM25_QUERIES = [
    "What is the precise innervation density of Merkel's receptors per square millimeter in human digits at the age of 50?",
    "Define the explicit difference between intrafusal fibers and extrafusal fibers in muscle spindles.",
    "What exact temperature range in degrees Celsius triggers a linear response rate change in polymodal nociceptors?",
]

FUSION_QUERIES = [
    "Compare the adaptation speeds and receptive field sizes of Pacinian corpuscles versus Meissner corpuscles, mapping them to real-world tactile examples.",
    "Differentiate between first, second, and third pain signals based on their transmission speeds, ease of localization, and anatomical origins.",
    "How do alpha motor neurons, gamma motor neurons, and Golgi tendon organs interact to create feedback loops for muscle length and velocity?",
]


def print_result(query: str, analysis) -> None:
    """Pretty print query routing analysis."""
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'-'*80}")
    print(f"Query Type:       {analysis.query_type.value.upper()}")
    print(f"Confidence:       {analysis.confidence:.2%}")
    print(f"Retrievers Used:  {' + '.join(analysis.recommended_retrievers).upper()}")
    print(f"Reasoning:")
    for reason in analysis.reasons:
        print(f"  • {reason}")
    print()


def main():
    """Run query routing tests."""
    print("\n" + "="*80)
    print("MEDLEARN ASSISTANT - QUERY ROUTER TEST")
    print("="*80)

    # Test Vector Search Queries
    print("\n\n" + "█"*80)
    print("█ VECTOR SEARCH QUERIES (Conceptual/Semantic)")
    print("█"*80)
    print("\nExpected: Only VECTOR retriever should be selected")
    print("Reason: These queries ask 'how' and 'why' about mechanisms and processes")
    
    for query in VECTOR_QUERIES:
        analysis = QueryRouter.analyze(query)
        print_result(query, analysis)
        QueryRouter.log_routing_decision(query, analysis)
        
        # Verify expectation
        if analysis.query_type == QueryType.VECTOR_ONLY:
            print("✓ CORRECT - Vector-only retriever selected")
        else:
            print(f"⚠ UNEXPECTED - Expected VECTOR_ONLY but got {analysis.query_type.value}")

    # Test BM25 Queries
    print("\n\n" + "█"*80)
    print("█ BM25 QUERIES (Exact/Technical)")
    print("█"*80)
    print("\nExpected: Only BM25 retriever should be selected")
    print("Reason: These queries ask for exact definitions, numbers, and technical specifications")
    
    for query in BM25_QUERIES:
        analysis = QueryRouter.analyze(query)
        print_result(query, analysis)
        QueryRouter.log_routing_decision(query, analysis)
        
        # Verify expectation
        if analysis.query_type == QueryType.BM25_ONLY:
            print("✓ CORRECT - BM25-only retriever selected")
        else:
            print(f"⚠ UNEXPECTED - Expected BM25_ONLY but got {analysis.query_type.value}")

    # Test Fusion Queries
    print("\n\n" + "█"*80)
    print("█ FUSION QUERIES (Hybrid)")
    print("█"*80)
    print("\nExpected: FUSION (both retrievers) should be selected")
    print("Reason: These queries combine comparison/differentiation with specific terms")
    
    for query in FUSION_QUERIES:
        analysis = QueryRouter.analyze(query)
        print_result(query, analysis)
        QueryRouter.log_routing_decision(query, analysis)
        
        # Verify expectation
        if analysis.query_type == QueryType.FUSION:
            print("✓ CORRECT - Fusion retriever selected")
        else:
            print(f"⚠ UNEXPECTED - Expected FUSION but got {analysis.query_type.value}")

    print("\n\n" + "="*80)
    print("QUERY ROUTING TEST COMPLETE")
    print("="*80)
    print("\nKey Insights:")
    print("  • Vector Search: Best for conceptual understanding (how, why, explain)")
    print("  • BM25 Search:   Best for exact facts (define, list, specific numbers)")
    print("  • Fusion:        Best for complex analysis (compare, differentiate, interact)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
