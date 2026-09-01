"""
Query Router Service
Analyzes queries and intelligently routes them to the appropriate retriever(s):
- Vector Search (semantic/conceptual queries)
- BM25 Search (exact/keyword/technical queries)
- Fusion (hybrid queries requiring both)
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Literal
from enum import Enum


class QueryType(str, Enum):
    """Classification of query types for retriever routing."""
    VECTOR_ONLY = "vector_only"      # Conceptual, semantic, understanding-based
    BM25_ONLY = "bm25_only"          # Exact keywords, technical terms, specific facts
    FUSION = "fusion"                 # Hybrid queries requiring both methods


@dataclass
class QueryAnalysis:
    """Result of query analysis for routing decisions."""
    query_type: QueryType
    confidence: float  # 0.0 to 1.0
    reasons: List[str]
    recommended_retrievers: List[str]


class QueryRouter:
    """
    Analyzes queries and determines the optimal retriever strategy.
    
    Strategy:
    - VECTOR_ONLY: Questions asking "why", "how", "explain", "describe", conceptual understanding
    - BM25_ONLY: Questions with specific terms, numbers, drug names, exact phrases, "what is"
    - FUSION: Questions combining both conceptual and factual elements
    """
    
    logger = logging.getLogger(__name__)
    
    # Patterns for vector search (conceptual/semantic)
    VECTOR_PATTERNS = [
        r'\b(how|why|explain|describe|understand|mechanism|process|relationship|difference|compare|contrast|adapt|function|role|interact|work)\b',
        r'\b(sense|perceive|respond|detect|signal|pathway|regulation)\b',
        r'\b(what.*?difference|how.*?work|why.*?occur|explain.*?mechanism)\b',
    ]
    
    # Patterns for BM25 search (exact/technical)
    BM25_PATTERNS = [
        r'\b(define|list|what is|name|specific|exact|precision|precise)\b',
        r'\b(density|range|temperature|percentage|ratio|concentration)\b',
        r'\b(receptor|fiber|neuron|synapse|muscle|reflex)\b',  # Specific anatomical terms
        r'\d+\s*(mm|cm|°C|°F|%|units|degrees)',  # Specific measurements
        r'\b(drug|medication|compound|interaction|contraindication)\b',  # Clinical terminology
        r'\b(intrafusal|extrafusal|Meissner|Pacinian|Merkel|nociceptor|mechanoreceptor)\b',  # Specific technical terms
    ]
    
    # Patterns suggesting hybrid/fusion queries
    HYBRID_PATTERNS = [
        r'\b(compare|differentiate|distinguish|map|relate|associate|connect)\b.*\b(based on|with|using)\b',
        r'\b(feedback|interaction|integration|coordination)\b',
    ]

    # The indexed learning material currently covers the somatosensory system.
    # This intentionally runs before any LLM/agent invocation: unrelated
    # questions must not be sent to the model merely because they look like a
    # valid natural-language question.
    MEDLEARN_TOPIC_PATTERN = re.compile(
        r"\b(?:somatosensory|sensory|sensation|tactile|touch|skin|"
        r"receptor|mechanoreceptor|nociceptor|thermoreceptor|proprioception|"
        r"pain|temperature|vibration|texture|merkel|meissner|pacinian|ruffini|"
        r"muscle|spindle|intrafusal|extrafusal|afferent|neuron|neural|nervous|"
        r"joint|tendon|golgi|motor|reflex|limb|grip)\b",
        re.IGNORECASE,
    )

    @classmethod
    def is_relevant_to_medlearn_documents(cls, query_text: str) -> bool:
        """Return whether a query is within the indexed MedLearn subject area.

        This is a deterministic scope gate, not an LLM classification, so it
        can safely be used before agent execution.
        """
        return bool(cls.MEDLEARN_TOPIC_PATTERN.search(query_text))
    
    @classmethod
    def analyze(cls, query_text: str) -> QueryAnalysis:
        """
        Analyze a query to determine the optimal retriever strategy.
        
        Args:
            query_text: The user's query
            
        Returns:
            QueryAnalysis with query type, confidence, reasons, and recommended retrievers
        """
        query_lower = query_text.lower()
        reasons = []
        scores = {QueryType.VECTOR_ONLY: 0.0, QueryType.BM25_ONLY: 0.0, QueryType.FUSION: 0.0}
        
        # Check for conceptual/semantic patterns
        for pattern in cls.VECTOR_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                scores[QueryType.VECTOR_ONLY] += 0.3
                reasons.append(f"Contains conceptual keyword: {pattern}")
                break
        
        # Check for exact/technical patterns
        bm25_matches = 0
        for pattern in cls.BM25_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                bm25_matches += 1
                scores[QueryType.BM25_ONLY] += 0.25
        
        if bm25_matches > 0:
            reasons.append(f"Contains {bm25_matches} technical/exact patterns")
        
        # Check for hybrid patterns
        for pattern in cls.HYBRID_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                scores[QueryType.FUSION] += 0.5
                reasons.append(f"Contains hybrid pattern requiring both methods")
                break
        
        # Check query length and complexity
        words = query_text.split()
        if len(words) > 15:
            scores[QueryType.FUSION] += 0.1
            reasons.append("Complex query (>15 words) may benefit from fusion")
        
        # Determine the winner
        max_score = max(scores.values())
        if max_score == 0:
            # Default to fusion if no patterns match
            query_type = QueryType.FUSION
            confidence = 0.3
            reasons.append("No strong patterns detected, using fusion as default")
        else:
            query_type = max(scores.items(), key=lambda x: x[1])[0]
            confidence = min(max_score, 1.0)
        
        # Map to recommended retrievers
        if query_type == QueryType.VECTOR_ONLY:
            recommended_retrievers = ["vector"]
        elif query_type == QueryType.BM25_ONLY:
            recommended_retrievers = ["bm25"]
        else:  # FUSION
            recommended_retrievers = ["vector", "bm25"]
        
        return QueryAnalysis(
            query_type=query_type,
            confidence=confidence,
            reasons=reasons,
            recommended_retrievers=recommended_retrievers,
        )
    
    @classmethod
    def log_routing_decision(cls, query_text: str, analysis: QueryAnalysis) -> None:
        """Log the routing decision with detailed information."""
        cls.logger.info(
            "query.routing query_type=%s confidence=%.2f retrievers=%s query_chars=%d",
            analysis.query_type.value,
            analysis.confidence,
            "|".join(analysis.recommended_retrievers),
            len(query_text),
        )
        for reason in analysis.reasons:
            cls.logger.debug("query.routing.reason: %s", reason)
