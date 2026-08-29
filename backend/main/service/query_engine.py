"""
Query Engine Module
Handles query engine creation from vector index.
"""

import logging

logger = logging.getLogger(__name__)


def create_query_engine(index, llm, similarity_top_k=2):
    """
    Create Query Engine from index.
    
    Args:
        index: VectorStoreIndex
        llm: Azure OpenAI LLM
        similarity_top_k: Number of top similar nodes to retrieve (default: 2)
        
    Returns:
        QueryEngine: The created query engine
    """
    logger.info("Creating query engine...")
    
    # Create query engine with similarity search
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=similarity_top_k
    )
    
    logger.info(f"✓ Query engine created (similarity_top_k={similarity_top_k})")
    
    return query_engine

