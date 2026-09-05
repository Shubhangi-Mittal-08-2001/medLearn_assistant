"""
Query Engine Module
Handles query engine creation from vector index.
"""

import logging
from main.logs.logging import get_logger

logger = get_logger(__name__)


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
    logger.info(f"Creating query engine with similarity_top_k={similarity_top_k}...")
    
    try:
        # Create query engine with similarity search
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=similarity_top_k
        )
        
        logger.info(f"✓ Query engine created successfully (similarity_top_k={similarity_top_k})")
        logger.debug(f"Query engine type: {type(query_engine).__name__}")
        
        return query_engine
    except Exception as e:
        logger.error(f"Failed to create query engine: {e}")
        logger.exception("Query engine creation error trace:")
        raise