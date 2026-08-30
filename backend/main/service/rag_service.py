"""
RAG Service Module
Provides shared services for RAG pipeline initialization and management.
"""

import os
import logging
from typing import Optional, Dict
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from main.service.query_engine import create_query_engine

from main.service.fusion_retrieval import FusionRetrievalService
from main.service.guardrails import GuardrailsValidator

logger = logging.getLogger(__name__)

# Global instances
_index: Optional[VectorStoreIndex] = None
_query_engine_cache: Dict[int, any] = {}
_llm = None
_embed_model = None


def initialize_services():
    """Initialize all services (LLM, embeddings, index, query engine)."""
    global _index, _query_engine_cache, _llm, _embed_model
    
    if _index is not None:
        logger.info("Services already initialized, reusing existing instances")
        return
    
    load_dotenv()
    
    # Configure Azure OpenAI
    logger.info("Initializing Azure OpenAI services...")
    _llm = AzureOpenAI(
        model=os.getenv('AZURE_OPENAI_LLM_MODEL', 'gpt-4o-mini'),
        deployment_name=os.getenv('AZURE_OPENAI_LLM_DEPLOYMENT'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview'),
        temperature = 1.0
    )
    
    _embed_model = AzureOpenAIEmbedding(
        model=os.getenv('AZURE_OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
        deployment_name=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
    )
    
    Settings.llm = _llm
    Settings.embed_model = _embed_model
    
    # Load or create index
    _index = load_or_create_index(_embed_model)
    
    # Initialize query engine cache
    _query_engine_cache = {}
    
    logger.info("✓ All services initialized successfully")


def load_or_create_index(embed_model) -> VectorStoreIndex:
    """Load existing index from PostgreSQL or create a new one."""
    logger.info("Loading or creating vector index...")
    
    # Get table name from environment or use default
    table_name = os.getenv('DB_TABLE_NAME', 'medical_documents_embeddings')
    
    # Initialize PGVector store
    vector_store = PGVectorStore.from_params(
        database=os.getenv('DB_NAME'),
        host=os.getenv('DB_HOST'),
        password=os.getenv('DB_PASSWORD'),
        port=os.getenv('DB_PORT'),
        user=os.getenv('DB_USER'),
        table_name=table_name,
        embed_dim=1536
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Try to load existing index
    try:
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )
        logger.info(f"✓ Loaded existing index from PostgreSQL (table: {table_name})")
        return index
    except Exception as e:
        logger.info(f"Could not load existing index: {e}")
        logger.info(f"Creating new empty index in table: {table_name}...")
        index = VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            embed_model=embed_model
        )
        logger.info(f"✓ Created new empty index in table: {table_name}")
        return index


def get_index() -> VectorStoreIndex:
    """Get the vector index instance."""
    if _index is None:
        initialize_services()
    return _index


def get_query_engine(similarity_top_k: int = 2):
    """
    Get or create a query engine instance with specified similarity_top_k.
    
    Args:
        similarity_top_k: Number of top similar nodes to retrieve
        
    Returns:
        QueryEngine instance (cached per similarity_top_k value)
    """
    if _index is None:
        initialize_services()
    
    # Return cached query engine if available
    if similarity_top_k in _query_engine_cache:
        return _query_engine_cache[similarity_top_k]
    
    # Create new query engine and cache it
    query_engine = create_query_engine(_index, _llm, similarity_top_k=similarity_top_k)
    _query_engine_cache[similarity_top_k] = query_engine
    logger.debug(f"Cached query engine for similarity_top_k={similarity_top_k}")
    
    return query_engine


def get_embed_model():
    """Get the embedding model instance."""
    if _embed_model is None:
        initialize_services()
    return _embed_model


def get_llm():
    """Get the LLM instance."""
    if _llm is None:
        initialize_services()
    return _llm


def add_documents_to_index(documents):
    """Add new documents to the existing index."""
    global _index, _query_engine_cache
    
    if _index is None:
        initialize_services()
    
    logger.info(f"Adding {len(documents)} document(s) to index...")
    
    for i, doc in enumerate(documents, 1):
        try:
            _index.insert(doc)
            logger.info(f"  Inserted document {i}/{len(documents)}")
        except Exception as e:
            logger.error(f"Error inserting document {i}: {e}")
            raise
    
    logger.info("✓ Documents added to index successfully")
    
    # Clear query engine cache since index has been updated
    _query_engine_cache.clear()
    logger.debug("Cleared query engine cache after index update")
    
    return {"message": f"Successfully indexed {len(documents)} document(s)"}