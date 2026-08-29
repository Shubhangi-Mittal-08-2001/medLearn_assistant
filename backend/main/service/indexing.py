"""
Indexing Module
Handles document loading and vector index building operations.
"""

import os
import logging
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.postgres import PGVectorStore

logger = logging.getLogger(__name__)


def load_documents():
    """
    Load documents from document folder using SimpleDirectoryReader.
    
    Returns:
        list: List of Document objects
    """
    logger.info("Loading documents from document folder...")
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    documents_folder = os.path.join(project_root, "document")
    
    # Check if document folder exists
    if not os.path.exists(documents_folder):
        raise FileNotFoundError(f"Document folder not found at {documents_folder}")
    
    logger.info(f"✓ Found document folder at {documents_folder}")
    
    # Use SimpleDirectoryReader to load all documents
    reader = SimpleDirectoryReader(input_dir=documents_folder)
    documents = reader.load_data()
    
    logger.info(f"✓ Loaded {len(documents)} document(s)")
    
    return documents


def build_index(documents, embed_model):
    """
    Build Vector Store Index from documents using PGVector.
    
    Args:
        documents: List of Document objects
        embed_model: Azure OpenAI embedding model
        
    Returns:
        VectorStoreIndex: The created index
    """
    logger.info("Building vector index with PGVector...")
    
    logger.info(f"✓ Connecting to PostgreSQL at {os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")
    
    # Initialize PGVector store
    vector_store = PGVectorStore.from_params(
        database=os.getenv('DB_NAME'),
        host=os.getenv('DB_HOST'),
        password=os.getenv('DB_PASSWORD'),
        port=os.getenv('DB_PORT'),
        user=os.getenv('DB_USER'),
        table_name=os.getenv('DB_TABLE_NAME', 'vector_store'),
        embed_dim=1536  # Azure text-embedding-3-small dimension
    )
    
    logger.info(f"✓ PGVector store initialized with table: {os.getenv('DB_TABLE_NAME', 'vector_store')}")
    
    # Create a StorageContext bound to this store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Build the index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    
    logger.info("✓ Vector index built successfully and stored in PostgreSQL")
    return index

