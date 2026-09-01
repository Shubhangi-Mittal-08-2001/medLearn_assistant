"""
RAG Service Module
Provides shared services for RAG pipeline initialization and management.
"""

import os
import logging
from typing import Optional, Dict
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from main.service.agent_factory import create_embel_model, create_llm
from llama_index.vector_stores.postgres import PGVectorStore
from llama_parse import LlamaParse
from main.service.query_engine import create_query_engine

from main.service.fusion_retrieval import FusionRetrievalService
from main.service.guardrails import GuardrailsValidator
from llama_index.core.schema import TextNode
from typing import List, Optional, Dict, Any, Tuple
from llama_index.core.schema import BaseNode


from .semantic_chunking import SemanticChunker
from .table_extraction import find_markdown_tables
from .table_processing import build_nodes_from_tables
from .image_extraction import extract_images_from_pdf
from .captioning import generate_caption
from .query_engine import create_query_engine
from .metadata import get_file_metadata, extract_medical_metadata_from_filename
from main.logs.logging import get_logger

logger = get_logger(__name__)

# Global instances
_index: Optional[VectorStoreIndex] = None
_query_engine_cache: Dict[int, any] = {}
_llm = None
_embed_model = None
_semantic_chunker = None


def initialize_services():
    """Initialize all services (LLM, embeddings, index, query engine)."""
    global _index, _query_engine_cache, _llm, _embed_model, _semantic_chunker
    
    if _index is not None:
        logger.info("Services already initialized, reusing existing instances")
        return
    
    logger.info("Initializing RAG services (LLM, embeddings, index)...")
    load_dotenv()
    
    try:
        # Configure Azure OpenAI
        logger.debug("Creating LLM instance...")
        _llm = create_llm()
        Settings.llm = _llm
        logger.debug("LLM instance created successfully")
        
        logger.debug("Creating embedding model instance...")
        _embed_model = create_embel_model()
        Settings.embed_model = _embed_model
        logger.debug("Embedding model instance created successfully")
        
        # Initialize semantic chunker
        logger.debug("Creating semantic chunker instance...")
        _semantic_chunker = SemanticChunker(embed_model=_embed_model)
        logger.debug("Semantic chunker instance created successfully")
        
        # Load or create index
        logger.info("Loading or creating vector index...")
        _index = load_or_create_index(_embed_model)
        
        # Initialize query engine cache
        _query_engine_cache = {}
        logger.debug("Query engine cache initialized")
        
        logger.info("✓ All RAG services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG services: {e}")
        logger.exception("RAG services initialization error trace:")
        raise

def create_vector_store():
    """Create and return a PostgreSQL vector store instance."""
    logger.debug("Creating vector store...")
    table_name = os.getenv('DB_TABLE_NAME', 'medical_documents_embeddings')
    logger.debug(f"Vector store table name: {table_name}")
    
    try:
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
        logger.info(f"✓ Vector store created successfully for table: {table_name}")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise


def load_or_create_index(embed_model) -> VectorStoreIndex:
    """Load existing index from PostgreSQL or create a new one."""
    logger.info("Loading or creating vector index...")
    
    try:
        # Get table name from environment or use default
        table_name = os.getenv('DB_TABLE_NAME', 'medical_documents_embeddings')
        
        logger.debug(f"Creating vector store for table: {table_name}")
        vector_store = create_vector_store()
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        logger.debug("Storage context created")
        
        # Try to load existing index
        try:
            logger.info(f"Attempting to load existing index from table: {table_name}...")
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
    except Exception as e:
        logger.error(f"Failed to load or create index: {e}")
        logger.exception("Index loading/creation error trace:")
        raise


def get_index() -> VectorStoreIndex:
    """Get the vector index instance."""
    logger.debug("Accessing vector index...")
    if _index is None:
        logger.info("Vector index not yet initialized, initializing services...")
        initialize_services()
    logger.debug("Vector index retrieved")
    return _index


def get_query_engine(similarity_top_k: int = 2):
    """
    Get or create a query engine instance with specified similarity_top_k.
    
    Args:
        similarity_top_k: Number of top similar nodes to retrieve
        
    Returns:
        QueryEngine instance (cached per similarity_top_k value)
    """
    logger.debug(f"Accessing query engine with similarity_top_k={similarity_top_k}")
    
    if _index is None:
        logger.info("Vector index not yet initialized, initializing services...")
        initialize_services()
    
    # Return cached query engine if available
    if similarity_top_k in _query_engine_cache:
        logger.debug(f"Returning cached query engine for similarity_top_k={similarity_top_k}")
        return _query_engine_cache[similarity_top_k]
    
    logger.info(f"Creating new query engine with similarity_top_k={similarity_top_k}")
    # Create new query engine and cache it
    query_engine = create_query_engine(_index, _llm, similarity_top_k=similarity_top_k)
    _query_engine_cache[similarity_top_k] = query_engine
    logger.info(f"✓ Query engine created and cached for similarity_top_k={similarity_top_k}")
    
    return query_engine


def get_embed_model():
    """Get the embedding model instance."""
    logger.debug("Accessing embedding model...")
    if _embed_model is None:
        logger.info("Embedding model not yet initialized, initializing services...")
        initialize_services()
    logger.debug("Embedding model retrieved")
    return _embed_model


def get_llm():
    """Get the LLM instance."""
    logger.debug("Accessing LLM instance...")
    if _llm is None:
        logger.info("LLM not yet initialized, initializing services...")
        initialize_services()
    logger.debug("LLM instance retrieved")
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


def load_documents(pdf_path: str) -> str:
    """
    Load and parse PDF document to markdown using LlamaParse.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Full markdown text from the document
    """
    logger.info(f"Loading PDF document: {pdf_path}")
    llama_parse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not llama_parse_api_key:
        logger.error("Missing LLAMA_CLOUD_API_KEY in environment")
        raise SystemExit(
            "Missing LLAMA_CLOUD_API_KEY in environment. Get one from https://cloud.llamaindex.ai/"
        )

    logger.info("Initializing LlamaParse parser...")
    parser_lp = LlamaParse(result_type="markdown", verbose=True)
    documents = parser_lp.load_data(pdf_path)

    if not documents:
        logger.error("No content returned by LlamaParse")
        raise SystemExit("No content returned by LlamaParse.")

    full_text = "\n\n".join(doc.text for doc in documents if getattr(doc, "text", ""))
    if not full_text.strip():
        logger.error("Parsed document appears empty")
        raise SystemExit("Parsed document appears empty.")

    logger.info(f"Document parsed successfully, extracted {len(full_text)} characters of text")
    return full_text

def process_document(
        pdf_path: str,
        original_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a PDF document and add it to the multi-modal index.
        
        Processes text (with semantic chunking), tables, and images.
        
        Args:
            pdf_path: Path to the PDF file (may be temporary)
            original_filename: Original filename to use in metadata
            
        Returns:
            Dictionary with processing results including node statistics
        """
        logger.info("Processing document (boilerplate): %s", pdf_path)
        
        # Ensure services are initialized
        initialize_services()

        # TODO: Process the document and index it into the vector store, then return a result dict.
        # HINT: Ensure services are initialized (LLM, embedding model, semantic chunker, etc.)
        # HINT: Build metadata using `get_file_metadata(pdf_path)`
        # HINT: If `original_filename` is provided, override `source`/`file_path` and re-extract
        #       medical tags using `extract_medical_metadata_from_filename(original_filename)`
        # HINT: Parse the PDF into plain text/markdown (extract tables/images if present)
        # HINT: Turn the parsed content into nodes/chunks and attach metadata to each node
        # HINT: Create/load the vector store using `create_vector_store(...)`
        # HINT: Create or load a `VectorStoreIndex`, then insert nodes into it
        # HINT: Return a dict like:
        #   - message: str
        #   - documents_indexed: int
        #   - optional: total_nodes/text_nodes/table_nodes/image_nodes, file_path, etc.
        # Your code here:
        
        # Get file metadata - use original filename if provided
        metadata = get_file_metadata(pdf_path)
        if original_filename:
            # Override source with original filename for better metadata
            source_name = original_filename
            metadata["source"] = original_filename
            medical_tags = extract_medical_metadata_from_filename(original_filename)
            metadata.update(medical_tags)
        else:
            source_name = metadata.get("source", pdf_path)
        
        # all_nodes can contain TextNode, ImageNode, and other BaseNode types
        # ImageNode objects are created by image_processor (same approach as demo-2)
        all_nodes: List[BaseNode] = []
        
        # Step 1: Load document with LlamaParse 
        logger.info("Step 1: Loading and parsing PDF document with LlamaParse...")
        full_text = load_documents(pdf_path)
        logger.info("Creating semantic text chunks...")

        logger.info("Step 1.1: Creating semantic text chunks...")

        text_nodes = _semantic_chunker.process(
            text=full_text,
            metadata={
                **metadata,
                "source": source_name,
            },
        )

        all_nodes.extend(text_nodes)
        logger.info(f"Document loaded successfully, text length: {len(full_text)} characters")
        
        
        # Step 2: Extract and process tables 
        logger.info("Step 2: Extracting markdown tables from document...")
        table_strings = find_markdown_tables(full_text)
        if table_strings:
            logger.info(f"Found {len(table_strings)} table(s) in the document")
            for idx, (start, end, table_md) in enumerate(table_strings):
                logger.debug(f"Processing table {idx + 1}/{len(table_strings)}")
                table_nodes = build_nodes_from_tables(
                source_name=source_name,
                table_markdowns=[table_md],
                llm=_llm,
                additional_metadata={
                **metadata,
                "table_index": idx,
                "start": start,
                "end": end,
                }
            )
                all_nodes.extend(table_nodes)
                logger.debug(f"Table {idx + 1} processed, created {len(table_nodes)} node(s)")
        else:
            logger.info("No tables found in document")
        
        # Step 3: Extract and process images
        # Note: Temporary directory is only used for initial extraction from PDF.
        # The image_processor.process_from_path() copies images to permanent storage
        # (stored_images/) and ImageNode objects use the permanent path, not temp path.
        logger.info("Step 3: Extracting images from PDF...")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            images = extract_images_from_pdf(pdf_path, tmpdir)
            if not images:
                logger.info("No images found in document.")

            else:
                logger.info(f"Extracted {len(images)} images from PDF")

                for img_info in images:
                    try:
                        logger.debug(f"Processing image: {img_info['path']} (page={img_info.get('page')}, index={img_info.get('image_index')})")
                        # process_from_path() receives temp path but copies to permanent storage
                        # and returns ImageNode with permanent path
                        caption = generate_caption(img_info["path"])

                        image_node = TextNode(
                            text=caption,
                            metadata={
                                **metadata,
                                "source": source_name,
                                "content_type": "image_caption",
                                "page": img_info.get("page"),
                                "image_index": img_info.get("image_index"),
                            },
                        )

                        all_nodes.append(image_node)
                        logger.debug(f"Image processed, created 1 node with permanent paths")
                    except Exception as e:
                        logger.error(f"Failed to process image {img_info['path']}: {e}", exc_info=True)
            
        # Calculate statistics
        text_count = sum(1 for n in all_nodes if n.metadata.get('content_type') == 'text')
        table_count = sum(1 for n in all_nodes if n.metadata.get('content_type') == 'table_summary')
        image_count = sum(1 for n in all_nodes if n.metadata.get('content_type') == 'image_caption')
        
        stats = {
            "total_nodes": len(all_nodes),
            "text_nodes": text_count,
            "table_nodes": table_count,
            "image_nodes": image_count,
        }
        
        logger.info(f"Total nodes created: {len(all_nodes)} (text: {text_count}, tables: {table_count}, images: {image_count})")
        
        # Create or update index
        logger.info("Step 5: Creating PostgreSQL vector store and indexing documents...")

        vector_store = create_vector_store()

        if _index is None:
            logger.info("Creating new VectorStoreIndex...")
            index = VectorStoreIndex(
                nodes=[],
                vector_store=vector_store,
                embed_model=_embed_model,
            )
        else:
            index = _index

        logger.info(f"Inserting {len(all_nodes)} nodes into PGVector...")
        index.insert_nodes(all_nodes)

        logger.info("Nodes inserted successfully.")
        logger.info("Document processing completed successfully")
        
        # Use original filename in response if provided, otherwise use pdf_path
        response_file_path = original_filename if original_filename else pdf_path
        
        return {
            "file_path": response_file_path,
            "status": "success",
            "document_indexed": len(all_nodes),
            "message": f"Document processed and indexed successfully: {len(all_nodes)} nodes created",
            **stats
        }
    