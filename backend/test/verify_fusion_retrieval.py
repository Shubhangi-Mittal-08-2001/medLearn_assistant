#!/usr/bin/env python3
"""
Fusion Retrieval Verification Script
Tests the fusion retrieval implementation in MedLearn Assistant
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_dependencies():
    """Verify all required packages are installed"""
    logger.info("=" * 60)
    logger.info("STEP 1: Verifying Dependencies")
    logger.info("=" * 60)
    
    required_packages = {
        'llama_index.core': 'llama-index-core',
        'llama_index.retrievers.bm25': 'llama-index-retrievers-bm25',
        'llama_index.vector_stores.postgres': 'llama-index-vector-stores-postgres',
        'llama_index.llms.azure_openai': 'llama-index-llms-azure-openai',
        'llama_index.embeddings.azure_openai': 'llama-index-embeddings-azure-openai',
    }
    
    all_present = True
    for module, package_name in required_packages.items():
        try:
            __import__(module)
            logger.info(f"✓ {package_name} is installed")
        except ImportError:
            logger.error(f"✗ {package_name} is NOT installed")
            all_present = False
    
    return all_present


def verify_environment_variables():
    """Verify all required environment variables are set"""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Verifying Environment Variables")
    logger.info("=" * 60)
    
    # Load .env file if it exists
    load_dotenv()
    
    required_vars = {
        # Azure OpenAI
        'AZURE_OPENAI_ENDPOINT': 'Azure OpenAI Endpoint',
        'AZURE_OPENAI_API_KEY': 'Azure OpenAI API Key',
        'AZURE_OPENAI_LLM_DEPLOYMENT': 'Azure LLM Deployment',
        'AZURE_OPENAI_EMBEDDING_DEPLOYMENT': 'Azure Embedding Deployment',
        'AZURE_OPENAI_API_VERSION': 'Azure API Version',
        
        # PostgreSQL
        'DB_HOST': 'Database Host',
        'DB_PORT': 'Database Port',
        'DB_NAME': 'Database Name',
        'DB_USER': 'Database User',
        'DB_PASSWORD': 'Database Password',
        'DB_TABLE_NAME': 'Database Table Name',
        
        # LlamaParse
        'LLAMA_CLOUD_API_KEY': 'LlamaParse API Key',
    }
    
    all_present = True
    for var_name, description in required_vars.items():
        value = os.getenv(var_name)
        if value:
            # Mask sensitive values
            if 'KEY' in var_name or 'PASSWORD' in var_name:
                display_value = value[:8] + '*' * (len(value) - 16) if len(value) > 16 else '*' * len(value)
            else:
                display_value = value
            logger.info(f"✓ {description} ({var_name}): {display_value}")
        else:
            logger.warning(f"⚠ {description} ({var_name}): NOT SET")
            all_present = False
    
    return all_present


def verify_fusion_retrieval_service():
    """Verify FusionRetrievalService can be imported and initialized"""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Verifying FusionRetrievalService")
    logger.info("=" * 60)
    
    try:
        from main.service.fusion_retrieval import FusionRetrievalService, RetrievedChunk
        logger.info("✓ FusionRetrievalService imported successfully")
        
        # Check class methods
        required_methods = ['_init_retrievers', 'query', 'retrieve']
        for method_name in required_methods:
            if hasattr(FusionRetrievalService, method_name):
                logger.info(f"✓ Method '{method_name}' exists")
            else:
                logger.error(f"✗ Method '{method_name}' NOT found")
                return False
        
        # Check RetrievedChunk dataclass
        logger.info("✓ RetrievedChunk dataclass imported successfully")
        logger.info(f"  Fields: {RetrievedChunk.__dataclass_fields__.keys()}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to import FusionRetrievalService: {e}")
        return False


def verify_rag_service():
    """Verify RAG service functions"""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Verifying RAG Service")
    logger.info("=" * 60)
    
    try:
        from main.service.rag_service import create_vector_store
        logger.info("✓ create_vector_store imported successfully")
        
        # Try to create vector store (might fail if DB not accessible)
        try:
            vector_store = create_vector_store()
            logger.info("✓ Vector store created successfully")
            logger.info(f"  Vector store type: {type(vector_store).__name__}")
            return True
        except Exception as e:
            logger.warning(f"⚠ Vector store creation failed (DB might not be accessible): {e}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Failed to import RAG service: {e}")
        return False


def verify_routes():
    """Verify API routes are configured"""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Verifying API Routes")
    logger.info("=" * 60)
    
    try:
        from main.routes.routes import router
        logger.info("✓ Routes imported successfully")
        
        # Check for fusion retrieval usage in routes
        import inspect
        source = inspect.getsource(router)
        
        if 'FusionRetrievalService' in source:
            logger.info("✓ FusionRetrievalService is used in routes")
        else:
            logger.warning("⚠ FusionRetrievalService usage not found in routes")
        
        if 'fusion_service.query' in source:
            logger.info("✓ fusion_service.query() is called in routes")
        else:
            logger.warning("⚠ fusion_service.query() call not found")
        
        if 'fusion_service.retrieve' in source:
            logger.info("✓ fusion_service.retrieve() is called in routes")
        else:
            logger.warning("⚠ fusion_service.retrieve() call not found")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to verify routes: {e}")
        return False


def verify_imports():
    """Verify all necessary imports in fusion_retrieval.py"""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Verifying Required Imports")
    logger.info("=" * 60)
    
    try:
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.vector_stores.postgres import PGVectorStore
        from llama_index.core.vector_stores import MetadataFilters
        
        logger.info("✓ All core imports available")
        logger.info("  - StorageContext")
        logger.info("  - VectorStoreIndex")
        logger.info("  - RetrieverQueryEngine")
        logger.info("  - QueryFusionRetriever")
        logger.info("  - BM25Retriever")
        logger.info("  - PGVectorStore")
        logger.info("  - MetadataFilters")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False


def print_summary(results):
    """Print verification summary"""
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    
    logger.info(f"\nPassed: {passed_tests}/{total_tests}")
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status}: {test_name}")
    
    if passed_tests == total_tests:
        logger.info("\n✓ All verifications passed! Fusion retrieval is ready to use.")
        return 0
    elif passed_tests >= total_tests - 2:  # Allow some environmental failures
        logger.info("\n⚠ Some verifications failed, but core functionality may still work.")
        logger.info("  Check environment setup and database connectivity.")
        return 1
    else:
        logger.error("\n✗ Critical verifications failed. Cannot proceed.")
        logger.error("  Install missing dependencies and set up environment variables.")
        return 1


def main():
    """Run all verification steps"""
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 58 + "║")
    logger.info("║" + " " * 14 + "Fusion Retrieval Verification" + " " * 15 + "║")
    logger.info("║" + " " * 16 + "MedLearn Assistant" + " " * 25 + "║")
    logger.info("║" + " " * 58 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    logger.info("\n")
    
    results = {
        "Dependencies": verify_dependencies(),
        "Environment Variables": verify_environment_variables(),
        "FusionRetrievalService": verify_fusion_retrieval_service(),
        "RAG Service": verify_rag_service(),
        "API Routes": verify_routes(),
        "Imports": verify_imports(),
    }
    
    exit_code = print_summary(results)
    
    logger.info("\n" + "=" * 60)
    logger.info("Next Steps:")
    logger.info("=" * 60)
    logger.info("1. Ensure PostgreSQL is running with pgvector extension")
    logger.info("2. Run database migrations")
    logger.info("3. Upload medical documents via /upload endpoint")
    logger.info("4. Test queries via POST /query endpoint")
    logger.info("5. Monitor logs for 'retrieval_type=hybrid(vector+bm25)'")
    logger.info("\nDocumentation: See FUSION_RETRIEVAL_GUIDE.md for details")
    logger.info("=" * 60 + "\n")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
