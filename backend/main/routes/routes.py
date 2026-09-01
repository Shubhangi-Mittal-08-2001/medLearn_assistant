"""
Routes Module
FastAPI route handlers for document upload and querying.
"""

import os
import re
import logging
import tempfile
from dotenv import load_dotenv

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks

from main.service.rag_service import (
    add_documents_to_index,
    create_vector_store,
    get_index,
    process_document,
)
from main.service.fusion_retrieval import FusionRetrievalService
from main.service.query_router import QueryRouter, QueryAnalysis
from main.service.sql_database import get_sql_database
from main.service.nl_sql_query_engine import get_nl_sql_engine
from main.service.hybrid_query_service import HybridQueryService
from main.evaluation.dataset import get_langfuse_client
from main.evaluation.dataset_evaluation import (
    evaluate_faithfulness_score,
    evaluate_answer_relevance,
)
from main.handoff.handoff_service import (
    evaluate_score,
    send_handoff_email,
    generate_handoff_reference_id,
    evaluate_explicit_user_request,
    evaluate_confidence_score,
)
import secrets
import datetime

from main.schema.models import QueryRequest,QueryResponse,UploadResponse
from main.logs.logging import get_logger

logger = get_logger(__name__)

# Load environment variables
load_dotenv()

router = APIRouter()

# Configuration from environment
MAX_FILE_SIZE_KB = int(os.getenv('MAX_UPLOAD_SIZE_KB', '200'))  # 200 KB default
MAX_FILE_SIZE = MAX_FILE_SIZE_KB * 1024
ALLOWED_EXTENSIONS = set(os.getenv('ALLOWED_FILE_TYPES', '.pdf,.docx,.doc,.txt,.xlsx').split(','))

# Images folder should be in backend directory (one level up from main/)
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_FOLDER = os.path.join(BACKEND_DIR, os.getenv('IMAGES_FOLDER', 'stored_images'))

MAX_QUERY_LENGTH = 2000
MIN_SIMILARITY_TOP_K = 1
MAX_SIMILARITY_TOP_K = 4

# Agent service will be set by app.py
_agent_service = None


def set_agent_service(agent_service):
    """Set the agent service instance for use in routes."""
    global _agent_service
    _agent_service = agent_service
    logger.info(f"Agent service initialized in routes. Service: {type(agent_service).__name__}")
    logger.debug(f"Agent initialized: {_agent_service.is_initialized() if hasattr(_agent_service, 'is_initialized') else 'unknown'}")


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check endpoint called")
    return {"status": "healthy", "service": "MedLearn AI Assistant is running ok"}


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and other security issues.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem operations
    """
    logger.debug(f"Sanitizing filename: {filename}")
    
    if not filename:
        logger.error("Filename cannot be empty")
        raise ValueError("Filename cannot be empty")
    
    filename = os.path.basename(filename)
    filename = filename.replace('/', '_').replace('\\', '_')
    filename = re.sub(r'[^\w\s.-]', '', filename)
    
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
        logger.info(f"Filename truncated to 255 characters: {filename}")
    
    logger.debug(f"Sanitized filename: {filename}")
    return filename


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a medical document for indexing with multimodal content extraction.
    
    Accepts: .pdf, .docx, .doc, .txt, .xlsx (configurable in .env)
    Max size: 200 KB (configurable in .env as MAX_UPLOAD_SIZE_KB)
    
    Extracts and processes:
    - Text content
    - Images (saved to backend/stored_images folder)
    - Tables
    
    All extracted content is indexed in pgvector with metadata.
    
    Returns:
        UploadResponse with message and number of documents indexed
    """
    logger.info(f"[UPLOAD] Starting document upload: {file.filename}")
    logger.debug(f"[UPLOAD] Content-Type: {file.content_type}, Size: {file.size if hasattr(file, 'size') else 'unknown'}")

    tmp_path = None

    try:
        # Validate filename
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            logger.warning(f"[UPLOAD] Invalid file type attempted: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported",
            )

        # Read uploaded bytes once
        content = await file.read()
        file_size = len(content)
        logger.info(f"[UPLOAD] File read: {file.filename}, Size: {file_size} bytes ({file_size/1024:.2f} KB)")

        # Validate file size (10 MB)
        if file_size > 10 * 1024 * 1024:
            logger.warning(
                f"[UPLOAD] File size exceeds limit: {file.filename} ({file_size} bytes)"
            )
            raise HTTPException(
                status_code=400,
                detail="File size exceeds 10MB limit",
            )

        # Save uploaded content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        logger.info(f"[UPLOAD] Temporary file created: {tmp_path}")
        logger.debug(f"[UPLOAD] Beginning document processing for {file.filename}")

        # Process document
        
        result = process_document(
            tmp_path,
            original_filename=file.filename,
        )

        logger.info(
            f"[UPLOAD] Document processed successfully: {result.get('document_indexed', 0)} document(s) indexed from {file.filename}"
        )
        logger.debug(f"[UPLOAD] Processing result: {result}")

        return UploadResponse(
            message=result.get("message", "Document processed successfully"),
            documents_indexed=result.get("document_indexed", 0),
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"[UPLOAD] Unexpected error while processing upload {file.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error occurred while processing the document",
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.debug(f"[UPLOAD] Cleaned up temporary file: {tmp_path}")
            except Exception as e:
                logger.warning(
                    f"[UPLOAD] Failed to delete temporary file: {tmp_path}, error: {e}"
                )


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Query unstructured medical documents using RAG with intelligent fusion retrieval.
    
    **Retrieval Architecture — Hybrid/Fusion Retrieval:**
    
    This endpoint uses a hybrid retrieval system combining two complementary approaches:
    
    1. **Vector Retrieval** (Semantic Search):
       - Captures semantic similarity and conceptual relationships
       - Effective for mechanism-oriented questions (how, why, explain)
       - Examples: "How does the nervous system sense physical changes?"
       - Uses embeddings and pgvector for similarity search
    
    2. **BM25 Retrieval** (Lexical Search):
       - Captures exact terminology and keyword matches
       - Effective for technical terms, drug names, receptor names, identifiers
       - Examples: "Define intrafusal fibers" or "What is Pacinian corpuscle?"
       - Performs full-text search on document content
    
    3. **Fusion Retrieval** (Combined):
       - Intelligently combines both approaches using Reciprocal Rank Fusion
       - Dynamically adjusts vector vs BM25 contribution based on query characteristics
       - Analyzes query patterns to determine optimal retriever selection
       - May use vector-only, BM25-only, or both retrievers depending on query type
    
    **Query Analysis & Routing:**
    - Analyzes query characteristics to determine retriever strategy
    - Pattern detection identifies question intent (conceptual vs factual)
    - Confidence score indicates reliability of retrieval strategy
    - Retrieved results are ranked by relevance fusion
    
    **Evaluation & Human Handoff:**
    - Faithfulness: Evaluates if answer is grounded in retrieved context
    - Relevance: Assesses if answer directly addresses the question
    - Confidence: LLM-based confidence in answer quality
    - Explicit Intent: Detects if user explicitly requests escalation
    - Automatic escalation to human reviewer if scores below thresholds
    
    **Returns:**
        QueryResponse with answer, source nodes, and evaluation metadata
    """
    #Implement query endpoint with handoff logic
    
    # 1. Setup: Generate session_id, initialize conversation_flow
    # 2. RAG Query: Get query engine, execute query, build full_chunks list
    # 3. Langfuse: Start span, get trace_id
    # 4. Evaluations: Run faithfulness, relevance, confidence, explicit request checks (all async)
    # 5. Handoff Decision (priority order):
    #    - Explicit request → high priority
    #    - Confidence + risk → high priority
    #    - Score-based → normal priority
    # 6. If handoff triggered:
    #    - Generate reference_id, build handoff_context with all fields
    #    - Send email via background_tasks
    #    - Return fallback message with reference_id
    # 7. Normal response: Format source_nodes (truncate to 200 chars), return answer
    # 8. Error handling: Wrap in try/except, log errors, raise HTTPException
    
    try:
        # ----------------------------------------------------
        # 0. Session ID + conversation memory
        # ----------------------------------------------------
        session_id = secrets.token_hex(8)
        conversation_flow = []

     # ----------------------------------------------------
        # 1. Query Routing + RAG Query
        # ----------------------------------------------------
        # Analyze the query to determine optimal retriever strategy
        query_analysis: QueryAnalysis = QueryRouter.analyze(request.question)
        QueryRouter.log_routing_decision(request.question, query_analysis)
        
        # Ensure the shared LLM and embedding settings are initialized before
        # constructing the fusion query engine.
        get_index()
        fusion_service = FusionRetrievalService(
            vector_store=create_vector_store(),
            table_name=os.getenv("DB_TABLE_NAME", "medical_documents_embeddings"),
            similarity_top_k=MAX_SIMILARITY_TOP_K,
        )
        
        # Use routed retriever based on query analysis
        answer = fusion_service.query_with_routing(request.question, query_analysis.recommended_retrievers)

        # Retrieve the same routed chunks for evaluation and handoff context.
        full_chunks = [
            {
                "chunk_id": None,
                "text": chunk.text,
                "source": chunk.source,
                "score": float(chunk.score) if chunk.score is not None else None,
                "metadata": {},
            }
            for chunk in fusion_service.retrieve_with_routing(request.question, query_analysis.recommended_retrievers)
        ]

        eval_context = "\n\n".join([c["text"] for c in full_chunks])
# ----------------------------------------------------
        # 2. Langfuse tracing
        # ----------------------------------------------------
        langfuse = get_langfuse_client()
        span = langfuse.start_span(name="api_rag_query_handoff", input={"question": request.question})
        trace_id = span.trace_id

        # Log routing decision to trace
        span.update(
            metadata={
                "query_type": query_analysis.query_type.value,
                "routing_confidence": query_analysis.confidence,
                "retrievers_used": "|".join(query_analysis.recommended_retrievers),
            }
        )

        # ----------------------------------------------------
        # 3. Evaluation
        # ----------------------------------------------------
        try:
            faithfulness = await evaluate_faithfulness_score(
                langfuse, trace_id, request.question, eval_context, answer
            )
        except Exception as exc:
            logger.warning("Faithfulness evaluation unavailable: %s", exc)
            faithfulness = 1.0

        try:
            relevance = await evaluate_answer_relevance(
                langfuse, trace_id, request.question, answer
            )
        except Exception as exc:
            logger.warning("Relevance evaluation unavailable: %s", exc)
            relevance = 1.0

        # LLM confidence 
        try:
            confidence_handoff = await evaluate_confidence_score(answer)
        except Exception as exc:
            logger.warning("Confidence evaluation unavailable: %s", exc)
            confidence_handoff = {"trigger": False, "confidence": 50}
        confidence_score = confidence_handoff["confidence"]

        # ----------------------------------------------------
        # 4. LLM Intent Detection 
        # ----------------------------------------------------
        try:
            explicit_handoff = await evaluate_explicit_user_request(request.question)
        except Exception as exc:
            logger.warning("Explicit user request evaluation unavailable: %s", exc)
            explicit_handoff = {"trigger": False, "reason": "evaluation unavailable"}

        # ----------------------------------------------------
        # 5. FINAL HANDOFF DECISION 
        # ----------------------------------------------------
        retrieval_empty = len(full_chunks) == 0
        score_based_handoff = evaluate_score(
            faithfulness, relevance, request.question, no_chunks=retrieval_empty
        )

        # 1. HARD TRIGGER → Explicit user request
        if explicit_handoff["trigger"]:
            handoff = explicit_handoff
            priority = "high"

        # 2. Confidence is a SECONDARY signal (only with risk)
        elif confidence_handoff["trigger"] and (
            score_based_handoff["trigger"] or retrieval_empty
        ):
            handoff = confidence_handoff
            priority = "high"

        # 3. PRIMARY trigger → Evaluation-based
        elif score_based_handoff["trigger"]:
            handoff = score_based_handoff
            priority = "normal"

        # 4. No handoff
        else:
            handoff = {"trigger": False}
            priority = None

        if handoff["trigger"]:
            reference_id = generate_handoff_reference_id()
            timestamp_utc = datetime.datetime.now(datetime.UTC).isoformat()

            # ----------------------------------------------------
            # Build FULL context payload
            # ----------------------------------------------------
            handoff_context = {
                "reference_id": reference_id,
                "trace_id": trace_id,
                "timestamp_utc": timestamp_utc,
                "session_id": session_id,
                "priority": priority,
                "trigger_reason": handoff["reason"],

                "query_history": [
                    {"role": "user", "message": request.question}
                ],

                "generated_answer": answer,

                "evaluation_scores": {
                    "faithfulness": faithfulness,
                    "relevance": relevance,
                    "confidence": confidence_score
                },

                "retrieved_chunks": full_chunks,

                "user_metadata": {
                    "email": request.user_email
                },

                "conversation_flow": conversation_flow
            }

            # ----------------------------------------------------
            # Send email async
            # ----------------------------------------------------
            background_tasks.add_task(send_handoff_email, handoff_context)

            span.update(output={"handoff": True})
            span.end()

            return QueryResponse(
                answer=(
                    "I don't have enough reliable information to answer confidently. "
                    "Your request has been escalated to a human reviewer. "
                    f"Reference ID: {reference_id}"
                ),
                source_nodes=[],
            )

        # ----------------------------------------------------
        # 6. Normal response
        # ----------------------------------------------------
        span.update(output={"handoff": False})
        span.end()

        return QueryResponse(answer=answer, source_nodes=full_chunks)

    except ValueError as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid query request")
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process query")


@router.post("/query-sql")
async def query_sql(request: QueryRequest):
    """
    Query structured clinical database using natural language to SQL conversion.
    
    **SQL Pipeline Architecture:**
    
    This endpoint converts natural language medical questions into executable SQL queries
    against the structured clinical database, ideal for precise clinical data retrieval.
    
    **Pipeline Stages:**
    
    1. **Schema Identification**: Analyze the question to identify relevant database tables
       and columns (e.g., somatosensory_receptors, drug_interactions, lab_reference_ranges)
    
    2. **SQL Generation**: Use LLM to generate accurate SQL from natural language
       - Provides schema information, table descriptions, and sample data to LLM
       - Handles complex queries with joins, aggregations, and filtering
       - Cleans up SQL response (removes markdown blocks, etc.)
    
    3. **Validation**: Verify generated SQL is syntactically correct and safe
       - Checks for injection vulnerabilities
       - Validates table and column existence
    
    4. **Execution**: Run validated SQL against PostgreSQL database
       - Tracks execution time
       - Handles errors gracefully
    
    5. **Result Formatting**: Return structured results with metadata
       - Row count and execution time
       - Column names and data types
       - Error messages if query fails
    
    **Best Use Cases:**
    - Precise, structured, computational information
    - Specific drug interactions and contraindications
    - Receptor measurements, densities, and distributions
    - Lab reference ranges and clinical thresholds
    - Aggregate queries (counts, averages, distributions)
    - Database-backed clinical facts and measurements
    
    **Example Questions:**
    - "What are the major drug interactions with warfarin?"
    - "What is the receptor density in fingertips?"
    - "Which drugs have the highest affinity for D2 receptors?"
    - "What are the lab reference ranges for cortisol levels?"
    
    **Returns:**
        JSON response with:
        - success: Boolean indicating query success
        - query: The generated SQL query
        - results: Array of result rows (each row is a dict)
        - row_count: Number of rows returned
        - execution_time_ms: Query execution time in milliseconds
        - error: Error message if query failed
    """
    try:
        logger.info(f"[SQL QUERY] Processing question: {request.question[:100]}")
        
        # Initialize SQL engine
        sql_db = get_sql_database()
        nl_sql_engine = get_nl_sql_engine(sql_db)
        
        # Step 1: Generate SQL from natural language
        sql_query, sql_result = nl_sql_engine.query(request.question)
        
        logger.info(
            f"[SQL QUERY] Generated SQL returned {sql_result.row_count} rows "
            f"in {sql_result.execution_time:.1f}ms"
        )
        
        if sql_result.error:
            logger.error(f"[SQL QUERY] Execution error: {sql_result.error}")
            return {
                "success": False,
                "error": sql_result.error,
                "query": sql_query,
            }
        
        # Step 2: Format results
        return {
            "success": True,
            "query": sql_query,
            "results": sql_result.results,
            "row_count": sql_result.row_count,
            "execution_time_ms": sql_result.execution_time,
        }
        
    except Exception as e:
        logger.exception(f"[SQL QUERY] Failed to process SQL query: {e}")
        raise HTTPException(status_code=500, detail=f"SQL query failed: {str(e)}")


@router.post("/query-hybrid")
async def query_hybrid(request: QueryRequest):
    """
    Query using hybrid approach combining RAG and SQL with intelligent routing.
    
    **Automatic Query Routing Architecture:**
    
    This endpoint intelligently routes medical questions to the appropriate information
    source(s) based on query characteristics, enabling comprehensive answers from both
    unstructured documents and structured clinical data.
    
    **Routing Logic:**
    
    The router analyzes query content to determine which information source is required:
    
    1. **SQL-Only Route**:
       - Used when the answer is primarily contained in structured clinical data
       - Examples: "What is the receptor density of D2 receptors in the striatum?"
       - Queries: specific measurements, counts, drug properties, concentrations
    
    2. **RAG-Only Route**:
       - Used when the answer requires knowledge from unstructured documents
       - Examples: "Why does D2 receptor activation affect reward processing?"
       - Queries: mechanisms, explanations, guidelines, conceptual knowledge
    
    3. **Hybrid Route (SQL + RAG)**:
       - Used when answering requires both structured data and unstructured knowledge
       - Examples: "Which D2-targeting drugs have the highest affinity, and how do their mechanisms differ?"
       - Combines precise measurements (SQL) with explanatory context (RAG)
    
    **RAG Component — Hybrid/Fusion Retrieval:**
    
    For RAG queries, uses intelligent fusion retrieval combining:
    - **Vector Search**: Semantic similarity for conceptual questions
    - **BM25 Search**: Lexical matching for technical terms and exact phrases
    - **Dynamic Adjustment**: Analyzes query characteristics to determine retriever strategy
      • Vector-only for mechanism and explanation questions (how, why)
      • BM25-only for definition and terminology questions (define, identify)
      • Fusion (both) for complex comparative queries (compare, differentiate)
    - **Reciprocal Rank Fusion**: Combines rankings from both retrievers
    
    **SQL Component — Natural Language to SQL:**
    
    For SQL queries, converts natural language to executable SQL:
    - Identifies relevant schema and tables
    - Generates SQL with schema context
    - Validates and executes against PostgreSQL
    - Returns structured results
    
    **Answer Synthesis:**
    
    For hybrid queries, synthesizes final answer by combining:
    - Structured results from SQL (precise measurements, facts)
    - Contextual knowledge from RAG (explanations, mechanisms)
    - Reasoning that integrates both perspectives
    
    **Response Metadata:**
    - hybrid_route: Which routing decision was made (sql/rag/hybrid)
    - data_sources: Which sources were actually used
    - rag_routing: Detailed fusion retrieval information (which retrievers used)
    - sql_query: The generated SQL if applicable
    - sql_results: Structured data results if applicable
    - confidence: Overall confidence score
    - reasoning: Explanation of how answer was derived
    
    **Example Questions and Routing:**
    
    1. Conceptual/RAG:
       "How does the nervous system sense physical changes?"
       → Routes to RAG with fusion retrieval
    
    2. Factual/SQL:
       "What is the precise innervation density of Merkel receptors?"
       → Routes to SQL for structured data
    
    3. Hybrid/Both:
       "Compare the adaptation speeds and receptive field sizes of Pacinian vs Meissner corpuscles, and explain why these differences matter functionally."
       → Routes to both: SQL for measurements + RAG for explanation
    
    **Returns:**
        JSON response with:
        - success: Boolean indicating query success
        - answer: Synthesized answer combining both sources
        - hybrid_route: Routing decision made (sql/rag/hybrid)
        - data_sources: Which sources were used (["sql"], ["rag"], or ["sql", "rag"])
        - confidence: Confidence score (0.0-1.0)
        - reasoning: Explanation of synthesis process
        - retrieval_details:
          - rag_routing: Fusion retrieval strategy (query_type, retrievers_used, confidence)
          - sql_query: Generated SQL if applicable
          - sql_results: Result rows if applicable
          - sql_result_count: Number of SQL result rows
          - rag_context_length: Character count of RAG context
    """
    try:
        logger.info(f"[HYBRID QUERY] Processing question: {request.question[:100]}")
        
        # Initialize services
        get_index()
        fusion_service = FusionRetrievalService(
            vector_store=create_vector_store(),
            table_name=os.getenv("DB_TABLE_NAME", "medical_documents_embeddings"),
            similarity_top_k=MAX_SIMILARITY_TOP_K,
        )
        
        sql_db = get_sql_database()
        nl_sql_engine = get_nl_sql_engine(sql_db)
        
        # Create hybrid query service
        hybrid_service = HybridQueryService(fusion_service, nl_sql_engine)
        
        # Step 1: Determine hybrid routing (sql vs rag vs hybrid)
        hybrid_route = hybrid_service.route_query(request.question)
        logger.info(f"[HYBRID QUERY] Hybrid route decision: {hybrid_route}")
        
        # Step 2: Execute hybrid query (with fusion retrieval for RAG using Vector + BM25)
        # Pass the hybrid_route to skip re-routing and use the pre-determined decision
        result = hybrid_service.query(request.question, predefined_route=hybrid_route)
        
        logger.info(
            f"[HYBRID QUERY] Completed with sources: {','.join([s.value for s in result.data_sources])} "
            f"confidence={result.confidence:.2f} route={hybrid_route}"
        )
        
        # Step 3: Format response
        return {
            "success": True,
            "answer": result.answer,
            "hybrid_route": hybrid_route,  # Shows if routed to sql, rag, or hybrid
            "data_sources": [s.value for s in result.data_sources],  # Shows which sources were actually used
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "retrieval_details": {
                "rag_routing": result.rag_routing_info,  # Fusion retrieval routing (vector-only, BM25-only, fusion)
                "sql_query": result.sql_query,
                "sql_results": result.sql_results,
                "sql_result_count": len(result.sql_results) if result.sql_results else 0,
                "rag_context_length": len(result.rag_context) if result.rag_context else 0,
            },
        }
        
    except Exception as e:
        logger.exception(f"[HYBRID QUERY] Failed to process hybrid query: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid query failed: {str(e)}")


@router.get("/schema")
async def get_database_schema():
    """
    Get the complete medical database schema information.
    
    **Purpose:**
    Returns metadata about all available tables in the structured clinical database.
    Useful for understanding available tables, columns, and data types for SQL query generation
    and for users wanting to explore the database structure.
    
    **Returns:**
        JSON response with:
        - success: Boolean indicating if schema retrieval succeeded
        - table_count: Number of tables in database
        - tables: Dictionary of table_name -> table_info
          - Each table_info contains column names, data types, constraints, and descriptions
    
    **Example Response:**
        {
            "success": true,
            "table_count": 6,
            "tables": {
                "somatosensory_receptors": "...",
                "drug_interactions": "...",
                "lab_reference_ranges": "...",
                ...
            }
        }
    """
    try:
        logger.info("[SCHEMA] Fetching database schema")
        
        sql_db = get_sql_database()
        
        tables = {}
        for table_name in sql_db.get_table_names():
            table_info = sql_db.get_table_info(table_name)
            if table_info:
                tables[table_name] = table_info
        
        return {
            "success": True,
            "table_count": len(tables),
            "tables": tables,
        }
        
    except Exception as e:
        logger.exception(f"[SCHEMA] Failed to fetch schema: {e}")
        raise HTTPException(status_code=500, detail=f"Schema retrieval failed: {str(e)}")


@router.get("/schema/{table_name}")
async def get_table_schema(table_name: str):
    """
    Get detailed schema and sample data for a specific table.
    
    **Purpose:**
    Returns comprehensive metadata about a specific table including column definitions,
    data types, constraints, and sample data. Useful for understanding table structure
    before constructing SQL queries or exploring available data.
    
    **Path Parameters:**
        table_name: Name of the table (e.g., 'somatosensory_receptors', 'drug_interactions')
    
    **Returns:**
        JSON response with:
        - success: Boolean indicating if schema retrieval succeeded
        - table_info: Column information (names, types, constraints, descriptions)
        - sample_data: Array of up to 5 sample rows from the table
        - sample_count: Number of sample rows returned
    
    **Example Request:**
        GET /schema/drug_interactions
    
    **Example Response:**
        {
            "success": true,
            "table_info": "drug_interactions table:\n  id (INTEGER, PRIMARY KEY)\n  drug1 (VARCHAR)\n  drug2 (VARCHAR)\n  severity (VARCHAR)\n  description (TEXT)",
            "sample_data": [
                {"id": 1, "drug1": "warfarin", "drug2": "aspirin", "severity": "high", ...},
                ...
            ],
            "sample_count": 5
        }
    """
    try:
        logger.info(f"[TABLE SCHEMA] Fetching schema for table: {table_name}")
        
        sql_db = get_sql_database()
        
        if not sql_db.validate_table_exists(table_name):
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
        
        table_info = sql_db.get_table_info(table_name)
        sample_data = sql_db.get_sample_data(table_name, limit=5)
        
        return {
            "success": True,
            "table_info": table_info,
            "sample_data": sample_data,
            "sample_count": len(sample_data),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[TABLE SCHEMA] Failed to fetch table schema: {e}")
        raise HTTPException(status_code=500, detail=f"Table schema retrieval failed: {str(e)}")


@router.get("/")
async def root():
    """
    Medical AI Assistant API documentation.
    
    **Service Overview:**
    
    MedLearn AI Assistant provides intelligent medical question answering through three
    complementary approaches:
    
    1. **RAG (Retrieval-Augmented Generation)** — Query unstructured documents using
       intelligent fusion retrieval combining semantic (vector) and lexical (BM25) search.
    
    2. **SQL** — Query structured clinical databases using natural language to SQL conversion.
    
    3. **Hybrid** — Combine both RAG and SQL for comprehensive answers integrating
       unstructured knowledge with structured clinical data.
    
    **Available Endpoints:**
    
    Document Management:
    - POST /upload — Upload and index medical PDF documents
    
    Query Endpoints:
    - POST /query — Query documents with fusion retrieval and evaluation
    - POST /query-sql — Query structured database using natural language
    - POST /query-hybrid — Hybrid query combining RAG + SQL with automatic routing
    
    Schema Exploration:
    - GET /schema — List all database tables and columns
    - GET /schema/{table_name} — Get detailed schema and sample data for specific table
    
    Health:
    - GET /health — Service health check
    
    **Query Examples:**
    - RAG: \"How does the nervous system sense physical changes in limbs?\"
    - SQL: \"What are the major drug interactions with warfarin?\"
    - Hybrid: \"Compare adaptation speeds of Pacinian vs Meissner corpuscles and explain why\"
    
    **For detailed endpoint documentation:**
    - See individual endpoint docstrings
    - Refer to QUERY_ROUTING_IMPLEMENTATION.md for architecture details
    - Check README.md for setup and usage instructions
    """
    return {
        "message": "MedLearn AI Assistant",
        "version": "1.0.0",
        "endpoints": {
            "POST /upload": "Upload PDF documents for indexing (RAG)",
            "POST /query": "Query RAG system for conceptual medical information",
            "POST /query-sql": "Query structured clinical database using natural language",
            "POST /query-hybrid": "Query using hybrid approach (RAG + SQL combined)",
            "GET /schema": "Get medical database schema information",
            "GET /schema/{table_name}": "Get detailed schema for specific table",
            "GET /health": "Health check endpoint"
        },
        "query_examples": {
            "rag": "How does the nervous system sense physical changes in limbs?",
            "sql": "What are the major drug interactions with warfarin?",
            "hybrid": "Compare the adaptation speeds and receptive field sizes of Pacinian vs Meissner corpuscles"
        }
    }


