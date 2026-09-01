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
    """Query RAG system with evaluation and human handoff logic."""
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
    
    # Prefer the agentic hybrid service.  It exposes the tool trace to clients and
    # lets a failed SQL/MCP/vector dependency degrade independently.
    if _agent_service is not None and hasattr(_agent_service, "execute_query"):
        try:
            result = await _agent_service.execute_query(request.question)
            return QueryResponse(
                answer=result.answer,
                source_nodes=[],
                tools_used=result.tools_used,
                tool_errors=result.tool_errors,
                routing=result.routing,
            )
        except Exception as exc:
            logger.exception("Hybrid agent query failed; using legacy retrieval: %s", exc)

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


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "MedLearn AI Assistant",
        "version": "1.0.0",
        "endpoints": {
            "POST /upload": "Upload documents for indexing",
            "POST /query": "Query the RAG system",
            "GET /health": "Health check"
        }
    }
