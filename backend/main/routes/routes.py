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

from main.service.rag_service import get_query_engine, add_documents_to_index
from main.service.document_processor import DocumentProcessor
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

logger = logging.getLogger(__name__)

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
MAX_SIMILARITY_TOP_K = 20

# Agent service will be set by app.py
_agent_service = None


def set_agent_service(agent_service):
    """Set the agent service instance for use in routes."""
    global _agent_service
    _agent_service = agent_service
    logger.info("Agent service initialized in routes")


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "MedLearn AI Assistant is running ok"}


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and other security issues.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem operations
    """
    if not filename:
        raise ValueError("Filename cannot be empty")
    
    filename = os.path.basename(filename)
    filename = filename.replace('/', '_').replace('\\', '_')
    filename = re.sub(r'[^\w\s.-]', '', filename)
    
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    
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
    file_path = None
    processor = None
    
    try:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Filename is required"
            )
        
        safe_filename = sanitize_filename(file.filename)
        file_ext = os.path.splitext(safe_filename)[1].lower()
        
        # Validate file extension
        if file_ext not in ALLOWED_EXTENSIONS:
            allowed = ', '.join(sorted(ALLOWED_EXTENSIONS))
            raise HTTPException(
                status_code=400,
                detail=f"File type '{file_ext}' not supported. Allowed types: {allowed}"
            )
        
        # Read file content
        content = await file.read()
        
        # Validate file size
        file_size_kb = len(content) / 1024
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size ({file_size_kb:.1f}KB) exceeds maximum allowed size of {MAX_FILE_SIZE_KB}KB"
            )
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(
            mode='wb',
            suffix=file_ext,
            delete=False
        ) as tmp_file:
            tmp_file.write(content)
            file_path = tmp_file.name
        
        logger.info(f"File uploaded: {safe_filename} ({file_size_kb:.1f}KB)")
        
        # Process document with multimodal extraction
        processor = DocumentProcessor(images_folder=IMAGES_FOLDER)
        text_nodes, metadata = processor.process_document(file_path, safe_filename)
        
        if not text_nodes:
            raise HTTPException(
                status_code=400,
                detail="Could not extract content from document"
            )
        
        logger.info(f"Extracted {len(text_nodes)} text node(s), "
                   f"{metadata.get('extracted_images', 0)} image(s), "
                   f"{metadata.get('total_tables', 0)} table(s) from {safe_filename}")
        
        # Add documents to index
        result = add_documents_to_index(text_nodes)
        
        return UploadResponse(
            message=result["message"],
            documents_indexed=len(text_nodes)
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file or request: {str(e)}"
        )
    except ImportError as e:
        logger.error(f"Missing dependency: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Document processing not fully configured: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {file_path}: {e}")


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
    
    try:
        # ----------------------------------------------------
        # 0. Session ID + conversation memory
        # ----------------------------------------------------
        session_id = secrets.token_hex(8)
        conversation_flow = []

     # ----------------------------------------------------
        # 1. RAG Query
        # ----------------------------------------------------
        query_engine = get_query_engine(similarity_top_k=MAX_SIMILARITY_TOP_K)
        response = query_engine.query(request.question)
        answer = str(response)

        # Build full retrieved chunks 
        full_chunks = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                full_chunks.append({
                    "chunk_id": node.node_id,
                    "text": node.text,
                    "source": node.metadata.get("file_path"),
                    "score": float(node.score),
                    "metadata": node.metadata
                })

        eval_context = "\n\n".join([c["text"] for c in full_chunks])
# ----------------------------------------------------
        # 2. Langfuse tracing
        # ----------------------------------------------------
        langfuse = get_langfuse_client()
        span = langfuse.start_span(name="api_rag_query_handoff", input={"question": request.question})
        trace_id = span.trace_id

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

