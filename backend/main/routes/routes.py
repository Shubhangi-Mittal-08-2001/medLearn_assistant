"""
Routes Module
FastAPI route handlers for document upload and querying.
"""

import os
import re
import logging
import tempfile
import time
import json
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends, status
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer

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
    CONFIDENCE_THRESHOLD,
    evaluate_score,
    send_handoff_email,
    generate_handoff_reference_id,
    is_explicit_handoff_request,
    classify_medical_intent,
    evaluate_confidence_score,
)
import secrets
import datetime

from main.schema.models import (
    NodeResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
    ValidationResults,
)
from main.logs.logging import get_logger
from main.service.guardrails import GuardrailsValidator
from main.security.security import validate_token, get_authenticated_user

logger = get_logger(__name__)

# Load environment variables
load_dotenv()

# Authentication configuration
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"
logger.info(f"Authentication enabled: {AUTH_ENABLED}")


async def get_current_user(
    token: Optional[str] = Depends(OAuth2PasswordBearer(tokenUrl="token", auto_error=False))
) -> Dict[str, Any]:
    """
    Get authenticated user. Raises 401 if not authenticated or AUTH_ENABLED=true but token missing.
    
    In production (AUTH_ENABLED=true):
    - Requires valid JWT token
    - Returns user_id, email, role
    
    In development (AUTH_ENABLED=false):
    - Returns mock user with role='user'
    """
    if not AUTH_ENABLED:
        # Development mode: return mock user
        logger.debug("Using mock user (auth disabled)")
        return {
            "user_id": "dev-user",
            "email": "dev@example.com",
            "role": "user",
            "iss": "dev-mode",
            "aud": "dev-mode",
        }
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Production: validate JWT and extract user info
    try:
        payload = validate_token(token)
        user_role = payload.get("https://medlearn.local/roles", ["user"])
        if isinstance(user_role, str):
            user_role = [user_role]
        role = "admin" if any(r.lower() == "admin" for r in user_role) else "user"
        
        return {
            "user_id": payload.get("sub", "unknown"),
            "email": payload.get("email", "unknown"),
            "role": role,
            "iss": payload.get("iss"),
            "aud": payload.get("aud"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

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
_guardrails_validator = GuardrailsValidator()


def _guarded_response(question: str, answer: str, source_nodes: list[dict], input_decision, **kwargs) -> QueryResponse:
    """Apply output guardrails and consistently expose their decision to clients."""
    output = _guardrails_validator.validate_and_sanitize_output(answer)
    sanitized_source_nodes = []
    source_phi_summaries = []
    for node in source_nodes:
        sanitized_node = dict(node)
        node_text = str(node.get("text", node.get("content", "")))
        node_output = _guardrails_validator.validate_and_sanitize_output(node_text)
        if "text" in sanitized_node:
            sanitized_node["text"] = node_output.sanitized_text
        if "content" in sanitized_node:
            sanitized_node["content"] = node_output.sanitized_text
        sanitized_source_nodes.append(sanitized_node)
        source_phi_summaries.extend(node_output.pii_summaries)

    pii_summaries = list(dict.fromkeys(
        input_decision.phi_summaries + output.pii_summaries + source_phi_summaries
    ))
    return QueryResponse(
        query=question,
        answer=output.sanitized_text,
        source_nodes=sanitized_source_nodes,
        validation_results=ValidationResults(
            input_allowed=True,
            # This legacy field consistently describes the original request.
            # Output/source sanitization is reported separately below.
            pii_detected=input_decision.phi_detected,
            pii_summaries=pii_summaries,
            output_sanitized=output.output_sanitized or bool(source_phi_summaries),
            output_blocked=output.blocked,
            output_block_reason=output.reason,
        ),
        **kwargs,
    )


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
async def upload_document(
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Upload a medical document for indexing with multimodal content extraction.
    
    **Requires Authentication**
    
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
    logger.info(f"[UPLOAD] Document upload initiated by user: {current_user.get('user_id')} ({current_user.get('email')})")
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
async def query_rag(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Query RAG system with evaluation and human handoff logic.
    
    **Requires Authentication**
    """
    start_time = time.perf_counter()
    logger.info(f"[QUERY] Query received from user: {current_user.get('user_id')} ({current_user.get('email')})")
    logger.info(f"[QUERY] User role: {current_user.get('role')}")
    
    # Override email from authenticated user if in production mode
    if AUTH_ENABLED:
        request.user_email = current_user.get("email", request.user_email)
    
    input_decision = _guardrails_validator.decide_input(request.question)
    if not input_decision.allowed:
        logger.warning("Blocked query input: %s", input_decision.reason)
        raise HTTPException(status_code=400, detail=input_decision.reason)
    request.question = input_decision.sanitized_text
    # Classify intent before invoking any agent or retrieval tool.
    intent = await classify_medical_intent(request.question)
    if not intent["medical"]:
        # Do not reject a question simply because the current knowledge base
        # cannot cover it.  It cannot be answered reliably with the available
        # sources/tools, so create the same human-review handoff used for
        # low-confidence and empty-retrieval queries.
        reference_id = generate_handoff_reference_id()
        handoff_context = {
            "reference_id": reference_id,
            "trace_id": "scope-gate",
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "session_id": secrets.token_hex(8),
            "priority": "normal",
            "trigger_reason": intent["reason"],
            "query_history": [{"role": "user", "message": request.question}],
            "generated_answer": None,
            "evaluation_scores": {
                "faithfulness": None,
                "relevance": None,
                "confidence": None,
            },
            "retrieved_chunks": [],
            "user_metadata": {"email": request.user_email},
            "conversation_flow": [],
        }
        background_tasks.add_task(send_handoff_email, handoff_context)
        logger.info("Escalated unsupported query before retrieval: %s", reference_id)
        return _guarded_response(
            request.question,
            answer=(
                "Your request cannot be answered with the currently available "
                "knowledge base and tools, so it has been escalated to a human support team who will review and respond within 24 hours. "
                f"reviewer. Reference ID: {reference_id}"
            ),
            source_nodes=[],
            input_decision=input_decision,
            routing="handoff",
        )
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
            # The agentic path is the normal /query path.  It used to return here,
            # bypassing the legacy branch below where the Langfuse span and scores
            # are created.  Record the complete request before returning so traces
            # and evaluations appear for agent responses as well.
            langfuse = None
            span = None
            try:
                langfuse = get_langfuse_client()
                span = langfuse.start_span(
                    name="api_query",
                    input={"question": request.question},
                    metadata={
                        "route": result.routing,
                        "tools_used": result.tools_used,
                        "tool_errors": result.tool_errors,
                    },
                )
                trace_id = span.trace_id
                # HybridQueryService has already evaluated the final answer against
                # the evidence type actually used. Do not run document evaluators
                # against MCP/Hugging Face or SQL output here.
                evaluation = result.evaluation
                faithfulness = evaluation.get("faithfulness")
                relevance = evaluation.get("relevance")

                # The agent path is the primary /query path.  Apply the same
                # handoff decision and delivery logic used by legacy retrieval
                # before returning its response.
                overall_confidence = evaluation.get("overall_confidence")
                confidence_percent = (
                    round(float(overall_confidence) * 100)
                    if overall_confidence is not None else None
                )
                confidence_handoff = {
                    "trigger": confidence_percent is not None and confidence_percent < CONFIDENCE_THRESHOLD,
                    "reason": "source-aware confidence below threshold",
                    "confidence": confidence_percent,
                }
                explicit_handoff = {
                    "trigger": is_explicit_handoff_request(request.question),
                    "reason": "explicit user request for human help",
                }

                full_chunks = list(result.source_nodes)
                rag_used = evaluation.get("mode") in {"rag", "rag+mcp"}
                # Chunk-based handoff scoring applies only when document retrieval
                # actually contributed evidence.
                score_based_handoff = (
                    evaluate_score(
                        faithfulness,
                        relevance,
                        request.question,
                        no_chunks=not full_chunks,
                    )
                    if rag_used
                    else {"trigger": False, "reason": "non-RAG source-aware evaluation"}
                )
                if explicit_handoff["trigger"]:
                    handoff, priority = explicit_handoff, "high"
                elif confidence_handoff["trigger"] and score_based_handoff["trigger"]:
                    handoff, priority = confidence_handoff, "high"
                elif score_based_handoff["trigger"]:
                    handoff, priority = score_based_handoff, "normal"
                else:
                    handoff, priority = {"trigger": False}, None

                if handoff["trigger"]:
                    reference_id = generate_handoff_reference_id()
                    handoff_context = {
                        "reference_id": reference_id,
                        "trace_id": trace_id,
                        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                        "session_id": secrets.token_hex(8),
                        "priority": priority,
                        "trigger_reason": handoff["reason"],
                        "query_history": [{"role": "user", "message": request.question}],
                        "generated_answer": result.answer,
                        "evaluation_scores": {
                            "faithfulness": faithfulness,
                            "relevance": relevance,
                            "confidence": confidence_handoff["confidence"],
                        },
                        "retrieved_chunks": full_chunks,
                        "user_metadata": {"email": request.user_email},
                        "conversation_flow": [],
                    }
                    background_tasks.add_task(send_handoff_email, handoff_context)
                    span.update(output={"handoff": True, "reference_id": reference_id})
                    result.answer = (
                        "I don't have enough reliable information to answer confidently. "
                        "Your request has been escalated to a human reviewer. "
                        f"Reference ID: {reference_id}"
                    )
                    result.source_nodes = []
                else:
                    span.update(output={"handoff": False})
                span.update(output={
                    "answer": result.answer,
                    "faithfulness": faithfulness,
                    "answer_relevance": relevance,
                })
            except Exception as observability_exc:
                # Observability must never turn a successful query into an API error.
                logger.warning("Unable to record Langfuse observability for /query: %s", observability_exc)
            finally:
                if span is not None:
                    try:
                        span.end()
                    except Exception:
                        logger.warning("Unable to end Langfuse /query span", exc_info=True)
                if langfuse is not None:
                    try:
                        langfuse.flush()
                    except Exception:
                        logger.warning("Unable to flush Langfuse /query events", exc_info=True)
            response = _guarded_response(
                request.question,
                answer=result.answer,
                source_nodes=result.source_nodes,
                input_decision=input_decision,
                tools_used=result.tools_used,
                tool_errors=result.tool_errors,
                routing=result.routing,
            )
            logger.info("Query completed in %.2f ms", (time.perf_counter() - start_time) * 1000)
            return response
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
            confidence_handoff = await evaluate_confidence_score(
                answer=answer,
                question=request.question,
                relevant_chunks=full_chunks,
            )
        except Exception as exc:
            logger.warning("Confidence evaluation unavailable: %s", exc)
            confidence_handoff = {"trigger": False, "confidence": 50}
        confidence_score = confidence_handoff["confidence"]

        # ----------------------------------------------------
        # 4. LLM Intent Detection 
        # ----------------------------------------------------
        explicit_handoff = {
            "trigger": is_explicit_handoff_request(request.question),
            "reason": "explicit user request for human help",
        }

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

            response = _guarded_response(
                request.question,
                answer=(
                    "I don't have enough reliable information to answer confidently. "
                    "Your request has been escalated to a human reviewer. "
                    f"Reference ID: {reference_id}"
                ),
                source_nodes=[],
                input_decision=input_decision,
            )
            logger.info("Query completed in %.2f ms", (time.perf_counter() - start_time) * 1000)
            return response

        # ----------------------------------------------------
        # 6. Normal response
        # ----------------------------------------------------
        span.update(output={"handoff": False})
        span.end()

        response = _guarded_response(
            request.question,
            answer=answer,
            source_nodes=full_chunks,
            input_decision=input_decision,
        )
        logger.info("Query completed in %.2f ms", (time.perf_counter() - start_time) * 1000)
        return response

    except ValueError as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid query request")
    except Exception as e:
        # A retrieval, tool, or downstream service failure means this request
        # cannot be processed by the currently available system. Escalate it
        # rather than leaving the learner with a generic server error.
        logger.error("Error processing query; escalating to human review: %s", e, exc_info=True)
        reference_id = generate_handoff_reference_id()
        handoff_context = {
            "reference_id": reference_id,
            "trace_id": "processing-error",
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "session_id": secrets.token_hex(8),
            "priority": "high",
            "trigger_reason": "available knowledge base or tools could not process the query",
            "query_history": [{"role": "user", "message": request.question}],
            "generated_answer": None,
            "evaluation_scores": {
                "faithfulness": None,
                "relevance": None,
                "confidence": None,
            },
            "retrieved_chunks": [],
            "user_metadata": {"email": request.user_email},
            "conversation_flow": [],
        }
        background_tasks.add_task(send_handoff_email, handoff_context)
        return _guarded_response(
            request.question,
            answer=(
                "Your request could not be processed with the currently available "
                "knowledge base and tools, so it has been escalated to a human "
                f"reviewer. Reference ID: {reference_id}"
            ),
            source_nodes=[],
            input_decision=input_decision,
            routing="handoff",
        )


@router.post("/query/stream")
async def query_rag_stream(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Stream the existing query response without changing query generation."""
    response = await query_rag(request, background_tasks, current_user)
    answer = response.answer

    async def events():
        for index in range(0, len(answer), 160):
            yield f"event: chunk\ndata: {json.dumps({'content': answer[index:index + 160]})}\n\n"
        yield f"event: done\ndata: {json.dumps(response.model_dump(mode='json'))}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
