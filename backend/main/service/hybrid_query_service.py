"""Hybrid SQL, document-retrieval, and MCP agent orchestration for MedLearn.

The service intentionally keeps tool construction separate from execution.  A failed
database or an empty vector index therefore removes only that capability instead of
making the whole query API unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from dotenv import load_dotenv
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine, RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from main.service.agent_factory import create_llm
from main.service.fusion_retrieval import FusionRetrievalService
from main.service.mcp_client import HuggingFaceMCPService, iter_leaf_exceptions, root_exception_message
from main.service.query_router import QueryRouter
from main.service.rag_service import create_vector_store, get_llm, get_query_engine
from main.tools.custom_tools import custom_inference_tool
from llama_index.llms.openai import utils as openai_utils

load_dotenv()
logger = logging.getLogger(__name__)

CLINICAL_TABLES = [
    "somatosensory_receptors", "receptor_density", "pain_signal_types",
    "drug_interactions", "lab_reference_ranges",
]
SQL_PATTERNS = re.compile(
    r"\b(drug interaction|contraindication|reference range|lab(?:oratory)? value|"
    r"how many|count|density|dose|severity|warfarin|hemoglobin|platelet)\b", re.I
)

_GPT5_CONTEXT_WINDOWS = {
    "gpt-5": 400000,
    "gpt-5-mini": 400000,
    "gpt-5-nano": 400000,
}
openai_utils.ALL_AVAILABLE_MODELS.update(_GPT5_CONTEXT_WINDOWS)
openai_utils.CHAT_MODELS.update(_GPT5_CONTEXT_WINDOWS)

class ToolUnavailableError(RuntimeError):
    """A recoverable failure for one query capability."""


@dataclass
class ToolExecution:
    name: str
    status: str
    detail: str | None = None


@dataclass
class HybridQueryResult:
    question: str
    answer: str
    source_nodes: list[dict[str, Any]] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    tool_errors: list[dict[str, str]] = field(default_factory=list)
    routing: str = "agent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "source_nodes": self.source_nodes,
            "tools_used": self.tools_used,
            "tool_errors": self.tool_errors,
            "routing": self.routing,
        }


class TrackingQueryEngine:
    """Records every QueryEngineTool call without changing LlamaIndex behaviour.

    QueryEngineTool.acall uses ``aquery``, not ``query``. A wrapper that only
    implements ``query`` makes every async document search look like a hard failure.
    """

    def __init__(self, name: str, query_engine: Any, events: list[ToolExecution], source_nodes: list[dict[str, Any]] | None = None):
        self.name, self.query_engine, self.events = name, query_engine, events
        self.source_nodes = source_nodes

    def _record_source_nodes(self, response: Any) -> None:
        if self.source_nodes is None:
            return
        for item in getattr(response, "source_nodes", []) or []:
            node = getattr(item, "node", item)
            metadata = dict(getattr(node, "metadata", {}) or {})
            self.source_nodes.append({
                "chunk_id": getattr(node, "node_id", None),
                "text": getattr(node, "text", ""),
                "source": metadata.get("source") or metadata.get("file_name") or metadata.get("filename"),
                "score": float(item.score) if getattr(item, "score", None) is not None else None,
                "metadata": metadata,
            })

    def query(self, query: str) -> Any:
        logger.info("tool.execution.start tool=%s mode=sync", self.name)
        try:
            response = self.query_engine.query(query)
            self._record_source_nodes(response)
            self.events.append(ToolExecution(self.name, "success"))
            logger.info("tool.execution.end tool=%s status=success", self.name)
            return response
        except Exception as exc:
            self.events.append(ToolExecution(self.name, "error", str(exc)))
            logger.warning("tool.execution.end tool=%s status=error error=%s", self.name, exc)
            raise

    async def aquery(self, query: str) -> Any:
        logger.info("tool.execution.start tool=%s mode=async", self.name)
        try:
            if hasattr(self.query_engine, "aquery"):
                response = await self.query_engine.aquery(query)
            else:
                response = await asyncio.to_thread(self.query_engine.query, query)
            self._record_source_nodes(response)
            self.events.append(ToolExecution(self.name, "success"))
            logger.info("tool.execution.end tool=%s status=success", self.name)
            return response
        except Exception as exc:
            self.events.append(ToolExecution(self.name, "error", str(exc)))
            logger.warning("tool.execution.end tool=%s status=error error=%s", self.name, exc)
            raise


class TrackingTool:
    """Wraps LlamaIndex tools so MCP/custom tool calls are recorded in the same event stream."""

    def __init__(self, tool: Any, events: list[ToolExecution]):
        self.tool = tool
        self.events = events
        self.metadata = getattr(tool, "metadata", None)
        self.name = getattr(self.metadata, "name", None) or getattr(tool, "name", type(tool).__name__)
        self.__name__ = self.name

    def __getattr__(self, item: str) -> Any:
        return getattr(self.tool, item)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)

    def call(self, *args: Any, **kwargs: Any) -> Any:
        logger.info("tool.execution.start tool=%s mode=sync arguments=%s", self.name, _safe_arguments(args, kwargs))
        try:
            self._require_search_before_inference()
            validated_kwargs = self._validate_arguments(args, kwargs)
            response = self.tool.call(*args, **validated_kwargs)
            error = _response_error(response)
            if error:
                raise ToolUnavailableError(error)
            self.events.append(ToolExecution(self.name, "success"))
            logger.info("tool.execution.success tool=%s", self.name)
            return response
        except Exception as exc:
            detail = root_exception_message(exc)
            self.events.append(ToolExecution(self.name, "error", detail))
            _log_tool_exception(self.name, exc)
            raise

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        logger.info("tool.execution.start tool=%s mode=async arguments=%s", self.name, _safe_arguments(args, kwargs))
        try:
            self._require_search_before_inference()
            validated_kwargs = self._validate_arguments(args, kwargs)
            if hasattr(self.tool, "acall"):
                response = await self.tool.acall(*args, **validated_kwargs)
            else:
                response = await asyncio.to_thread(self.tool.call, *args, **validated_kwargs)
            error = _response_error(response)
            if error:
                raise ToolUnavailableError(error)
            self.events.append(ToolExecution(self.name, "success"))
            logger.info("tool.execution.success tool=%s", self.name)
            return response
        except Exception as exc:
            detail = root_exception_message(exc)
            self.events.append(ToolExecution(self.name, "error", detail))
            _log_tool_exception(self.name, exc)
            raise

    def _require_search_before_inference(self) -> None:
        """Prevent inference when this request has no successful model search."""
        if self.name != "hf_inference":
            return
        last_search = next(
            (event for event in reversed(self.events) if event.name == "hub_repo_search"),
            None,
        )
        if last_search is None or last_search.status != "success":
            raise ToolUnavailableError(
                "hf_inference requires a successful hub_repo_search result with a valid model ID"
            )

    def _validate_arguments(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        """Enforce schemas for tools whose argument names must not be guessed."""
        if args:
            return kwargs
        schema_model = getattr(self.metadata, "fn_schema", None)
        if schema_model is None:
            return kwargs
        expected_names = set(schema_model.model_fields)
        unexpected_names = set(kwargs) - expected_names
        if unexpected_names:
            error = (
                f"unexpected argument names {sorted(unexpected_names)}; "
                f"allowed names are {sorted(expected_names)}"
            )
            logger.warning(
                "tool.arguments.invalid tool=%s input_schema=%s received_argument_names=%s validation_error=%s",
                self.name,
                schema_model.model_json_schema(),
                sorted(kwargs),
                error,
            )
            raise ValueError(f"Invalid arguments for {self.name}: {error}")
        try:
            validated = schema_model.model_validate(kwargs)
        except Exception as exc:
            logger.warning(
                "tool.arguments.invalid tool=%s input_schema=%s received_argument_names=%s validation_error=%s",
                self.name,
                schema_model.model_json_schema(),
                sorted(kwargs),
                exc,
            )
            raise ValueError(
                f"Invalid arguments for {self.name}; expected {sorted(expected_names)}, "
                f"received {sorted(kwargs)}: {exc}"
            ) from exc
        values = validated.model_dump(exclude_unset=True)
        if self.name == "hub_repo_details" and not values.get("repo_ids"):
            raise ValueError("hub_repo_details requires a non-empty repo_ids list")
        return values


_SECRET_ARGUMENT_KEYS = {"authorization", "token", "api_key", "password", "secret"}


def _safe_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Summarize tool arguments without logging credentials or full user text."""
    summary: dict[str, Any] = {"positional_count": len(args)}
    for key, value in kwargs.items():
        if any(secret in key.lower() for secret in _SECRET_ARGUMENT_KEYS):
            summary[key] = "[REDACTED]"
        elif isinstance(value, str):
            summary[key] = {"type": "str", "length": len(value)}
        else:
            summary[key] = value
    return summary


def _response_error(response: Any) -> str | None:
    """Recognize MCP protocol failures that FunctionTool otherwise marks successful."""
    raw = getattr(response, "raw_output", response)
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, dict) and payload.get("error"):
            return str(payload.get("details") or payload["error"])
    if not (getattr(raw, "isError", False) or getattr(raw, "is_error", False)):
        return None
    content = getattr(raw, "content", None)
    if isinstance(content, list):
        messages = [str(getattr(item, "text", item)) for item in content]
        return "; ".join(messages) or "MCP tool returned an error"
    return str(content or "MCP tool returned an error")


def _log_tool_exception(tool_name: str, exc: BaseException) -> None:
    for leaf in iter_leaf_exceptions(exc):
        logger.warning(
            "tool.execution.error tool=%s root_exception_type=%s root_exception_message=%s",
            tool_name,
            type(leaf).__name__,
            leaf,
        )


def _database_url() -> str | URL:
    if url := os.getenv("DATABASE_URL"):
        return url
    required = ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD")
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise ToolUnavailableError("SQL database is not configured (missing " + ", ".join(missing) + ")")
    return URL.create(
        "postgresql+psycopg2",
        username=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"), port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME"),
    )


def _sql_query_engine() -> NLSQLTableQueryEngine:
    """Create a read-only, schema-constrained natural-language SQL engine."""
    try:
        engine = create_engine(_database_url(), pool_pre_ping=True)
        with engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")
        database = SQLDatabase(engine, include_tables=CLINICAL_TABLES)
        return NLSQLTableQueryEngine(sql_database=database, tables=CLINICAL_TABLES, llm=get_llm())
    except ToolUnavailableError:
        raise
    except Exception as exc:
        raise ToolUnavailableError(f"Clinical SQL database is unavailable: {exc}") from exc


def create_query_engine_tools(
    events: list[ToolExecution], source_nodes: list[dict[str, Any]] | None = None
) -> list[QueryEngineTool]:
    """Create SQL, semantic, BM25, and fusion ``QueryEngineTool`` instances."""
    tools: list[QueryEngineTool] = []
    builders: Iterable[tuple[str, str, Any]] = (
        ("clinical_sql", "Query the read-only structured clinical database for exact rows, counts, drug interactions, lab reference ranges, receptor densities, and other tabular facts. Input is a natural-language question. Combine with document or Hugging Face tools when the question explicitly needs both structured facts and another source.", _sql_query_engine),
        ("semantic_search", "Search uploaded/indexed medical documents by meaning. Use for conceptual explanations, mechanisms, and paraphrased textbook questions. Input is the document-focused question. Combine with another independent source when the user requests comparison or supplementary information.", lambda: get_query_engine(similarity_top_k=4)),
        ("bm25_search", "Search uploaded/indexed medical documents using exact keywords. Use for named terms, quotations, definitions, citations, numbers, and exact wording. Input is the document-focused question. Combine with semantic_search only when separate results are truly needed; otherwise prefer hybrid_document_search.", lambda: _fusion_service()._build_routed_retriever(["bm25"])),
        ("hybrid_document_search", "Search uploaded/indexed medical documents with fused semantic and BM25 retrieval. Use for complex document questions requiring both conceptual and exact evidence, including comparisons. Input is the document-focused question. It may run alongside SQL or Hugging Face tools for multi-source requests.", lambda: _fusion_service().fusion_retriever),
    )
    for name, description, builder in builders:
        try:
            value = builder()
            # BM25/fusion builders return retrievers; SQL/semantic return query engines.
            engine = value if hasattr(value, "query") else RetrieverQueryEngine.from_args(retriever=value)
            tools.append(QueryEngineTool.from_defaults(
                query_engine=TrackingQueryEngine(name, engine, events, source_nodes), name=name, description=description
            ))
        except Exception as exc:
            events.append(ToolExecution(name, "unavailable", str(exc)))
            logger.warning("Tool %s was not created: %s", name, exc)
    return tools


def _fusion_service() -> FusionRetrievalService:
    return FusionRetrievalService(
        vector_store=create_vector_store(),
        table_name=os.getenv("DB_TABLE_NAME", "medical_documents_embeddings"),
        similarity_top_k=4,
    )


class HybridQueryService:
    """Builds an OpenAI-compatible agent and executes hybrid medical queries."""

    SYSTEM_PROMPT = """You are MedLearn, a medical education assistant with several
independent information sources. Identify the user's information requirements, then
select exactly the tool or tools that satisfy them before answering. Uploaded or
indexed textbook content uses semantic_search, bm25_search, or
hybrid_document_search. Structured clinical records use clinical_sql. Hugging Face
MCP tools and hf_inference are first-class tools for information or model operations
the user asks to obtain from Hugging Face; they are not fallbacks.

For a multi-source request, call every required source and synthesize all successful
results. Issue independent calls together so the runtime can execute them in
parallel; use sequential calls only when a later call needs an earlier result (for
example model search followed by inference). Do not call every tool by default. If
one call fails, continue with other independent selected tools, disclose the missing
source, and never invent its result. When asked to find and run a Hugging Face model,
first call hub_repo_search using its discovered schema; only after it returns a valid
repository ID call hf_inference with that exact ID and the user's text. Never call
hf_inference after a failed/empty search. For hf_inference, the only argument names are
model_id (the exact repository identifier) and text (the user's input); never use model,
repo_id, engine, or inputs. hub_repo_details is optional and should be called only when
its metadata is necessary to choose a search result or confirm inference availability;
its repo_ids argument must be a non-empty list of repository IDs. Prefer candidates with
a live Hugging Face inference provider. For hub_repo_search, use only its discovered
schema: repo_types=["model"], filters=["text-classification", "language:en"] and
sort="downloads" are valid; pipeline_tag and language are not fields. Never claim a source is unavailable unless its
tool returned an error. State when evidence is insufficient and never expose PII."""

    DATA_TOOLS = {"clinical_sql", "semantic_search", "bm25_search", "hybrid_document_search"}

    def __init__(self) -> None:
        self.agent: Any = None
        self.tools: list[Any] = []
        self.events: list[ToolExecution] = []
        self.source_nodes: list[dict[str, Any]] = []
        self.initialization_errors: list[ToolExecution] = []
        self._initialized = False

    @staticmethod
    def _tool_name(tool: Any) -> str:
        metadata = getattr(tool, "metadata", None)
        if metadata is not None:
            name = getattr(metadata, "name", None) or getattr(metadata, "tool_name", None)
            if name:
                return str(name)
        return getattr(tool, "name", type(tool).__name__)

    def _register_tool(self, tool: Any, source: str) -> None:
        if tool is None:
            return
        wrapped_tool = TrackingTool(tool, self.events) if not isinstance(tool, TrackingTool) else tool
        tool_name = self._tool_name(wrapped_tool)
        if any(self._tool_name(existing) == tool_name for existing in self.tools):
            logger.debug("Tool %s already registered from %s; skipping duplicate registration", tool_name, source)
            return
        self.tools.append(wrapped_tool)
        logger.info("Registered %s tool from %s", tool_name, source)

    async def initialize(self) -> None:
        if self._initialized:
            return
        self.events = []
        self.tools = []
        for tool in create_query_engine_tools(self.events, self.source_nodes):
            self._register_tool(tool, "data_tools")
        # Local HF inference is independently useful and must not depend on MCP
        # discovery or on document/embedding initialization.
        self._register_tool(custom_inference_tool, "custom_tool")
        try:
            mcp_tools = await HuggingFaceMCPService().load_tools()
            for tool in mcp_tools:
                self._register_tool(tool, "mcp")
        except Exception as exc:
            self.events.append(ToolExecution("huggingface_mcp", "unavailable", str(exc)))
            logger.warning("Hugging Face MCP unavailable; continuing with local tools: %s", exc)
        self.initialization_errors = list(self.events)
        logger.info(
            "agent.tools.available tools=%s unavailable=%s",
            [self._tool_name(tool) for tool in self.tools],
            [event.name for event in self.initialization_errors],
        )
        if self.tools:
            try:
                self.agent = self._create_openai_agent(self.tools)
            except Exception as exc:
                self.events.append(ToolExecution("agent", "unavailable", str(exc)))
                self.initialization_errors = list(self.events)
                logger.warning("Agent unavailable; queries will use routed tool fallback: %s", exc)
        self._initialized = True

    def _create_openai_agent(self, tools: list[Any]) -> Any:
        """Prefer FunctionAgent (native Azure tool calling). Fall back to ReAct."""
        llm = create_llm()
        agent_kwargs = {
            "tools": tools,
            "llm": llm,
            "system_prompt": self.SYSTEM_PROMPT,
            "verbose": True,
            "streaming": False,
            "timeout": 120.0,
        }
        try:
            from llama_index.core.agent.workflow import FunctionAgent
            logger.info("Creating FunctionAgent with native tool calling")
            return FunctionAgent(**agent_kwargs)
        except Exception as exc:
            from llama_index.core.agent.workflow import ReActAgent
            logger.info("FunctionAgent unavailable (%s); using ReActAgent", exc)
            return ReActAgent(**agent_kwargs)

    async def execute(self, question: str) -> HybridQueryResult:
        await self.initialize()
        logger.info("query.agent.received query_chars=%d", len(question))
        # TrackingQueryEngine objects retain this list reference, so mutate it in
        # place instead of rebinding it for each request.
        self.events[:] = self.initialization_errors
        self.source_nodes.clear()
        if self.agent is not None:
            try:
                # Workflow.run() takes StartEvent kwargs. A positional string is
                # treated as ctx (or dropped), so the model never sees the question.
                response = await self.agent.run(user_msg=question)
                result = self._result(question, str(response), "agent")
                if not result.tools_used and not result.tool_errors:
                    logger.warning("Agent returned without invoking a tool; running intent-routed fallback")
                    return await self._fallback(question)
                if result.tool_errors and not result.tools_used:
                    logger.warning("Agent tool execution failed; returning recorded error without unrelated fallback")
                logger.info("query.synthesis.completed routing=agent tools_used=%s", result.tools_used)
                return result
            except Exception as exc:
                logger.warning("Agent execution failed; attempting deterministic fallback: %s", exc)
                self.events.append(ToolExecution("agent", "error", str(exc)))
        return await self._fallback(question)

    async def _fallback(self, question: str) -> HybridQueryResult:
        """Degrade gracefully when model/tool calling cannot run; never hide tool errors."""
        analysis = QueryRouter.analyze(question)
        preferred = "clinical_sql" if SQL_PATTERNS.search(question) else (
            "semantic_search" if analysis.recommended_retrievers == ["vector"] else
            "bm25_search" if analysis.recommended_retrievers == ["bm25"] else "hybrid_document_search"
        )
        tool = next((item for item in self.tools if getattr(item.metadata, "name", "") == preferred), None)
        if tool is None:
            return self._result(question, "The requested data source is currently unavailable. Please try again later.", "fallback")
        try:
            response = await asyncio.to_thread(tool.call, question)
            return self._result(question, str(response), "fallback")
        except Exception as exc:
            self.events.append(ToolExecution(preferred, "error", str(exc)))
            return self._result(question, "I could not complete that query because the selected data source failed.", "fallback")

    def _result(self, question: str, answer: str, routing: str) -> HybridQueryResult:
        used = list(dict.fromkeys(event.name for event in self.events if event.status == "success"))
        errors = []
        seen_errors: set[tuple[str, str]] = set()
        for event in self.events:
            if event.status not in {"error", "unavailable"}:
                continue
            key = (event.name, event.detail or event.status)
            if key not in seen_errors:
                errors.append({"tool": key[0], "message": key[1]})
                seen_errors.add(key)
        actual_routing = f"{routing}:{'+'.join(used)}" if used else routing
        logger.info(
            "query.result.collected routing=%s tools_used=%s tool_error_count=%d source_count=%d",
            actual_routing, used, len(errors), len(self.source_nodes),
        )
        return HybridQueryResult(question, answer, list(self.source_nodes), used, errors, actual_routing)
