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
from main.service.mcp_client import (
    HuggingFaceMCPService,
    MedicalMCPService,
    iter_leaf_exceptions,
    root_exception_message,
)
from main.service.prompt_manager import get_agent_system_prompt
from main.service.query_router import QueryRouter
from main.service.rag_service import create_vector_store, get_llm, get_query_engine
from main.tools.custom_tools import custom_inference_tool
from main.evaluation.dataset_evaluation import evaluate_answer_relevance, evaluate_faithfulness_score
from main.evaluation.dataset import get_langfuse_client
from main.handoff.handoff_service import evaluate_confidence_score
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
RAG_TOOLS = {"semantic_search", "bm25_search", "hybrid_document_search"}
SQL_TOOLS = {"clinical_sql"}
HF_ZERO_SHOT_MODEL = os.getenv("HF_ZERO_SHOT_MODEL", "facebook/bart-large-mnli")
HF_ZERO_SHOT_LABELS = [
    "somatosensory education",
    "clinical symptom report",
    "unrelated",
]
KNOWN_MCP_TOOLS = {"hf_inference", "hub_repo_search", "hub_repo_details"}

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
    evaluation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "source_nodes": self.source_nodes,
            "tools_used": self.tools_used,
            "tool_errors": self.tool_errors,
            "routing": self.routing,
            "evaluation": self.evaluation,
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

    def __init__(
        self,
        tool: Any,
        events: list[ToolExecution],
        external_evidence: list[dict[str, Any]] | None = None,
    ):
        self.tool = tool
        self.events = events
        self.metadata = getattr(tool, "metadata", None)
        self.name = getattr(self.metadata, "name", None) or getattr(tool, "name", type(tool).__name__)
        self.__name__ = self.name
        self.external_evidence = external_evidence

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
            self._record_external_evidence(response)
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
            self._record_external_evidence(response)
            self.events.append(ToolExecution(self.name, "success"))
            logger.info("tool.execution.success tool=%s", self.name)
            return response
        except Exception as exc:
            detail = root_exception_message(exc)
            self.events.append(ToolExecution(self.name, "error", detail))
            _log_tool_exception(self.name, exc)
            raise

    def _record_external_evidence(self, response: Any) -> None:
        if self.external_evidence is None:
            return
        content = _normalize_external_result(response)
        if content:
            self.external_evidence.append({"tool": self.name, "content": content})

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
        kwargs = dict(kwargs)
        # QueryEngineTool exposes one canonical ``input`` field. Tool-calling
        # models nevertheless commonly emit ``query``/``question`` and a top_k
        # hint. Normalize that harmless variation at our boundary; retrieval
        # depth is owned by each configured query engine, not by the model.
        if self.name in RAG_TOOLS | SQL_TOOLS and "input" not in kwargs:
            aliases = [name for name in ("query", "question") if name in kwargs]
            if len(aliases) == 1:
                kwargs["input"] = kwargs.pop(aliases[0])
            kwargs.pop("top_k", None)
        # The Hugging Face MCP tools use different schemas from local
        # QueryEngineTool instances.  Function-calling models sometimes still
        # emit the generic `input` field, so normalize only unambiguous aliases.
        if self.name == "hub_repo_search" and "query" not in kwargs:
            for alias in ("input", "question"):
                if alias in kwargs:
                    kwargs["query"] = kwargs.pop(alias)
                    break
        if self.name == "hub_repo_details" and "repo_ids" not in kwargs:
            candidate = kwargs.pop("query", kwargs.pop("input", None))
            if isinstance(candidate, str) and re.fullmatch(r"[\w.-]+/[\w.-]+", candidate):
                kwargs["repo_ids"] = [candidate]
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


def _normalize_external_result(response: Any, limit: int = 12_000) -> str:
    """Create bounded evaluator input without traversing framework/session state."""
    value = getattr(response, "raw_output", response)
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif not isinstance(value, (str, int, float, bool, list, tuple, dict, type(None))):
        content = getattr(value, "content", None)
        value = content if content is not None else str(value)

    def sanitize(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                str(key): "[REDACTED]" if any(secret in str(key).lower() for secret in _SECRET_ARGUMENT_KEYS) else sanitize(val)
                for key, val in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [sanitize(entry) for entry in item[:100]]
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        text = getattr(item, "text", None)
        return text if isinstance(text, str) else str(item)

    normalized = sanitize(value)
    text = normalized if isinstance(normalized, str) else json.dumps(normalized, ensure_ascii=False, default=str)
    return text[:limit]


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

    SYSTEM_PROMPT = get_agent_system_prompt() + """

Tool-call contract:
- For clinical_sql, semantic_search, bm25_search, and
  hybrid_document_search, pass the natural-language question as `input` only.
- For a request to find and run a Hugging Face model, call hub_repo_search
  once first, using `query`, `repo_types=["model"]`,
  `filters=["text-classification"]`, `sort="downloads"`, and `limit`.
  Do not use generic `input` for Hub tools. Call hub_repo_details only with a
  selected repository ID as `repo_ids=["org/model"]`. Call hf_inference only
  with the returned exact `model_id` and the user's `text`. Do not run
  arbitrary, fill-mask, or unrelated models.
"""

    DATA_TOOLS = {"clinical_sql", "semantic_search", "bm25_search", "hybrid_document_search"}

    def __init__(self) -> None:
        self.agent: Any = None
        self.tools: list[Any] = []
        self.events: list[ToolExecution] = []
        self.source_nodes: list[dict[str, Any]] = []
        self.external_evidence: list[dict[str, Any]] = []
        self.tool_sources: dict[str, str] = {}
        self.tool_origins: dict[str, str] = {}
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
        original_name = self._tool_name(tool)
        evaluation_source = (
            "rag" if original_name in RAG_TOOLS else
            "sql" if original_name in SQL_TOOLS else
            "mcp" if source in {"huggingface_mcp", "medical_mcp", "custom_tool"} else
            "other"
        )
        wrapped_tool = (
            TrackingTool(
                tool,
                self.events,
                self.external_evidence if evaluation_source == "mcp" else None,
            )
            if not isinstance(tool, TrackingTool) else tool
        )
        tool_name = self._tool_name(wrapped_tool)
        if any(self._tool_name(existing) == tool_name for existing in self.tools):
            logger.debug("Tool %s already registered from %s; skipping duplicate registration", tool_name, source)
            return
        self.tools.append(wrapped_tool)
        self.tool_sources[tool_name] = evaluation_source
        self.tool_origins[tool_name] = source
        logger.info("Registered %s tool from %s", tool_name, source)

    async def initialize(self) -> None:
        if self._initialized:
            return
        self.events = []
        self.tools = []
        self.tool_sources = {}
        self.tool_origins = {}
        for tool in create_query_engine_tools(self.events, self.source_nodes):
            self._register_tool(tool, "data_tools")
        # Local HF inference is independently useful and must not depend on MCP
        # discovery or on document/embedding initialization.
        self._register_tool(custom_inference_tool, "custom_tool")
        if os.getenv("HF_TOKEN"):
            try:
                for tool in await HuggingFaceMCPService().load_tools():
                    self._register_tool(tool, "huggingface_mcp")
            except Exception as exc:
                self.events.append(ToolExecution("huggingface_mcp", "unavailable", str(exc)))
                logger.warning("Hugging Face MCP unavailable; continuing with local tools: %s", exc)
        try:
            for tool in await MedicalMCPService().load_tools():
                self._register_tool(tool, "medical_mcp")
        except Exception as exc:
            self.events.append(ToolExecution("medical_mcp", "unavailable", str(exc)))
            logger.warning("Medical MCP unavailable; continuing with configured tools: %s", exc)
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
        self.external_evidence.clear()
        await self._refresh_document_retrievers()
        if self._is_huggingface_classification_request(question) and self._has_huggingface_classification_tools():
            return await self._execute_huggingface_classification(question)
        if self.agent is not None:
            try:
                # Workflow.run() takes StartEvent kwargs. A positional string is
                # treated as ctx (or dropped), so the model never sees the question.
                response = await self.agent.run(user_msg=question)
                answer = await self._complete_requested_hf_inference(question, str(response))
                result = await self._build_result(question, answer, "agent")
                result = await self._redirect_weak_rag_to_huggingface_mcp(question, result)
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

    async def _refresh_document_retrievers(self) -> None:
        """Refresh BM25/hybrid tools so uploads after startup are searchable."""
        try:
            fusion_service = await asyncio.to_thread(_fusion_service)
            refreshed = {
                "bm25_search": fusion_service._build_routed_retriever(["bm25"]),
                "hybrid_document_search": fusion_service.fusion_retriever,
            }
            for tool in self.tools:
                tool_name = self._tool_name(tool)
                if tool_name not in refreshed:
                    continue
                query_engine = getattr(tool, "query_engine", None)
                if isinstance(query_engine, TrackingQueryEngine):
                    query_engine.query_engine = RetrieverQueryEngine.from_args(
                        retriever=refreshed[tool_name]
                    )
            logger.info("query.document_retrievers.refreshed tools=%s", list(refreshed))
        except Exception as exc:
            logger.warning("query.document_retrievers.refresh_failed error=%s", exc)

    @staticmethod
    def _is_huggingface_classification_request(question: str) -> bool:
        """Recognize the explicit search-then-inference workflow."""
        return bool(
            re.search(r"\bhugging\s*face\b", question, re.I)
            and re.search(r"\b(?:classify|classification|run|infer(?:ence)?)\b", question, re.I)
        )

    def _has_huggingface_classification_tools(self) -> bool:
        available = {self._tool_name(tool) for tool in self.tools}
        return {"hub_repo_search", "hf_inference"}.issubset(available)

    async def _execute_huggingface_classification(self, question: str) -> HybridQueryResult:
        """Run the bounded Hub-search -> inference sequence without agent retries."""
        await self._search_for_medical_classifiers()
        answer = await self._complete_requested_hf_inference(
            question,
            "No Hugging Face classification result has been produced yet.",
        )
        if any(event.name == "hf_inference" and event.status == "error" for event in self.events):
            answer = await self._classify_with_medical_mcp(question, answer)
        return await self._build_result(question, answer, "huggingface:classification")

    async def _classify_with_medical_mcp(self, question: str, previous_answer: str) -> str:
        """Use medical literature MCP when HF cannot execute the requested task."""
        medical_tools = [
            tool for tool in self.tools
            if self.tool_origins.get(self._tool_name(tool)) == "medical_mcp"
        ]
        if not medical_tools:
            return previous_answer
        try:
            agent = self._create_openai_agent(medical_tools)
            response = await agent.run(user_msg=(
                "Use the available medical literature tools to assess this clinical text. "
                "Return a concise evidence-grounded classification, explain the clinical "
                "meaning of the terms, and cite the retrieved medical source identifiers. "
                "Do not claim that a machine-learning model classified it. Question: "
                f"{question}"
            ))
            return str(response)
        except Exception as exc:
            logger.warning("Medical MCP classification fallback failed: %s", exc)
            return previous_answer

    async def _redirect_weak_rag_to_huggingface_mcp(
        self, question: str, result: HybridQueryResult
    ) -> HybridQueryResult:
        """Re-plan weak local-document answers using configured HF MCP tools."""
        threshold = float(os.getenv("RAG_FAITHFULNESS_THRESHOLD", "0.70"))
        threshold = min(max(threshold, 0.0), 1.0)
        relevance_threshold = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.70"))
        relevance_threshold = min(max(relevance_threshold, 0.0), 1.0)
        faithfulness = result.evaluation.get("faithfulness")
        relevance = result.evaluation.get("relevance")
        local_evidence_is_weak = (
            isinstance(faithfulness, (int, float)) and faithfulness < threshold
        ) or (
            isinstance(relevance, (int, float)) and relevance < relevance_threshold
        )
        if (
            not any(tool_name in RAG_TOOLS for tool_name in result.tools_used)
            or not local_evidence_is_weak
            or any(self.tool_origins.get(name) == "huggingface_mcp" for name in result.tools_used)
        ):
            return result

        huggingface_tools = [
            tool for tool in self.tools
            if self.tool_origins.get(self._tool_name(tool)) == "huggingface_mcp"
        ]
        if not huggingface_tools:
            logger.info(
                "Weak RAG evidence faithfulness=%s relevance=%s, but Hugging Face MCP is unavailable",
                faithfulness, relevance,
            )
            return result

        names = [self._tool_name(tool) for tool in huggingface_tools]
        logger.info(
            "Weak RAG evidence faithfulness=%s relevance=%s; re-planning with Hugging Face MCP tools=%s",
            faithfulness, relevance, names,
        )
        successful_before = {
            event.name for event in self.events if event.status == "success"
        }
        try:
            huggingface_agent = self._create_openai_agent(huggingface_tools)
            response = await huggingface_agent.run(user_msg=(
                "The local document evidence was insufficient. "
                f"Answer this medical-education question using the most suitable available "
                f"Hugging Face MCP tool ({', '.join(names)}). Use only a tool whose description "
                "can retrieve relevant external content (for example papers, documentation, or "
                "datasets). A model/repository listing is not medical evidence. You must call a "
                "tool before answering, cite source identifiers or links returned by it, and state "
                f"when the available Hugging Face evidence is insufficient. Question: {question}"
            ))
        except Exception as exc:
            logger.warning("Hugging Face MCP evidence retry failed: %s", exc)
            self.events.append(ToolExecution("huggingface_mcp_retry", "error", str(exc)))
            return await self._build_result(question, result.answer, "agent")

        successful_after = {
            event.name for event in self.events
            if event.status == "success" and self.tool_origins.get(event.name) == "huggingface_mcp"
        }
        if not successful_after - successful_before:
            logger.warning("Hugging Face MCP retry returned without successfully calling a tool")
            return await self._build_result(question, result.answer, "agent")
        return await self._build_result(question, str(response), "agent:mcp_fallback")

    async def _complete_requested_hf_inference(self, question: str, answer: str) -> str:
        """Finish an explicit search-then-classify request if the agent stops after search."""
        successful = {event.name for event in self.events if event.status == "success"}
        wants_inference = bool(re.search(r"\b(?:classify|classification|run|infer(?:ence)?)\b", question, re.I))
        if not wants_inference or "hub_repo_search" not in successful or "hf_inference" in successful:
            return answer

        inference_text = self._classification_text(question)
        inference_tool = next(
            (tool for tool in self.tools if self._tool_name(tool) == "hf_inference"), None
        )
        candidates = self._medical_model_ids(self._model_ids_from_search_evidence())
        if not candidates:
            await self._search_for_medical_classifiers()
            candidates = self._medical_model_ids(self._model_ids_from_search_evidence())
        if not candidates or not inference_text or inference_tool is None:
            logger.warning(
                "Unable to complete requested HF inference candidates_found=%s text_found=%s tool_available=%s",
                bool(candidates), bool(inference_text), inference_tool is not None,
            )
            if inference_text:
                return self._medical_classification_fallback(
                    inference_text,
                    "No suitable hosted biomedical classifier was returned by Hugging Face search.",
                )
            return answer
        last_failure = "No hosted candidate completed inference."
        for model_id in candidates[:5]:
            try:
                output = await inference_tool.acall(model_id=model_id, text=inference_text)
                logger.info("Completed requested search-then-inference sequence with model=%s", model_id)
                return f"Model: {model_id}\n\nClassification result:\n{output}"
            except Exception as exc:
                logger.warning("Requested HF inference failed model=%s error=%s", model_id, exc)
                detail = root_exception_message(exc)
                last_failure = f"`{model_id}` was unavailable through the configured inference provider."
                if "403" in detail or "sufficient permissions" in detail.lower():
                    return self._medical_classification_fallback(
                        inference_text,
                        f"Hugging Face found `{model_id}`, but its inference endpoint rejected the configured token.",
                    )
        # Biomedical repositories are frequently not hosted by the configured
        # provider. Use a provider-hosted zero-shot model for the educational
        # routing task rather than returning a rule-based result as model output.
        try:
            output = await inference_tool.acall(
                model_id=HF_ZERO_SHOT_MODEL,
                text=inference_text,
                candidate_labels=HF_ZERO_SHOT_LABELS,
            )
            logger.info(
                "Completed HF zero-shot classification fallback with model=%s",
                HF_ZERO_SHOT_MODEL,
            )
            return (
                f"Model: {HF_ZERO_SHOT_MODEL}\n\n"
                "Zero-shot classification result:\n"
                f"{output}"
            )
        except Exception as exc:
            logger.warning("HF zero-shot classification fallback failed: %s", exc)
        return self._medical_classification_fallback(inference_text, last_failure)

    def _model_ids_from_search_evidence(self) -> list[str]:
        search_content = "\n".join(
            str(item.get("content", ""))
            for item in self.external_evidence
            if item.get("tool") == "hub_repo_search"
        )
        patterns = (
            r'"(?:repo_id|modelId|id)"\s*:\s*"([\w.-]+/[\w.-]+)"',
            r"https?://huggingface\.co/([\w.-]+/[\w.-]+)",
            r"\b([\w.-]+/[\w.-]+)\b",
        )
        candidates: list[str] = []
        for pattern in patterns:
            candidates.extend(re.findall(pattern, search_content))
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _best_medical_model_id(candidates: list[str]) -> str | None:
        suitable = HybridQueryService._medical_model_ids(candidates)
        return suitable[0] if suitable else None

    @staticmethod
    def _medical_model_ids(candidates: list[str]) -> list[str]:
        medical_terms = (
            "biomed", "medical", "clinical", "health", "disease", "symptom",
            "biobert", "medbert", "pubmed", "pathology", "radiology", "pharma",
        )
        unsuitable_terms = ("gguf", "pashto", "urdu", "arabic", "translation", "generation")
        suitable = [
            model_id for model_id in candidates
            if any(term in model_id.lower() for term in medical_terms)
            and not any(term in model_id.lower() for term in unsuitable_terms)
        ]
        # A repository name is not proof of its pipeline task, but this avoids
        # trying well-known base checkpoints before actual classifier heads.
        classifier_first = [
            model_id for model_id in suitable
            if "classif" in model_id.lower() or "classifier" in model_id.lower()
        ]
        return classifier_first + [model_id for model_id in suitable if model_id not in classifier_first]

    async def _search_for_medical_classifiers(self) -> None:
        search_tool = next(
            (tool for tool in self.tools if self._tool_name(tool) == "hub_repo_search"), None
        )
        if search_tool is None:
            return
        searches = (
            ("biomedical clinical text classification", ["text-classification"]),
            ("clinical symptom classification", ["text-classification"]),
            ("medical text classifier", ["text-classification"]),
        )
        for query, filters in searches:
            try:
                await search_tool.acall(
                    query=query,
                    repo_types=["model"],
                    filters=filters,
                    sort="downloads",
                    limit=10,
                )
                if self._best_medical_model_id(self._model_ids_from_search_evidence()):
                    return
            except Exception as exc:
                logger.warning("Focused biomedical model search failed query=%s error=%s", query, exc)

    @staticmethod
    def _medical_classification_fallback(text: str, external_status: str) -> str:
        lowered = text.lower()
        symptoms = [term for term in ("chest pain", "shortness of breath", "fever", "headache") if term in lowered]
        severity = next((term for term in ("severe", "moderate", "mild") if term in lowered), "unspecified")
        symptom_label = ", ".join(symptoms) if symptoms else "medical symptom statement"
        urgent = severity == "severe" or "chest pain" in symptoms
        urgency = "potential emergency / urgent assessment" if urgent else "clinical assessment advised"
        return (
            "Classification (rule-based fallback; not a Hugging Face model output):\n"
            f"- Input: {text}\n"
            f"- Symptom: {symptom_label}\n"
            f"- Severity: {severity}\n"
            "- Clinical intent: current symptom report\n"
            f"- Urgency: {urgency}\n\n"
            f"External-model status: {external_status} "
            "This fallback is for educational text classification and is not a diagnosis."
        )

    @staticmethod
    def _classification_text(question: str) -> str | None:
        matches = list(re.finditer(
            r"\bclassify(?:\s+(?:this\s+)?sentence)?\s*[:\-]?\s*[\"']?(.+?)[\"']?\s*$",
            question,
            re.I,
        ))
        return matches[-1].group(1).strip(" \"'") if matches else None

    async def _fallback(self, question: str) -> HybridQueryResult:
        """Degrade gracefully when model/tool calling cannot run; never hide tool errors."""
        analysis = QueryRouter.analyze(question)
        preferred = "clinical_sql" if SQL_PATTERNS.search(question) else (
            "semantic_search" if analysis.recommended_retrievers == ["vector"] else
            "bm25_search" if analysis.recommended_retrievers == ["bm25"] else "hybrid_document_search"
        )
        tool = next((item for item in self.tools if getattr(item.metadata, "name", "") == preferred), None)
        if tool is None:
            return await self._build_result(question, "The requested data source is currently unavailable. Please try again later.", "fallback")
        try:
            response = await asyncio.to_thread(tool.call, question)
            return await self._build_result(question, str(response), "fallback")
        except Exception as exc:
            self.events.append(ToolExecution(preferred, "error", str(exc)))
            return await self._build_result(question, "I could not complete that query because the selected data source failed.", "fallback")

    def _collect_execution(self, routing: str) -> tuple[list[str], list[dict[str, str]], str]:
        used = list(dict.fromkeys(event.name for event in self.events if event.status == "success"))
        errors = []
        seen_errors: set[tuple[str, str]] = set()
        for event in self.events:
            if event.status not in {"error", "unavailable"}:
                continue
            key = (event.name, event.detail or event.status)
            if key not in seen_errors:
                errors.append({"tool": key[0], "message": self._public_tool_error(key[1])})
                seen_errors.add(key)
        actual_routing = f"{routing}:{'+'.join(used)}" if used else routing
        logger.info(
            "query.result.collected routing=%s tools_used=%s tool_error_count=%d source_count=%d",
            actual_routing, used, len(errors), len(self.source_nodes),
        )
        return used, errors, actual_routing

    @staticmethod
    def _public_tool_error(detail: str) -> str:
        """Return an actionable error without leaking provider URLs or account metadata."""
        lowered = detail.lower()
        if "403" in detail or "sufficient permissions" in lowered:
            return (
                "Hugging Face rejected inference because the configured token lacks "
                "Inference Providers permission."
            )
        return detail[:1_000]

    async def _build_result(self, question: str, answer: str, routing: str) -> HybridQueryResult:
        used, errors, actual_routing = self._collect_execution(routing)
        evaluation = await self._evaluate_result(question, answer, used)
        return HybridQueryResult(
            question, answer, list(self.source_nodes), used, errors, actual_routing, evaluation
        )

    def _evaluation_source_for_tool(self, tool_name: str) -> str:
        if tool_name in RAG_TOOLS:
            return "rag"
        if tool_name in SQL_TOOLS:
            return "sql"
        if tool_name in KNOWN_MCP_TOOLS:
            return "mcp"
        return self.tool_sources.get(tool_name, "other")

    @staticmethod
    def _empty_evaluation(mode: str) -> dict[str, Any]:
        return {
            "mode": mode,
            "faithfulness": None,
            "relevance": None,
            "rag_confidence": None,
            "mcp_relevance": None,
            "mcp_confidence": None,
            "overall_confidence": None,
            "evaluation_error": None,
        }

    @staticmethod
    def _combine_confidence(
        rag_confidence: float | None, mcp_confidence: float | None
    ) -> float | None:
        scores = [score for score in (rag_confidence, mcp_confidence) if score is not None]
        return sum(scores) / len(scores) if scores else None

    async def _evaluate_rag_evidence(self, question: str, answer: str) -> tuple[float, float, float]:
        context = "\n\n".join(
            str(node.get("text", "")) for node in self.source_nodes if node.get("text")
        )
        
        # Get real Langfuse client for real-time score pushing
        langfuse_client = get_langfuse_client()
        trace_id = "hybrid-query-evaluation"
        
        # If Langfuse is not available, use a no-op sink
        if langfuse_client is None:
            class _NoOpScoreSink:
                def create_score(self, **_: Any) -> None:
                    return None
            langfuse_client = _NoOpScoreSink()
        
        faithfulness, relevance = await asyncio.gather(
            evaluate_faithfulness_score(langfuse_client, trace_id, question, context, answer),
            evaluate_answer_relevance(langfuse_client, trace_id, question, answer),
        )
        
        # Flush scores immediately to Langfuse for real-time pushing
        if langfuse_client is not None and hasattr(langfuse_client, 'flush'):
            try:
                langfuse_client.flush()
            except Exception as e:
                logger.warning(f"Failed to flush Langfuse scores: {e}")
        
        return faithfulness, relevance, (faithfulness + relevance) / 2

    async def _evaluate_mcp_evidence(self, question: str, answer: str) -> float:
        usable = [
            evidence for evidence in self.external_evidence
            if evidence.get("tool") in {event.name for event in self.events if event.status == "success"}
        ]
        if not usable:
            raise ValueError("No successful MCP/Hugging Face evidence was captured")
        usable.sort(key=lambda evidence: evidence.get("tool") != "hf_inference")
        confidence = await evaluate_confidence_score(answer, question, usable)
        return float(confidence["confidence"]) / 100.0

    async def _evaluate_result(
        self, question: str, answer: str, tools_used: list[str]
    ) -> dict[str, Any]:
        categories = {self._evaluation_source_for_tool(tool) for tool in tools_used}
        rag_used, mcp_used, sql_used = "rag" in categories, "mcp" in categories, "sql" in categories
        mode = "rag+mcp" if rag_used and mcp_used else "rag" if rag_used else "mcp" if mcp_used else "sql" if sql_used else "none"
        evaluation = self._empty_evaluation(mode)
        errors: list[str] = []
        rag_task = self._evaluate_rag_evidence(question, answer) if rag_used else None
        mcp_task = self._evaluate_mcp_evidence(question, answer) if mcp_used else None
        rag_result, mcp_result = await asyncio.gather(
            rag_task or asyncio.sleep(0, result=None),
            mcp_task or asyncio.sleep(0, result=None),
            return_exceptions=True,
        )
        if rag_used:
            if isinstance(rag_result, Exception):
                logger.warning("RAG answer evaluation failed: %s", rag_result)
                errors.append(f"RAG evaluation: {rag_result}")
            else:
                faithfulness, relevance, rag_confidence = rag_result
                evaluation.update(
                    faithfulness=faithfulness,
                    relevance=relevance,
                    rag_confidence=rag_confidence,
                )
        if mcp_used:
            if isinstance(mcp_result, Exception):
                logger.warning("MCP/Hugging Face answer evaluation failed: %s", mcp_result)
                errors.append(f"MCP evaluation: {mcp_result}")
            else:
                evaluation["mcp_confidence"] = mcp_result
        evaluation["overall_confidence"] = self._combine_confidence(
            evaluation["rag_confidence"], evaluation["mcp_confidence"]
        )
        evaluation["evaluation_error"] = "; ".join(errors) or None
        return evaluation
