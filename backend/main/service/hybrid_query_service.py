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
from main.service.mcp_client import HuggingFaceMCPService, MedicalMCPService, iter_leaf_exceptions, root_exception_message
from main.service.query_router import QueryRouter
from main.service.rag_service import create_vector_store, get_llm, get_query_engine
from main.tools.custom_tools import custom_inference_tool
from main.evaluation.dataset_evaluation import evaluate_answer_relevance, evaluate_faithfulness_score
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

    SYSTEM_PROMPT = """You are MedLearn, a medical education assistant with several
independent information sources. Identify the user's information requirements, then
select exactly the tool or tools that satisfy them before answering. Uploaded or
indexed textbook content uses semantic_search, bm25_search, or
hybrid_document_search. Structured clinical records use clinical_sql. Pass the user's
natural-language question to any of these four tools using the single argument `input`;
do not pass `query`, `question`, or `top_k`. Hugging Face MCP tools and hf_inference are for Hugging Face repository/model
operations; they are not medical evidence sources. Tools from the configured medical
MCP server are external medical evidence sources and may be used for current external
evidence or when local textbook/database evidence is insufficient.

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
sort="downloads" are valid; pipeline_tag and language are not fields. When the user
requests a biomedical or clinical model, include an explicit biomedical/clinical query
instead of browsing all text-classification models. Reject unrelated-language, GGUF,
generation, or non-medical results rather than running them. Never claim a source is unavailable unless its
tool returned an error. State when evidence is insufficient and never expose PII."""

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
        medical_mcp = MedicalMCPService.from_environment()
        if medical_mcp is not None:
            try:
                for tool in await medical_mcp.load_tools():
                    self._register_tool(tool, "medical_mcp")
            except Exception as exc:
                self.events.append(ToolExecution("medical_mcp", "unavailable", str(exc)))
                logger.warning("Medical MCP unavailable; continuing with other tools: %s", exc)
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
        if self.agent is not None:
            try:
                # Workflow.run() takes StartEvent kwargs. A positional string is
                # treated as ctx (or dropped), so the model never sees the question.
                response = await self.agent.run(user_msg=question)
                answer = await self._complete_requested_hf_inference(question, str(response))
                result = await self._build_result(question, answer, "agent")
                result = await self._redirect_low_faithfulness_to_medical_mcp(question, result)
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

    async def _redirect_low_faithfulness_to_medical_mcp(
        self, question: str, result: HybridQueryResult
    ) -> HybridQueryResult:
        """Retry a weak hybrid-document answer using only medical MCP tools."""
        threshold = float(os.getenv("RAG_FAITHFULNESS_THRESHOLD", "0.70"))
        threshold = min(max(threshold, 0.0), 1.0)
        faithfulness = result.evaluation.get("faithfulness")
        if (
            "hybrid_document_search" not in result.tools_used
            or not isinstance(faithfulness, (int, float))
            or faithfulness >= threshold
            or any(self.tool_origins.get(name) == "medical_mcp" for name in result.tools_used)
        ):
            return result

        medical_tools = [
            tool for tool in self.tools
            if self.tool_origins.get(self._tool_name(tool)) == "medical_mcp"
        ]
        if not medical_tools:
            logger.info(
                "Low RAG faithfulness %.3f < %.3f, but no medical MCP tools are configured",
                faithfulness,
                threshold,
            )
            return result

        names = [self._tool_name(tool) for tool in medical_tools]
        logger.info(
            "Low RAG faithfulness %.3f < %.3f; redirecting to medical MCP tools=%s",
            faithfulness,
            threshold,
            names,
        )
        successful_before = {
            event.name for event in self.events if event.status == "success"
        }
        try:
            medical_agent = self._create_openai_agent(medical_tools)
            response = await medical_agent.run(user_msg=(
                "The local hybrid document answer did not meet the faithfulness threshold. "
                f"Answer this medical-education question using the most suitable available "
                f"external medical MCP tool ({', '.join(names)}). You must call a tool before "
                f"answering, cite source identifiers or links returned by it, and state when "
                f"the external evidence is insufficient. Question: {question}"
            ))
        except Exception as exc:
            logger.warning("Medical MCP low-faithfulness retry failed: %s", exc)
            self.events.append(ToolExecution("medical_mcp_retry", "error", str(exc)))
            return await self._build_result(question, result.answer, "agent")

        successful_after = {
            event.name for event in self.events
            if event.status == "success" and self.tool_origins.get(event.name) == "medical_mcp"
        }
        if not successful_after - successful_before:
            logger.warning("Medical MCP retry returned without successfully calling a medical tool")
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
        return [
            model_id for model_id in candidates
            if any(term in model_id.lower() for term in medical_terms)
            and not any(term in model_id.lower() for term in unsuitable_terms)
        ]

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
        class _ScoreSink:
            def create_score(self, **_: Any) -> None:
                return None

        context = "\n\n".join(
            str(node.get("text", "")) for node in self.source_nodes if node.get("text")
        )
        trace_id = "hybrid-query-evaluation"
        faithfulness, relevance = await asyncio.gather(
            evaluate_faithfulness_score(_ScoreSink(), trace_id, question, context, answer),
            evaluate_answer_relevance(_ScoreSink(), trace_id, question, answer),
        )
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
        if rag_used:
            try:
                faithfulness, relevance, rag_confidence = await self._evaluate_rag_evidence(question, answer)
                evaluation.update(
                    faithfulness=faithfulness,
                    relevance=relevance,
                    rag_confidence=rag_confidence,
                )
            except Exception as exc:
                logger.warning("RAG answer evaluation failed: %s", exc)
                errors.append(f"RAG evaluation: {exc}")
        if mcp_used:
            try:
                evaluation["mcp_confidence"] = await self._evaluate_mcp_evidence(question, answer)
            except Exception as exc:
                logger.warning("MCP/Hugging Face answer evaluation failed: %s", exc)
                errors.append(f"MCP evaluation: {exc}")
        evaluation["overall_confidence"] = self._combine_confidence(
            evaluation["rag_confidence"], evaluation["mcp_confidence"]
        )
        evaluation["evaluation_error"] = "; ".join(errors) or None
        return evaluation
