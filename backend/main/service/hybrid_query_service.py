"""Hybrid SQL, document-retrieval, and MCP agent orchestration for MedLearn.

The service intentionally keeps tool construction separate from execution.  A failed
database or an empty vector index therefore removes only that capability instead of
making the whole query API unavailable.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine, RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from main.service.agent_factory import create_llm
from main.service.fusion_retrieval import FusionRetrievalService
from main.service.mcp_client import HuggingFaceMCPService
from main.service.query_router import QueryRouter
from main.service.rag_service import create_vector_store, get_llm, get_query_engine
from main.tools.custom_tools import custom_inference_tool
from llama_index.llms.openai import utils as openai_utils

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
    tools_used: list[str] = field(default_factory=list)
    tool_errors: list[dict[str, str]] = field(default_factory=list)
    routing: str = "agent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "tools_used": self.tools_used,
            "tool_errors": self.tool_errors,
            "routing": self.routing,
        }


class TrackingQueryEngine:
    """Records every QueryEngineTool call without changing LlamaIndex behaviour.

    QueryEngineTool.acall uses ``aquery``, not ``query``. A wrapper that only
    implements ``query`` makes every async document search look like a hard failure.
    """

    def __init__(self, name: str, query_engine: Any, events: list[ToolExecution]):
        self.name, self.query_engine, self.events = name, query_engine, events

    def query(self, query: str) -> Any:
        try:
            response = self.query_engine.query(query)
            self.events.append(ToolExecution(self.name, "success"))
            return response
        except Exception as exc:
            self.events.append(ToolExecution(self.name, "error", str(exc)))
            raise

    async def aquery(self, query: str) -> Any:
        try:
            if hasattr(self.query_engine, "aquery"):
                response = await self.query_engine.aquery(query)
            else:
                response = await asyncio.to_thread(self.query_engine.query, query)
            self.events.append(ToolExecution(self.name, "success"))
            return response
        except Exception as exc:
            self.events.append(ToolExecution(self.name, "error", str(exc)))
            raise


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


def create_query_engine_tools(events: list[ToolExecution]) -> list[QueryEngineTool]:
    """Create SQL, semantic, BM25, and fusion ``QueryEngineTool`` instances."""
    tools: list[QueryEngineTool] = []
    builders: Iterable[tuple[str, str, Any]] = (
        ("clinical_sql", "Use for factual clinical data: drug interactions, lab ranges, receptor densities, and counts.", _sql_query_engine),
        ("semantic_search", "Use for conceptual explanations from indexed medical documents.", lambda: get_query_engine(similarity_top_k=4)),
        ("bm25_search", "Use for exact terminology, definitions, numbers, and keyword-heavy document questions.", lambda: _fusion_service()._build_routed_retriever(["bm25"])),
        ("hybrid_document_search", "Use when a document question needs both semantic and exact-keyword evidence.", lambda: _fusion_service().fusion_retriever),
    )
    for name, description, builder in builders:
        try:
            value = builder()
            # BM25/fusion builders return retrievers; SQL/semantic return query engines.
            engine = value if hasattr(value, "query") else RetrieverQueryEngine.from_args(retriever=value)
            tools.append(QueryEngineTool.from_defaults(
                query_engine=TrackingQueryEngine(name, engine, events), name=name, description=description
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

    SYSTEM_PROMPT = """You are MedLearn, a medical education assistant for indexed
course materials and a clinical SQL database.
You MUST call a tool before answering. For lecture notes, citations, authors,
footnotes, anatomy, and somatosensory content, call bm25_search or
hybrid_document_search first (use semantic_search for conceptual explanations).
Use clinical_sql only for structured table facts (drug interactions, lab ranges,
densities stored in SQL). Hugging Face MCP tools are only for a model-search or
inference request. Never claim document search is unavailable unless a tool
returned an error. Never answer from general web knowledge when a document tool
is available. State uncertainty if retrieved context is insufficient and never
expose PII."""

    DATA_TOOLS = {"clinical_sql", "semantic_search", "bm25_search", "hybrid_document_search"}

    def __init__(self) -> None:
        self.agent: Any = None
        self.tools: list[Any] = []
        self.events: list[ToolExecution] = []
        self.initialization_errors: list[ToolExecution] = []
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        self.events = []
        self.tools = create_query_engine_tools(self.events)
        try:
            mcp_tools = await HuggingFaceMCPService().load_tools()
            self.tools.extend(mcp_tools)
            self.tools.append(custom_inference_tool)
        except Exception as exc:
            self.events.append(ToolExecution("huggingface_mcp", "unavailable", str(exc)))
            logger.warning("Hugging Face MCP unavailable; continuing with local tools: %s", exc)
        self.initialization_errors = list(self.events)
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
        # TrackingQueryEngine objects retain this list reference, so mutate it in
        # place instead of rebinding it for each request.
        self.events[:] = self.initialization_errors
        if self.agent is not None:
            try:
                # Workflow.run() takes StartEvent kwargs. A positional string is
                # treated as ctx (or dropped), so the model never sees the question.
                response = await self.agent.run(user_msg=question)
                result = self._result(question, str(response), "agent")
                if not any(event.name in self.DATA_TOOLS and event.status == "success" for event in self.events):
                    logger.warning("Agent returned without using data tools; running retrieval fallback")
                    return await self._fallback(question)
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
        errors = [{"tool": event.name, "message": event.detail or event.status}
                  for event in self.events if event.status in {"error", "unavailable"}]
        return HybridQueryResult(question, answer, used, errors, routing)
