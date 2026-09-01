"""
Hybrid Query Service
Routes queries to RAG (unstructured) or SQL (structured) based on query type.
Combines results from both approaches for comprehensive medical answers.
"""

import logging
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from enum import Enum

from main.service.query_router import QueryRouter, QueryAnalysis
from main.service.fusion_retrieval import FusionRetrievalService
from main.service.nl_sql_query_engine import NLSQLQueryEngine, SQLQueryResult
from main.service.agent_factory import create_llm

logger = logging.getLogger(__name__)


class DataSourceType(str, Enum):
    """Types of data sources used in hybrid query."""
    RAG = "rag"  # Unstructured (textbooks, documents)
    SQL = "sql"  # Structured (clinical database)
    HYBRID = "hybrid"  # Both


@dataclass
class HybridQueryResult:
    """Result of hybrid query combining RAG and SQL results."""
    answer: str  # Final synthesized answer
    data_sources: List[DataSourceType]  # Which data sources were used
    rag_context: Optional[str] = None  # Context from RAG retrieval
    sql_results: Optional[List[Dict[str, Any]]] = None  # Rows from SQL query
    sql_query: Optional[str] = None  # The SQL query that was executed
    confidence: float = 0.0  # Confidence score (0.0-1.0)
    reasoning: Optional[str] = None  # Explanation of how answer was derived
    rag_routing_info: Optional[Dict[str, Any]] = None  # RAG retriever selection details (query_type, retrievers_used, confidence)


class HybridQueryService:
    """
    Routes medical questions to appropriate data source(s) and synthesizes results.
    
    Strategy:
    - Conceptual questions (why, how, explain) → RAG (semantic search on textbooks)
    - Factual questions (what, define, list) → SQL (structured clinical data)
    - Complex questions (compare, differentiate, integrate) → HYBRID (both)
    """
    
    logger = logging.getLogger(__name__)
    
    # Keywords indicating need for structured/SQL data
    SQL_KEYWORDS = {
        "define", "list", "what is", "specific", "exact",
        "density", "range", "temperature", "percentage", "ratio",
        "concentration", "drug", "medication", "interaction",
        "contraindication", "side effect", "adverse",
    }
    
    # Keywords indicating need for conceptual/RAG data
    RAG_KEYWORDS = {
        "how", "why", "explain", "describe", "mechanism",
        "process", "relationship", "understand", "function",
        "role", "adapt", "respond", "perceive", "sense",
    }
    
    def __init__(
        self,
        fusion_service: FusionRetrievalService,
        nl_sql_engine: NLSQLQueryEngine,
    ):
        """
        Initialize hybrid query service.
        
        Args:
            fusion_service: FusionRetrievalService for RAG queries
            nl_sql_engine: NLSQLQueryEngine for SQL queries
        """
        self.fusion_service = fusion_service
        self.nl_sql_engine = nl_sql_engine
        self.llm = create_llm()
        
        self.logger.info("HybridQueryService initialized")
    
    def route_query(self, question: str) -> Literal["sql", "rag", "hybrid"]:
        """
        Determine which data source(s) to query.
        
        Args:
            question: Medical question
            
        Returns:
            "sql", "rag", or "hybrid"
        """
        question_lower = question.lower()
        words = set(question_lower.split())
        
        sql_score = len(words & self.SQL_KEYWORDS)
        rag_score = len(words & self.RAG_KEYWORDS)
        
        self.logger.debug(f"Query routing: sql_score={sql_score}, rag_score={rag_score}")
        
        # Determine routing
        if sql_score >= 2 and rag_score == 0:
            return "sql"
        elif rag_score >= 2 and sql_score == 0:
            return "rag"
        elif sql_score > 0 and rag_score > 0:
            return "hybrid"
        else:
            # Default to hybrid for complex queries
            return "hybrid"
    
    def query(self, question: str, predefined_route: Optional[Literal["sql", "rag", "hybrid"]] = None) -> HybridQueryResult:
        """
        Execute hybrid query routing to appropriate data source(s).
        
        Uses:
        - QueryRouter for intelligent retriever selection in RAG (vector-only, BM25-only, or fusion)
        - NLSQLQueryEngine for SQL database queries
        
        Args:
            question: Medical question
            predefined_route: Optional pre-determined route ("sql", "rag", or "hybrid") to skip routing
        
        Returns:
            HybridQueryResult with answer and metadata
        """
        try:
            self.logger.info(f"hybrid.query start question_chars={len(question)}")
            
            # Step 1: Route the query (hybrid vs sql vs rag)
            # If route is predefined, skip routing to avoid double-routing
            if predefined_route:
                hybrid_route = predefined_route
                self.logger.info(f"hybrid.query using predefined_route={hybrid_route}")
            else:
                hybrid_route = self.route_query(question)
                self.logger.info(f"hybrid.query computed hybrid_route={hybrid_route}")
            
            # Step 2: Execute queries based on route
            data_sources = []
            rag_context = None
            sql_results = None
            sql_query = None
            rag_routing_info = None
            
            if hybrid_route in ["rag", "hybrid"]:
                self.logger.debug("hybrid.query executing RAG retrieval with fusion (vector + BM25)")
                
                # Analyze query for intelligent retriever selection
                query_analysis: QueryAnalysis = QueryRouter.analyze(question)
                QueryRouter.log_routing_decision(question, query_analysis)
                rag_routing_info = {
                    "query_type": query_analysis.query_type.value,
                    "confidence": query_analysis.confidence,
                    "retrievers": query_analysis.recommended_retrievers,
                }
                
                # Execute RAG with routed retriever (vector-only, BM25-only, or fusion)
                rag_chunks = self.fusion_service.retrieve_with_routing(
                    question,
                    query_analysis.recommended_retrievers
                )
                rag_context = "\n\n".join([chunk.text for chunk in rag_chunks])
                data_sources.append(DataSourceType.RAG)
                
                self.logger.info(
                    f"hybrid.query RAG retrieved {len(rag_chunks)} chunks "
                    f"using {query_analysis.query_type.value} retriever"
                )
            
            if hybrid_route in ["sql", "hybrid"]:
                self.logger.debug("hybrid.query executing SQL query")
                try:
                    sql_query, sql_result = self.nl_sql_engine.query(question)
                    sql_results = sql_result.results
                    data_sources.append(DataSourceType.SQL)
                    
                    if sql_result.error:
                        self.logger.warning(f"SQL query error: {sql_result.error}")
                        sql_results = None
                    else:
                        self.logger.info(
                            f"hybrid.query SQL returned {len(sql_results)} rows "
                            f"in {sql_result.execution_time:.1f}ms"
                        )
                except Exception as e:
                    self.logger.error(f"SQL query failed: {e}")
                    sql_results = None
            
            # Step 3: Synthesize answer from all sources
            answer = self._synthesize_answer(
                question=question,
                route=hybrid_route,
                rag_context=rag_context,
                sql_results=sql_results,
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                route=hybrid_route,
                rag_context=rag_context,
                sql_results=sql_results,
            )
            
            reasoning = (
                f"Routed to {hybrid_route}. "
                f"RAG context: {len(rag_context) if rag_context else 0} chars "
                f"({rag_routing_info['query_type'] if rag_routing_info else 'N/A'}). "
                f"SQL results: {len(sql_results) if sql_results else 0} rows."
            )
            
            result = HybridQueryResult(
                answer=answer,
                data_sources=data_sources,
                rag_context=rag_context,
                sql_results=sql_results,
                sql_query=sql_query,
                confidence=confidence,
                reasoning=reasoning,
                rag_routing_info=rag_routing_info,
            )
            
            self.logger.info(
                f"hybrid.query done hybrid_route={hybrid_route} confidence={confidence:.2f} "
                f"sources={','.join([s.value for s in data_sources])} "
                f"rag_routing={rag_routing_info['query_type'] if rag_routing_info else 'N/A'}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Hybrid query failed: {e}", exc_info=True)
            raise
    
    def _synthesize_answer(
        self,
        question: str,
        route: str,
        rag_context: Optional[str],
        sql_results: Optional[List[Dict[str, Any]]],
    ) -> str:
        """
        Synthesize final answer from RAG and SQL results using LLM.
        
        Args:
            question: Original question
            route: Routing decision (sql, rag, hybrid)
            rag_context: Context from RAG retrieval
            sql_results: Results from SQL query
            
        Returns:
            Synthesized answer
        """
        if route == "sql" and sql_results:
            return self._format_sql_results(question, sql_results)
        elif route == "rag" and rag_context:
            return rag_context[:2000]  # Return top RAG context
        elif route == "hybrid" and (rag_context or sql_results):
            # Build comprehensive answer combining both sources
            synthesis_prompt = self._build_synthesis_prompt(
                question=question,
                rag_context=rag_context,
                sql_results=sql_results,
            )
            
            response = self.llm.complete(synthesis_prompt)
            return response.text
        else:
            return "Unable to find relevant information to answer this question."
    
    def _build_synthesis_prompt(
        self,
        question: str,
        rag_context: Optional[str],
        sql_results: Optional[List[Dict[str, Any]]],
    ) -> str:
        """Build synthesis prompt for LLM to combine sources."""
        prompt = f"""
Synthesize a comprehensive medical answer using the following sources:

MEDICAL QUESTION:
{question}

"""
        
        if rag_context:
            prompt += f"""TEXTBOOK/CONCEPTUAL INFORMATION:
{rag_context[:1500]}

"""
        
        if sql_results:
            prompt += f"""CLINICAL DATABASE RESULTS:
{self._format_sql_results_for_prompt(sql_results)}

"""
        
        prompt += """
Provide a comprehensive answer that:
1. Explains the concept clearly using the textbook information
2. Incorporates specific facts from the clinical database
3. Is medically accurate and relevant to the question
4. Uses proper medical terminology
5. Cites the data source (textbook or database) when relevant

ANSWER:
"""
        
        return prompt
    
    @staticmethod
    def _format_sql_results(question: str, results: List[Dict[str, Any]]) -> str:
        """Format SQL results as readable text."""
        if not results:
            return "No matching records found in the clinical database."
        
        output = f"Clinical Database Results ({len(results)} record(s)):\n\n"
        
        for i, row in enumerate(results, 1):
            output += f"Result {i}:\n"
            for key, value in row.items():
                output += f"  {key}: {value}\n"
            output += "\n"
        
        return output
    
    @staticmethod
    def _format_sql_results_for_prompt(results: List[Dict[str, Any]]) -> str:
        """Format SQL results for LLM synthesis prompt."""
        if not results:
            return "No clinical database results"
        
        lines = []
        for i, row in enumerate(results, 1):
            row_str = " | ".join([f"{k}: {v}" for k, v in row.items()])
            lines.append(row_str)
        
        return "\n".join(lines[:10])  # Limit to first 10 results
    
    @staticmethod
    def _calculate_confidence(
        route: str,
        rag_context: Optional[str],
        sql_results: Optional[List[Dict[str, Any]]],
    ) -> float:
        """Calculate confidence score for the answer."""
        confidence = 0.5
        
        if route == "sql" and sql_results and len(sql_results) > 0:
            confidence = min(0.95, 0.7 + (len(sql_results) * 0.05))
        elif route == "rag" and rag_context:
            confidence = 0.75
        elif route == "hybrid" and rag_context and sql_results:
            confidence = 0.85
        
        return confidence


def create_hybrid_query_service(
    fusion_service: FusionRetrievalService,
    nl_sql_engine: NLSQLQueryEngine,
) -> HybridQueryService:
    """Factory function to create hybrid query service."""
    return HybridQueryService(fusion_service, nl_sql_engine)
