"""
Natural Language to SQL Query Engine
Converts natural language medical questions to SQL queries using LLM.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

from llama_index.core import PromptTemplate
from main.service.agent_factory import create_llm
from main.service.sql_database import SQLDatabaseService

logger = logging.getLogger(__name__)


@dataclass
class SQLQueryResult:
    """Result of NL-to-SQL query execution."""
    query: str  # The generated SQL query
    results: List[Dict[str, Any]]  # Query results
    row_count: int  # Number of rows returned
    execution_time: float  # Time in milliseconds
    error: Optional[str] = None  # Error message if execution failed


class NLSQLQueryEngine:
    """
    Converts natural language medical questions to SQL queries.
    
    Uses an LLM to:
    1. Understand the medical question
    2. Generate appropriate SQL query
    3. Execute the query against the database
    4. Format and return results
    """
    
    logger = logging.getLogger(__name__)
    
    # Prompt for converting natural language to SQL
    NL_TO_SQL_PROMPT = PromptTemplate("""
You are an expert in converting natural language questions about medical data into SQL queries.

MEDICAL DATABASE SCHEMA:
{schema_info}

INSTRUCTIONS:
1. Analyze the natural language question carefully
2. Identify which tables and columns are needed
3. Generate a correct, efficient SQL query
4. Only return the SQL query, no explanation
5. Use proper SQL syntax and column names from the schema
6. For medical terminology, map to the most relevant database columns
7. Handle NULL values appropriately with IS NOT NULL or COALESCE

NATURAL LANGUAGE QUESTION:
{question}

MEDICAL DATABASE SCHEMA DETAILS:
{detailed_schema}

Generate the SQL query to answer the question:
""")
    
    # Detailed schema prompt for complex queries
    DETAILED_SCHEMA_PROMPT = PromptTemplate("""
TABLE: {table_name}
Description: {description}
Columns:
{columns}
Sample Data:
{sample_data}
""")
    
    def __init__(self, sql_db: Optional[SQLDatabaseService] = None):
        """
        Initialize NL-to-SQL query engine.
        
        Args:
            sql_db: SQLDatabaseService instance (creates new if not provided)
        """
        load_dotenv()
        
        self.sql_db = sql_db
        self.llm = create_llm()
        self.query_history = []
        
        self.logger.info("NLSQLQueryEngine initialized")
    
    def generate_sql_query(self, natural_language_question: str) -> str:
        """
        Convert natural language question to SQL query using LLM.
        
        Args:
            natural_language_question: Medical question in natural language
            
        Returns:
            Generated SQL query string
        """
        try:
            # Get database schema information
            schema_info = self.sql_db.get_schema_info()
            detailed_schema = self._get_detailed_schema_context()
            
            # Build the prompt
            prompt = self.NL_TO_SQL_PROMPT.format(
                schema_info=schema_info,
                question=natural_language_question,
                detailed_schema=detailed_schema,
            )
            
            self.logger.info(f"Generating SQL for question: {natural_language_question[:100]}...")
            
            # Call LLM to generate SQL
            response = self.llm.complete(prompt)
            sql_query = response.text.strip()
            
            # Clean up the response (remove markdown code blocks if present)
            if "```sql" in sql_query:
                sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql_query:
                sql_query = sql_query.split("```")[1].split("```")[0].strip()
            
            self.logger.debug(f"Generated SQL query: {sql_query}")
            
            return sql_query
            
        except Exception as e:
            self.logger.error(f"Failed to generate SQL query: {e}", exc_info=True)
            raise
    
    def execute_sql_query(self, sql_query: str) -> SQLQueryResult:
        """
        Execute SQL query and return results.
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            SQLQueryResult with query results and metadata
        """
        import time
        
        try:
            self.logger.info(f"Executing SQL query: {sql_query[:100]}...")
            
            t0 = time.perf_counter()
            results = self.sql_db.execute_query(sql_query)
            execution_time = (time.perf_counter() - t0) * 1000  # Convert to milliseconds
            
            result = SQLQueryResult(
                query=sql_query,
                results=results,
                row_count=len(results),
                execution_time=execution_time,
                error=None
            )
            
            self.logger.info(f"Query executed successfully in {execution_time:.1f}ms, returned {len(results)} rows")
            
            # Store in history
            self.query_history.append({
                "question": sql_query,
                "result_count": len(results),
                "execution_time": execution_time
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"SQL query execution failed: {e}", exc_info=True)
            
            return SQLQueryResult(
                query=sql_query,
                results=[],
                row_count=0,
                execution_time=0,
                error=str(e)
            )
    
    def query(self, natural_language_question: str) -> Tuple[str, SQLQueryResult]:
        """
        Convert natural language question to SQL and execute it.
        
        Args:
            natural_language_question: Medical question in natural language
            
        Returns:
            Tuple of (SQL query, SQLQueryResult)
        """
        try:
            # Step 1: Generate SQL from natural language
            sql_query = self.generate_sql_query(natural_language_question)
            
            # Step 2: Execute the SQL query
            result = self.execute_sql_query(sql_query)
            
            return sql_query, result
            
        except Exception as e:
            self.logger.error(f"NL-to-SQL query failed: {e}", exc_info=True)
            raise
    
    def _get_detailed_schema_context(self) -> str:
        """
        Build detailed schema context for LLM.
        
        Returns:
            Formatted schema information with descriptions and sample data
        """
        detailed_info = ""
        
        # Key medical tables to include
        key_tables = [
            "somatosensory_receptors",
            "receptor_density",
            "pain_signal_types",
            "drug_interactions",
            "lab_reference_ranges",
            "clinical_conditions",
        ]
        
        for table_name in key_tables:
            if self.sql_db.validate_table_exists(table_name):
                table_info = self.sql_db.get_table_info(table_name)
                if table_info:
                    sample_data = self.sql_db.get_sample_data(table_name, limit=2)
                    sample_str = self._format_sample_data(sample_data)
                    
                    columns_str = "\n".join([
                        f"  - {col['name']} ({col['type']}, {'nullable' if col['nullable'] else 'not null'})"
                        for col in table_info['columns']
                    ])
                    
                    table_desc = self._get_table_description(table_name)
                    
                    detailed_info += self.DETAILED_SCHEMA_PROMPT.format(
                        table_name=table_name,
                        description=table_desc,
                        columns=columns_str,
                        sample_data=sample_str,
                    )
        
        return detailed_info
    
    @staticmethod
    def _get_table_description(table_name: str) -> str:
        """Get description of a medical table."""
        descriptions = {
            "somatosensory_receptors": "Master reference table for all types of somatosensory receptors including mechanoreceptors, nociceptors, and proprioceptors",
            "receptor_density": "Quantitative data on receptor distribution across different body locations and age groups",
            "pain_signal_types": "Classification of pain pathways with transmission characteristics and clinical correlations",
            "drug_interactions": "Structured reference for drug-drug interactions relevant to pain management and medical care",
            "lab_reference_ranges": "Normal laboratory values and clinical significance of abnormal results",
            "clinical_conditions": "Disease states affecting the somatosensory system with diagnostic and treatment information",
        }
        return descriptions.get(table_name, f"Medical data table: {table_name}")
    
    @staticmethod
    def _format_sample_data(sample_data: List[Dict]) -> str:
        """Format sample data for display in prompt."""
        if not sample_data:
            return "No sample data available"
        
        rows = []
        for row in sample_data:
            row_str = " | ".join([f"{k}: {str(v)[:30]}" for k, v in row.items()])
            rows.append(row_str)
        
        return "\n".join(rows)


# Global instance
_nl_sql_engine: Optional[NLSQLQueryEngine] = None


def get_nl_sql_engine(sql_db: Optional[SQLDatabaseService] = None) -> NLSQLQueryEngine:
    """Get or create the global NL-to-SQL query engine instance."""
    global _nl_sql_engine
    
    if _nl_sql_engine is None:
        _nl_sql_engine = NLSQLQueryEngine(sql_db=sql_db)
    
    return _nl_sql_engine
