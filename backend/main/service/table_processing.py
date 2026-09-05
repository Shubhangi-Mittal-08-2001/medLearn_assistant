"""Table Processing Service Module
Handles table summarization and node creation from extracted tables.

This module provides:
- Table summarization interface
- Node creation interface for table summaries
"""
import logging
from typing import List, Dict, Any

from llama_index.core.schema import TextNode
from llama_index.llms.azure_openai import AzureOpenAI

logger = logging.getLogger(__name__)


def summarize_table(llm: AzureOpenAI, table_md: str) -> str:
    """
    Summarize a markdown table into a natural language description.
    
    Args:
        llm: The language model instance
        table_md: Markdown table string
        
    Returns:
        Single-sentence summary of the table
    """
    logger.info("Summarizing table (boilerplate)")

    # TODO: Summarize the markdown table into a single-sentence natural language summary.
    # HINT: Use the provided `llm` to generate the summary text.
    # HINT: Keep the summary concise and factual (include key figures/relationships).
    # HINT: Decide how you want to handle model/content-filter failures (retry/fallback).
    # Your code here:
    logger.debug(f"Generating summary for table (length: {len(table_md)} characters)")
    # Use a more neutral, structured prompt to avoid content filter triggers
    prompt = (
        "Summarize the following markdown table in a single sentence. "
        "Include key figures and relationships. Be concise and factual.\n\n"
        f"Table:\n{table_md}\n\n"
        "Summary:"
    )
    try:
        resp = llm.complete(prompt)
        summary = resp.text.strip()
        logger.debug(f"Generated summary (length: {len(summary)} characters): {summary[:100]}...")
        return summary
    except Exception as e:
        error_msg = str(e)
        if "content_filter" in error_msg or "content management policy" in error_msg:
            logger.warning(f"Content filter triggered. Using fallback summary based on table structure.")
            # Fallback: create a basic summary from table structure
            lines = table_md.strip().split('\n')
            if len(lines) > 1:
                headers = lines[0] if lines else "table"
                summary = f"Table containing data with columns: {headers[:100]}"
            else:
                summary = "Table with structured data"
            return summary
        raise

def build_nodes_from_tables(
    source_name: str, 
    table_markdowns: List[str], 
    llm: AzureOpenAI,
    additional_metadata: Dict[str, Any] = None
) -> List[TextNode]:
    """
    Build TextNode objects from table markdowns with summaries.
    
    Args:
        source_name: Name of the source document
        table_markdowns: List of markdown table strings
        llm: Language model instance for summarization
        additional_metadata: Optional additional metadata to include
        
    Returns:
        List of TextNode objects with table summaries
    """
    logger.info("Building nodes from tables (boilerplate)")

    # TODO: Create TextNode objects from table markdowns.
    # HINT: For each table markdown:
    #   - create a summary using `summarize_table(llm, table_md)`
    #   - create a TextNode(text=summary, metadata=...)
    # HINT: Include at least these metadata fields:
    #   - source: source_name
    #   - content_type: "table_summary"
    #   - table_index: incremental index
    # HINT: Merge `additional_metadata` into each node's metadata if provided.
    # Your code here:
    logger.info(f"Building nodes from {len(table_markdowns)} table(s)")
    nodes: List[TextNode] = []
    base_metadata = additional_metadata or {}
    
    for idx, tbl in enumerate(table_markdowns, 1):
        logger.debug(f"Processing table {idx}/{len(table_markdowns)}")
        summary = summarize_table(llm, tbl)
        node = TextNode(
            text=f"""
        Table Summary:
        {summary}

        Original Table:
        {tbl}
        """.strip(),
            metadata={
                **base_metadata,
                "source": source_name,
                "content_type": "table_summary",
                "table_index": idx - 1,
            },
        )
        nodes.append(node)
    
    logger.info(f"Created {len(nodes)} TextNode(s) with table summaries")
    return nodes
