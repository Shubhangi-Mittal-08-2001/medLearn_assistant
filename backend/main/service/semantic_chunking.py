"""Semantic chunking utilities for text content."""
import logging
from typing import List, Dict, Any

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import TextNode, Document
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Processes text content using semantic chunking."""
    
    def __init__(self, embed_model: AzureOpenAIEmbedding):
        """
        Initialize semantic chunker.
        
        Args:
            embed_model: Embedding model for semantic splitting
        """
        self.embed_model = embed_model
        self.parser = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
        )
        logger.info("Semantic chunker initialized")
    
    def process(self, text: str, metadata: Dict[str, Any] = None) -> List[TextNode]:
        """
        Process text into semantically chunked nodes.
        
        Args:
            text: Raw text content to process
            metadata: Additional metadata to attach to nodes
            
        Returns:
            List of TextNode objects with semantic chunks
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for semantic chunking")
            return []
        
        logger.info(f"Processing text with semantic chunking (length: {len(text)} characters)")
        
        # Create a temporary document node
        doc = Document(text=text, metadata=metadata or {})
        
        # Parse into nodes
        nodes = self.parser.get_nodes_from_documents([doc])
        
        # Add content_type metadata to all nodes
        for node in nodes:
            node.metadata = node.metadata or {}
            node.metadata["content_type"] = "text"
            if metadata:
                node.metadata.update(metadata)
        
        logger.info(f"Created {len(nodes)} semantic chunks from text")
        return nodes
