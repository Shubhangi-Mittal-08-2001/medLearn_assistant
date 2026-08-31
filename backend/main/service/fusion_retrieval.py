from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.vector_stores import MetadataFilters

import time


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: Optional[float]
    source: Optional[str]


class FusionRetrievalService:
    """BM25 + Vector + RRF fusion retriever over a Postgres/pgvector table."""

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        vector_store: PGVectorStore,
        table_name: str,
        similarity_top_k: int = 5,
        num_queries: int = 1,
        mode: str = "reciprocal_rerank",
        use_async: bool = False,
    ) -> None:
        self.vector_store = vector_store
        self.table_name = table_name
        self.similarity_top_k = similarity_top_k
        self.num_queries = num_queries
        self.mode = mode
        self.use_async = use_async

        self.logger.info(
            "fusion.init start table=%s top_k=%d configured_retrieval=vector+bm25",
            self.table_name,
            self.similarity_top_k,
        )
        t0 = time.perf_counter()
        self._init_retrievers()
        self.logger.info(
            "fusion.init done ms=%.1f table=%s top_k=%d mode=%s retrieval_type=%s",
            (time.perf_counter() - t0) * 1000,
            self.table_name,
            self.similarity_top_k,
            self.mode,
            self.retrieval_type,
        )

    def _init_retrievers(self) -> None:
        """
        Initialize the fusion retrieval system
        
        Steps to implement:
        1. Create a StorageContext from the vector store
        2. Create a VectorStoreIndex from the vector store and storage context
        3. Create a vector retriever from the index
        4. Load all nodes from the vector store for BM25 retriever
        5. Create BM25 retriever if nodes are available
        6. Create a QueryFusionRetriever combining vector and BM25 retrievers
        7. Create a RetrieverQueryEngine from the fusion retriever
        """
        self.logger.info("fusion.retrievers initializing vector retriever table=%s", self.table_name)
        # Create a StorageContext from the vector store
        # Hint: A StorageContext is LlamaIndex's way of managing where data is stored. Use StorageContext.from_defaults(vector_store=self.vector_store) to create one. This context will be used to access the stored embeddings and documents.
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Create a VectorStoreIndex from the vector store and storage context
        # Hint: A VectorStoreIndex is an index structure that allows you to query the vector store efficiently. Use VectorStoreIndex.from_vector_store(vector_store=self.vector_store, storage_context=storage_context) to create it from the existing vector store. This index will enable semantic search capabilities. Save it as self.index.
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store, storage_context=storage_context)
        # Create a vector retriever from the index
        # Hint: A retriever is a component that finds relevant documents based on a query. Use self.index.as_retriever(similarity_top_k=self.similarity_top_k) to convert the index into a retriever that will return the top k most similar documents. This retriever uses vector similarity search (semantic search). Save it as self.vector_retriever.
        self.vector_retriever = self.index.as_retriever(similarity_top_k=self.similarity_top_k)

        # Load all nodes from the vector store for BM25 retriever
        # Hint: BM25 is a keyword-based retrieval algorithm that complements vector search. Import MetadataFilters from llama_index.core.vector_stores, then use self.vector_store.get_nodes(filters=MetadataFilters(filters=[])) to retrieve all nodes with empty filters. Wrap this in a try/except block because the vector store might be empty or there might be connection issues - if it fails, you'll just use vector search alone. Store the result in a variable like all_nodes.
        try:
            all_nodes = self.vector_store.get_nodes(filters=MetadataFilters(filters=[]))
            self.logger.info("fusion.retrievers loaded_bm25_nodes count=%d", len(all_nodes))
        except Exception as e:
            self.logger.warning(
                "fusion.retrievers bm25_unavailable; falling_back_to=vector error=%s", e
            )
            all_nodes = []

        # Create BM25 retriever if nodes are available
        # Hint: BM25 retriever works on keyword matching and is great for exact term searches. Start with a list containing only the vector retriever: retrievers = [self.vector_retriever]. If nodes were successfully loaded, use BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=min(self.similarity_top_k, max(1, len(all_nodes)))) to create a BM25 retriever. Add it to the retrievers list and save it as self.bm25_retriever. If no nodes were loaded, set self.bm25_retriever = None - you'll only use vector search in that case.
        retrievers = [self.vector_retriever]
        if all_nodes:
            self.bm25_retriever = BM25Retriever.from_defaults(
                nodes=all_nodes,
                similarity_top_k=min(self.similarity_top_k, max(1, len(all_nodes))),
            )
            retrievers = [self.vector_retriever, self.bm25_retriever]
            self.retrieval_type = "hybrid(vector+bm25)"
        else:
            self.bm25_retriever = None
            retrievers = [self.vector_retriever]
            self.retrieval_type = "vector"

        self.logger.info(
            "fusion.retrievers configured retrieval_type=%s vector_enabled=%s bm25_enabled=%s",
            self.retrieval_type,
            self.vector_retriever is not None,
            self.bm25_retriever is not None,
        )

        # Create a QueryFusionRetriever combining the retrievers
        # Hint: Fusion retrieval combines results from multiple retrieval methods (like vector search and BM25) to get better results. The QueryFusionRetriever uses Reciprocal Rank Fusion (RRF) to merge results from different retrievers. Use QueryFusionRetriever(retrievers=retrievers, similarity_top_k=self.similarity_top_k, num_queries=self.num_queries, mode=self.mode, use_async=self.use_async) to create it. Save it as self.fusion_retriever.
        self.fusion_retriever = QueryFusionRetriever(
            retrievers=retrievers,
            similarity_top_k=self.similarity_top_k,
            num_queries=self.num_queries,
            mode=self.mode,
            use_async=self.use_async,
        )
        # Create a RetrieverQueryEngine from the fusion retriever
        # Hint: A QueryEngine not only retrieves documents but also generates answers using an LLM. Use RetrieverQueryEngine.from_args(retriever=self.fusion_retriever) to create one using the fusion retriever you just built, and save it as self.query_engine. This is what you'll use to get full answers, not just document chunks.
        self.query_engine = RetrieverQueryEngine.from_args(retriever=self.fusion_retriever)
        
    def query(self, query_text: str) -> str:
        """
        Query the fusion retrieval system and return the answer
        
        Steps to implement:
        1. Call the query_engine.query() method with the query text
        2. Convert the result to a string
        3. Return the string result
        """
        self.logger.info(
            "fusion.query start retrieval_type=%s retriever=%s query_chars=%d",
            self.retrieval_type,
            type(self.fusion_retriever).__name__,
            len(query_text),
        )
        t0 = time.perf_counter()

        # Query the query engine with the query text
        # Hint: The query engine will use the fusion retriever to find relevant documents, then use an LLM to generate an answer based on those documents. Call self.query_engine.query(query_text) to get the response object that contains the generated answer.
        response = self.query_engine.query(query_text)
        # Convert the result to a string and return it
        # Hint: The query engine returns a response object, but we need just the text answer. Use str() to convert the query result to a string representation before returning it.
        answer = str(response)
        self.logger.info(
            "fusion.query done retrieval_type=%s ms=%.1f answer_chars=%d",
            self.retrieval_type,
            (time.perf_counter() - t0) * 1000,
            len(answer),
        )
        return answer

    def retrieve(self, query_text: str) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks using the fusion retriever
        
        Steps to implement:
        1. Call the fusion_retriever.retrieve() method with the query text
        2. Transform each retrieved node into a RetrievedChunk object
        3. Extract text, score, and source from each node
        4. Return the list of RetrievedChunk objects
        
        Node structure:
        - n.text: the text content
        - n.score: the relevance score (may not exist, use getattr with None default)
        - n.node.metadata: dictionary containing metadata like "filename" or "source_path"
        """
        self.logger.info(
            "fusion.retrieve start retrieval_type=%s retriever=%s query_chars=%d",
            self.retrieval_type,
            type(self.fusion_retriever).__name__,
            len(query_text),
        )
        t0 = time.perf_counter()

        # Retrieve nodes using the fusion retriever
        # Hint: Unlike the query method which generates answers, this method just retrieves the relevant document chunks. Use self.fusion_retriever.retrieve(query_text) to get nodes (document chunks) that are relevant to the given query text. The fusion retriever will combine results from vector search and BM25 search to return the most relevant chunks.
        # Transform each node into a RetrievedChunk object
        # Hint: The retrieved nodes are LlamaIndex node objects that contain text, scores, and metadata. Create an empty list: out: List[RetrievedChunk] = []. Iterate through each retrieved node. For each node: extract the text content using n.text, get the relevance score using getattr(n, "score", None) (which may not exist on all nodes), and extract the source file information from n.node.metadata dictionary (check for "filename" first using .get("filename"), then "source_path" as fallback). Wrap the metadata extraction in try/except to handle cases where metadata might be missing. Create RetrievedChunk(text=n.text, score=score, source=src) and append it to your list.
               
        document_chunks = self.fusion_retriever.retrieve(query_text)
        retrieved_chunks: List[RetrievedChunk] = []
        for n in document_chunks:
            self.logger.debug("Retrieved node: text=%s score=%s metadata=%s", n.text, getattr(n, "score", None), getattr(n.node, "metadata", None))
            text = n.text
            score = getattr(n, "score", None)
            try:
                metadata = getattr(n.node, "metadata", {})
                source = metadata.get("filename") or metadata.get("source_path")
            except Exception as e:
                self.logger.warning("Error extracting metadata from node: %s", e)
                source = None
            retrieved_chunk = RetrievedChunk(text=text, score=score, source=source)
            retrieved_chunks.append(retrieved_chunk)
        
        # Return the list of RetrievedChunk objects
        # Hint: Return the list of RetrievedChunk objects you've created. This gives the caller access to the retrieved document chunks with their scores and source information, which is useful for showing citations or debugging.
        self.logger.info(
            "fusion.retrieve done retrieval_type=%s results=%d ms=%.1f",
            self.retrieval_type,
            len(retrieved_chunks),
            (time.perf_counter() - t0) * 1000,
        )
        return retrieved_chunks
