from types import SimpleNamespace
from unittest.mock import MagicMock

from main.service import fusion_retrieval
from main.service.fusion_retrieval import FusionRetrievalService


def make_service(nodes=None):
    vector_store = MagicMock()
    vector_store.get_nodes.return_value = nodes or []
    index = MagicMock()
    vector_retriever = MagicMock(name="vector")
    index.as_retriever.return_value = vector_retriever
    fusion_retriever = MagicMock(name="fusion")
    query_engine = MagicMock(name="engine")
    query_engine.query.return_value = "answer"

    old = (fusion_retrieval.StorageContext, fusion_retrieval.VectorStoreIndex, fusion_retrieval.QueryFusionRetriever, fusion_retrieval.RetrieverQueryEngine)
    fusion_retrieval.StorageContext = SimpleNamespace(from_defaults=MagicMock(return_value="storage"))
    fusion_retrieval.VectorStoreIndex = SimpleNamespace(from_vector_store=MagicMock(return_value=index))
    fusion_retrieval.QueryFusionRetriever = MagicMock(return_value=fusion_retriever)
    fusion_retrieval.RetrieverQueryEngine = SimpleNamespace(from_args=MagicMock(return_value=query_engine))
    try:
        service = FusionRetrievalService(vector_store=vector_store, table_name="docs", similarity_top_k=3)
    finally:
        fusion_retrieval.StorageContext, fusion_retrieval.VectorStoreIndex, fusion_retrieval.QueryFusionRetriever, fusion_retrieval.RetrieverQueryEngine = old
    return service, vector_retriever, fusion_retriever, query_engine


def test_fusion_service_vector_fallback_and_query():
    service, vector, fusion, engine = make_service()
    assert service.retrieval_type == "vector"
    assert service.bm25_retriever is None
    assert service._build_routed_retriever(["vector"]) is vector
    assert service._build_routed_retriever(["bm25"]) is vector
    assert service._build_routed_retriever([]) is fusion
    assert service.query("pain") == "answer"


def test_fusion_retrieve_extracts_filename_and_fallback_source():
    service, vector, fusion, _ = make_service()
    nodes = [
        SimpleNamespace(text="one", score=0.8, node=SimpleNamespace(metadata={"filename": "a.pdf"})),
        SimpleNamespace(text="two", node=SimpleNamespace(metadata={"source_path": "b.pdf"})),
        SimpleNamespace(text="three", score=None, node=SimpleNamespace(metadata=None)),
    ]
    fusion.retrieve.return_value = nodes
    results = service.retrieve("q")
    assert [(item.text, item.source) for item in results] == [("one", "a.pdf"), ("two", "b.pdf"), ("three", None)]
    assert results[1].score is None


def test_fusion_routed_query_uses_selected_retriever(monkeypatch):
    service, vector, _, _ = make_service()
    routed_engine = MagicMock()
    routed_engine.query.return_value = "routed answer"
    monkeypatch.setattr(fusion_retrieval.RetrieverQueryEngine, "from_args", MagicMock(return_value=routed_engine))
    assert service.query_with_routing("q", ["vector"]) == "routed answer"
    vector.retrieve.return_value = []
    assert service.retrieve_with_routing("q", ["vector"]) == []