from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from main.service import rag_service


@pytest.fixture(autouse=True)
def reset_rag_state():
    original = (rag_service._index, rag_service._query_engine_cache, rag_service._llm, rag_service._embed_model, rag_service._semantic_chunker)
    rag_service._index = None
    rag_service._query_engine_cache = {}
    rag_service._llm = None
    rag_service._embed_model = None
    rag_service._semantic_chunker = None
    yield
    rag_service._index, rag_service._query_engine_cache, rag_service._llm, rag_service._embed_model, rag_service._semantic_chunker = original


def test_create_vector_store_uses_environment(monkeypatch):
    monkeypatch.setenv("DB_NAME", "db")
    monkeypatch.setenv("DB_HOST", "host")
    monkeypatch.setenv("DB_PASSWORD", "pw")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_USER", "user")
    store = MagicMock()
    factory = MagicMock(return_value=store)
    monkeypatch.setattr(rag_service.PGVectorStore, "from_params", factory)
    assert rag_service.create_vector_store() is store
    assert factory.call_args.kwargs["database"] == "db"
    assert factory.call_args.kwargs["embed_dim"] == 1536


def test_load_or_create_index_loads_existing_or_creates_empty(monkeypatch):
    store = MagicMock()
    monkeypatch.setattr(rag_service, "create_vector_store", lambda: store)
    monkeypatch.setattr(rag_service.StorageContext, "from_defaults", MagicMock(return_value="context"))
    existing = MagicMock()
    monkeypatch.setattr(rag_service.VectorStoreIndex, "from_vector_store", MagicMock(return_value=existing))
    assert rag_service.load_or_create_index("embed") is existing

    created = MagicMock()
    class EmptyIndex:
        @classmethod
        def from_vector_store(cls, **kwargs):
            raise RuntimeError("empty")

        def __new__(cls, **kwargs):
            return created

    monkeypatch.setattr(rag_service, "VectorStoreIndex", EmptyIndex)
    assert rag_service.load_or_create_index("embed") is created


def test_initialize_services_sets_global_dependencies(monkeypatch):
    llm, embed, chunker, index = object(), object(), object(), object()
    monkeypatch.setattr(rag_service, "create_llm", lambda: llm)
    monkeypatch.setattr(rag_service, "create_embel_model", lambda: embed)
    monkeypatch.setattr(rag_service, "SemanticChunker", lambda embed_model: chunker)
    monkeypatch.setattr(rag_service, "load_or_create_index", lambda model: index)
    monkeypatch.setattr(rag_service, "Settings", SimpleNamespace(llm=None, embed_model=None))
    rag_service.initialize_services()
    assert rag_service._llm is llm
    assert rag_service._embed_model is embed
    assert rag_service._semantic_chunker is chunker
    assert rag_service._index is index
    rag_service.initialize_services()


def test_getters_initialize_and_cache_query_engine(monkeypatch):
    index = object()
    rag_service._index = index
    rag_service._llm = "llm"
    engine = object()
    factory = MagicMock(return_value=engine)
    monkeypatch.setattr(rag_service, "create_query_engine", factory)
    assert rag_service.get_index() is index
    assert rag_service.get_query_engine(3) is engine
    assert rag_service.get_query_engine(3) is engine
    assert factory.call_count == 1


def test_add_documents_inserts_and_clears_cache():
    index = MagicMock()
    rag_service._index = index
    rag_service._query_engine_cache = {2: object()}
    result = rag_service.add_documents_to_index(["a", "b"])
    assert result["message"] == "Successfully indexed 2 document(s)"
    assert index.insert.call_count == 2
    assert rag_service._query_engine_cache == {}


def test_load_documents_validates_parser_results(monkeypatch):
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "key")
    parser = MagicMock()
    monkeypatch.setattr(rag_service, "LlamaParse", MagicMock(return_value=parser))
    parser.load_data.return_value = [SimpleNamespace(text="one"), SimpleNamespace(text="two")]
    assert rag_service.load_documents("lesson.pdf") == "one\n\ntwo"
    parser.load_data.return_value = []
    with pytest.raises(SystemExit, match="No content"):
        rag_service.load_documents("lesson.pdf")
    parser.load_data.return_value = [SimpleNamespace(text=" ")]
    with pytest.raises(SystemExit, match="empty"):
        rag_service.load_documents("lesson.pdf")


def test_load_documents_requires_api_key(monkeypatch):
    monkeypatch.delenv("LLAMA_CLOUD_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="Missing LLAMA_CLOUD_API_KEY"):
        rag_service.load_documents("lesson.pdf")


def test_process_document_indexes_text_table_and_image_nodes(monkeypatch):
    rag_service._index = MagicMock()
    rag_service._llm = "llm"
    rag_service._semantic_chunker = MagicMock()
    text_node = SimpleNamespace(metadata={"content_type": "text"}, text="text")
    table_node = SimpleNamespace(metadata={"content_type": "table_summary"}, text="table")
    rag_service._semantic_chunker.process.return_value = [text_node]
    monkeypatch.setattr(rag_service, "initialize_services", lambda: None)
    monkeypatch.setattr(rag_service, "get_file_metadata", lambda path: {"source": path})
    monkeypatch.setattr(rag_service, "extract_medical_metadata_from_filename", lambda name: {"topic": "pain"})
    monkeypatch.setattr(rag_service, "load_documents", lambda path: "body")
    monkeypatch.setattr(rag_service, "find_markdown_tables", lambda text: [(1, 3, "| a | b |")])
    monkeypatch.setattr(rag_service, "build_nodes_from_tables", lambda **kwargs: [table_node])
    monkeypatch.setattr(rag_service, "extract_images_from_pdf", lambda path, folder: [])
    monkeypatch.setattr(rag_service, "create_vector_store", lambda: "store")
    result = rag_service.process_document("tmp.pdf", original_filename="lesson.pdf")
    assert result["document_indexed"] == 2
    assert result["text_nodes"] == 1
    assert result["table_nodes"] == 1
    rag_service._index.insert_nodes.assert_called_once()


def test_initialization_and_vector_store_failures_are_propagated(monkeypatch):
    monkeypatch.setattr(
        rag_service, "create_llm", MagicMock(side_effect=RuntimeError("Azure unavailable"))
    )
    with pytest.raises(RuntimeError, match="Azure unavailable"):
        rag_service.initialize_services()

    monkeypatch.setattr(
        rag_service.PGVectorStore,
        "from_params",
        MagicMock(side_effect=RuntimeError("database unavailable")),
    )
    with pytest.raises(RuntimeError, match="database unavailable"):
        rag_service.create_vector_store()


def test_load_index_propagates_storage_setup_failure(monkeypatch):
    monkeypatch.setattr(
        rag_service, "create_vector_store", MagicMock(side_effect=RuntimeError("bad store"))
    )
    with pytest.raises(RuntimeError, match="bad store"):
        rag_service.load_or_create_index("embed")


def test_lazy_getters_initialize_each_dependency(monkeypatch):
    def initialize():
        rag_service._index = "index"
        rag_service._llm = "llm"
        rag_service._embed_model = "embed"

    initializer = MagicMock(side_effect=initialize)
    monkeypatch.setattr(rag_service, "initialize_services", initializer)
    assert rag_service.get_index() == "index"

    rag_service._index = None
    monkeypatch.setattr(rag_service, "create_query_engine", MagicMock(return_value="engine"))
    assert rag_service.get_query_engine() == "engine"

    rag_service._embed_model = None
    assert rag_service.get_embed_model() == "embed"
    rag_service._llm = None
    assert rag_service.get_llm() == "llm"
    assert initializer.call_count == 4


def test_add_documents_initializes_and_propagates_insert_error(monkeypatch):
    index = MagicMock()

    def initialize():
        rag_service._index = index

    monkeypatch.setattr(rag_service, "initialize_services", initialize)
    index.insert.side_effect = RuntimeError("insert failed")
    with pytest.raises(RuntimeError, match="insert failed"):
        rag_service.add_documents_to_index(["document"])


def test_process_document_without_original_name_creates_index_and_images(monkeypatch):
    rag_service._llm = "llm"
    rag_service._embed_model = "embed"
    rag_service._semantic_chunker = MagicMock()
    text_node = SimpleNamespace(metadata={"content_type": "text"}, text="text")
    rag_service._semantic_chunker.process.return_value = [text_node]
    created_index = MagicMock()
    index_factory = MagicMock(return_value=created_index)

    monkeypatch.setattr(rag_service, "initialize_services", lambda: None)
    monkeypatch.setattr(
        rag_service, "get_file_metadata", lambda path: {"source": "source.pdf"}
    )
    monkeypatch.setattr(rag_service, "load_documents", lambda path: "body without table")
    monkeypatch.setattr(rag_service, "find_markdown_tables", lambda text: [])
    monkeypatch.setattr(
        rag_service,
        "extract_images_from_pdf",
        lambda path, folder: [
            {"path": "one.png", "page": 1, "image_index": 0},
            {"path": "broken.png", "page": 2, "image_index": 1},
        ],
    )

    def caption(path):
        if path == "broken.png":
            raise RuntimeError("caption failed")
        return "An anatomy diagram"

    monkeypatch.setattr(rag_service, "generate_caption", caption)
    monkeypatch.setattr(rag_service, "create_vector_store", lambda: "store")
    monkeypatch.setattr(rag_service, "VectorStoreIndex", index_factory)

    result = rag_service.process_document("source.pdf")

    assert result["file_path"] == "source.pdf"
    assert result["document_indexed"] == 2
    assert result["image_nodes"] == 1
    index_factory.assert_called_once_with(nodes=[], vector_store="store", embed_model="embed")
    inserted = created_index.insert_nodes.call_args.args[0]
    assert inserted[1].metadata["content_type"] == "image_caption"
