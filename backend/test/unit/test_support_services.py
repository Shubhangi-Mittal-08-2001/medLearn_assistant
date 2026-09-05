from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from fastapi import HTTPException

from main.security import security
from main.service import agent_factory, image_extraction
from main.service.agent_service import AgentService
from main.service.metadata import (
    extract_medical_metadata_from_filename,
    get_file_metadata,
)
from main.service.query_engine import create_query_engine
from main.service.semantic_chunking import SemanticChunker
from main.service.table_extraction import find_markdown_tables
from main.service.table_processing import build_nodes_from_tables, summarize_table


def test_metadata_extracts_every_supported_category():
    name = (
        "neurology_nervous_system_research_paper_head_neck_"
        "mechanoreceptor_diagnosis_tests_advanced.pdf"
    )
    metadata = extract_medical_metadata_from_filename(name)

    assert metadata == {
        "medical_specialty": "neurology",
        "body_system": "nervous-system",
        "content_type": "research-paper",
        "anatomical_region": "head-neck",
        "receptor_type": "mechanoreceptor",
        "clinical_topic": "diagnosis-tests",
        "difficulty_level": "advanced",
    }

    document = Path("test-fixtures") / name
    with patch.object(Path, "exists", return_value=True), patch.object(
        Path, "stat", return_value=SimpleNamespace(st_size=3)
    ):
        file_metadata = get_file_metadata(str(document))
    assert file_metadata["source"] == name
    assert file_metadata["file_extension"] == ".pdf"
    assert file_metadata["file_size"] == 3
    with patch.object(Path, "exists", return_value=False):
        assert get_file_metadata("test-fixtures/missing.txt")["file_size"] == 0


def test_markdown_table_extraction_returns_offsets_and_multiple_tables():
    text = "Before\n| A | B |\n|---|:---:|\n| 1 | 2 |\nAfter\n| X |\n|---|\n| y |"
    tables = find_markdown_tables(text)

    assert len(tables) == 2
    for start, end, table in tables:
        assert text[start:end] == table
    assert find_markdown_tables("No table here") == []
    assert find_markdown_tables("| header |\nnot a separator") == []


def test_table_summary_success_fallback_and_error():
    llm = MagicMock()
    llm.complete.return_value.text = "  Values increase.  "
    assert summarize_table(llm, "|A|\n|-|\n|1|") == "Values increase."

    llm.complete.side_effect = RuntimeError("content_filter blocked")
    assert "columns" in summarize_table(llm, "| A | B |\n|---|---|")
    assert summarize_table(llm, "single line") == "Table with structured data"

    llm.complete.side_effect = RuntimeError("network down")
    with pytest.raises(RuntimeError, match="network down"):
        summarize_table(llm, "|A|")


def test_build_table_nodes_preserves_content_and_metadata():
    llm = MagicMock()
    llm.complete.return_value.text = "A summary"
    nodes = build_nodes_from_tables(
        "lesson.pdf", ["|A|\n|-|\n|1|", "|B|\n|-|\n|2|"], llm, {"course": "MED101"}
    )

    assert len(nodes) == 2
    assert "Original Table" in nodes[0].text
    assert nodes[0].metadata == {
        "course": "MED101",
        "source": "lesson.pdf",
        "content_type": "table_summary",
        "table_index": 0,
    }
    assert build_nodes_from_tables("empty.pdf", [], llm) == []


def test_semantic_chunker_handles_empty_and_enriches_nodes():
    parser = MagicMock()
    parser.get_nodes_from_documents.return_value = [SimpleNamespace(metadata={})]
    with patch("main.service.semantic_chunking.SemanticSplitterNodeParser", return_value=parser):
        chunker = SemanticChunker(MagicMock())
        assert chunker.process("   ") == []
        nodes = chunker.process("Medical content", {"source": "lesson.pdf"})

    assert nodes[0].metadata == {"content_type": "text", "source": "lesson.pdf"}
    document = parser.get_nodes_from_documents.call_args.args[0][0]
    assert document.text == "Medical content"


def test_query_engine_delegates_and_propagates_errors():
    index = MagicMock()
    expected = index.as_query_engine.return_value
    llm = MagicMock()
    assert create_query_engine(index, llm, 4) is expected
    index.as_query_engine.assert_called_once_with(llm=llm, similarity_top_k=4)

    index.as_query_engine.side_effect = RuntimeError("index offline")
    with pytest.raises(RuntimeError, match="index offline"):
        create_query_engine(index, llm)


def test_agent_factories_validate_configuration_and_construct(monkeypatch):
    for key in (
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_LLM_DEPLOYMENT",
        "AZURE_OPENAI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="configuration required"):
        agent_factory.create_llm()

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://azure.test")
    monkeypatch.setenv("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-test")
    with pytest.raises(ValueError, match="API key required"):
        agent_factory.create_llm()

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "test-version")
    with patch("main.service.agent_factory.AzureOpenAI") as azure:
        assert agent_factory.create_llm() is azure.return_value
        azure.assert_called_once_with(
            engine="gpt-test", api_key="secret", azure_endpoint="https://azure.test",
            api_version="test-version", temperature=1.0,
        )

    with patch("main.service.agent_factory.AzureOpenAIEmbedding") as embedding:
        assert agent_factory.create_embel_model() is embedding.return_value

    with patch("main.service.agent_factory.create_llm", return_value="llm"), patch(
        "main.service.agent_factory.ReActAgent"
    ) as react:
        tools = [MagicMock()]
        assert agent_factory.create_agent(tools) is react.return_value
        react.assert_called_once_with(tools=tools, llm="llm")


@pytest.mark.asyncio
async def test_agent_service_lifecycle_delegation_and_sentiment_parsing():
    with patch("main.service.agent_service.HybridQueryService") as service_type:
        hybrid = service_type.return_value
        hybrid.initialize = AsyncMock()
        hybrid.agent = MagicMock()
        hybrid.tools = [MagicMock()]
        hybrid.events = []
        hybrid.execute = AsyncMock(return_value="answer")
        service = AgentService()

    assert not service.is_initialized()
    assert service.get_tools_count() == 0
    with pytest.raises(RuntimeError, match="not initialized"):
        await service.analyze_sentiment("good")

    await service.initialize()
    await service.initialize()
    hybrid.initialize.assert_awaited_once()
    assert service.is_initialized()
    assert service.is_mcp_connected()
    assert service.get_tools_count() == 1
    assert await service.execute_query("question") == "answer"

    async def response():
        return "Sentiment: POSITIVE\nConfidence: 0.98\nModel: test/model"

    service.agent.run.return_value = response()
    result = await service.analyze_sentiment("excellent")
    assert result == {
        "sentiment": "POSITIVE", "confidence": 0.98,
        "model_used": "test/model", "success": True,
    }
    assert service._parse_response('label: "negative" score: 0.7') == {
        "sentiment": "NEGATIVE", "confidence": 0.7, "model_used": "unknown"
    }
    assert service._parse_response("unstructured") is None


@pytest.mark.asyncio
async def test_agent_service_handles_unparseable_and_failed_agent():
    service = object.__new__(AgentService)
    service._initialized = True
    service._context = "context"
    service.agent = MagicMock()

    async def unparseable():
        return "nothing useful"

    service.agent.run.return_value = unparseable()
    result = await service.analyze_sentiment("text")
    assert result["sentiment"] == "UNKNOWN"
    assert result["success"] is False

    service.agent.run.side_effect = RuntimeError("agent offline")
    result = await service.analyze_sentiment("text")
    assert result["sentiment"] == "ERROR"
    assert result["error"] == "agent offline"


def test_security_jwks_cache_key_lookup_and_userinfo(monkeypatch):
    response = MagicMock()
    response.json.return_value = {"keys": [{"kid": "one", "kty": "RSA", "use": "sig", "n": "n", "e": "e"}]}
    monkeypatch.setattr(security.requests, "get", MagicMock(return_value=response))
    security._jwks_cache.update(keys=[], fetched_at=0)

    assert security._fetch_jwks()["keys"][0]["kid"] == "one"
    assert security._get_rsa_key({"kid": "one"})["kty"] == "RSA"
    assert security._get_rsa_key({"kid": "missing"}) is None
    # Cached fetch does not issue another HTTP request.
    security._fetch_jwks()
    assert security.requests.get.call_count == 1

    response.json.return_value = {"email": "learner@example.com"}
    assert security.get_user_email_from_auth0("token") == "learner@example.com"
    response.raise_for_status.side_effect = RuntimeError("offline")
    assert security.get_user_email_from_auth0("token") is None


@pytest.mark.parametrize(
    ("roles", "expected"),
    [([], "user"), (["USER"], "user"), (["Admin"], "admin"), ("admin", "admin")],
)
def test_security_roles(roles, expected):
    assert security.get_user_role({"https://medlearn.local/roles": roles}) == expected


@pytest.mark.asyncio
async def test_security_role_requirement_and_authenticated_user():
    admin = security.require_role(["admin"])
    payload = {
        "sub": "42", "email": "admin@example.com", "iss": "issuer", "aud": "audience",
        "https://medlearn.local/roles": ["admin"],
    }
    assert await admin(payload) == "admin"
    user = await security.get_authenticated_user(payload)
    assert user["user_id"] == "42"
    assert user["role"] == "admin"

    with pytest.raises(HTTPException) as exc:
        await admin({})
    assert exc.value.status_code == 403


def test_validate_token_success_and_authentication_failures(monkeypatch):
    key = {"kid": "one", "kty": "RSA", "use": "sig", "n": "n", "e": "e"}
    monkeypatch.setattr(security, "_get_rsa_key", MagicMock(return_value=key))
    monkeypatch.setattr(security.jwt, "get_unverified_header", MagicMock(return_value={"kid": "one"}))
    monkeypatch.setattr(security.jwt, "decode", MagicMock(return_value={"sub": "42"}))
    assert security.validate_token("valid") == {"sub": "42", "_raw_token": "valid"}

    security.jwt.get_unverified_header.side_effect = security.JWTError("bad")
    with pytest.raises(HTTPException, match="Invalid token header"):
        security.validate_token("bad")

    security.jwt.get_unverified_header.side_effect = None
    security._get_rsa_key.return_value = None
    monkeypatch.setattr(security, "_fetch_jwks", MagicMock())
    with pytest.raises(HTTPException, match="appropriate key"):
        security.validate_token("unknown")


def test_image_extraction_saves_images_and_always_closes():
    page_with_image = MagicMock()
    page_with_image.get_images.return_value = [(7,)]
    empty_page = MagicMock()
    empty_page.get_images.return_value = []
    document = MagicMock()
    document.__len__.return_value = 2
    document.load_page.side_effect = [page_with_image, empty_page]
    document.extract_image.return_value = {"image": b"image", "ext": "png"}

    file_handle = mock_open()
    output_dir = "test-output"
    with patch("main.service.image_extraction.fitz.open", return_value=document), patch(
        "builtins.open", file_handle
    ):
        result = image_extraction.extract_images_from_pdf("lesson.pdf", output_dir)

    assert result == [{
        "path": str(Path(output_dir) / "extracted_image_page0_img0.png"),
        "page": 0, "image_index": 0,
    }]
    file_handle.assert_called_once_with(Path(result[0]["path"]), "wb")
    file_handle().write.assert_called_once_with(b"image")
    document.close.assert_called_once()

    empty_document = MagicMock()
    empty_document.__len__.return_value = 0
    with patch("main.service.image_extraction.fitz.open", return_value=empty_document):
        assert image_extraction.extract_images_from_pdf("empty.pdf", output_dir) == []
    empty_document.close.assert_called_once()
