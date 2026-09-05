import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from main.handoff import handoff_service
from main.service import captioning, indexing


def test_indexing_load_documents_checks_directory(monkeypatch):
    monkeypatch.setattr(indexing.os.path, "exists", lambda path: False)
    with pytest.raises(FileNotFoundError):
        indexing.load_documents()


def test_indexing_load_documents_uses_reader(monkeypatch):
    reader = MagicMock()
    reader.load_data.return_value = ["doc"]
    monkeypatch.setattr(indexing.os.path, "exists", lambda path: True)
    monkeypatch.setattr(indexing, "SimpleDirectoryReader", MagicMock(return_value=reader))
    assert indexing.load_documents() == ["doc"]
    reader.load_data.assert_called_once_with()


def test_build_index_configures_postgres_and_index(monkeypatch):
    monkeypatch.setenv("DB_NAME", "db")
    monkeypatch.setenv("DB_HOST", "host")
    monkeypatch.setenv("DB_PASSWORD", "pw")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_USER", "user")
    store = MagicMock()
    context = MagicMock()
    index = MagicMock()
    pg = MagicMock(from_params=MagicMock(return_value=store))
    monkeypatch.setattr(indexing, "PGVectorStore", pg)
    monkeypatch.setattr(indexing.StorageContext, "from_defaults", MagicMock(return_value=context))
    monkeypatch.setattr(indexing.VectorStoreIndex, "from_documents", MagicMock(return_value=index))

    assert indexing.build_index(["doc"], "embed") is index
    pg.from_params.assert_called_once()
    indexing.VectorStoreIndex.from_documents.assert_called_once_with(
        ["doc"], storage_context=context, embed_model="embed", show_progress=True
    )


def test_captioning_builds_data_url_and_returns_caption(monkeypatch):
    image = __import__("pathlib").Path("test-caption-diagram.png")
    image.write_bytes(b"image bytes")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://endpoint")
    monkeypatch.setenv("AZURE_OPENAI_LLM_DEPLOYMENT", "vision")
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  caption  "))])
    client = MagicMock()
    client.chat.completions.create.return_value = response
    client_type = MagicMock(return_value=client)
    monkeypatch.setattr(captioning, "AzureOpenAIClient", client_type)

    try:
        assert captioning.generate_caption(str(image), "describe") == "caption"
        message = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert message[0]["text"] == "describe"
        assert message[1]["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(b"image bytes").decode()
    finally:
        image.unlink()


def test_captioning_requires_configuration(monkeypatch):
    image = __import__("pathlib").Path("test-caption-diagram.jpg")
    image.write_bytes(b"image")
    monkeypatch.setattr(captioning, "load_dotenv", lambda: None)
    for name in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_LLM_DEPLOYMENT"):
        monkeypatch.delenv(name, raising=False)
    try:
        with pytest.raises(EnvironmentError):
            captioning.generate_caption(str(image))
    finally:
        image.unlink()


def test_captioning_maps_unknown_extension_to_png(monkeypatch):
    image = __import__("pathlib").Path("test-caption-diagram.bmp")
    image.write_bytes(b"image")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "endpoint")
    monkeypatch.setenv("AZURE_OPENAI_LLM_DEPLOYMENT", "vision")
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    monkeypatch.setattr(captioning, "AzureOpenAIClient", MagicMock(return_value=client))
    try:
        captioning.generate_caption(str(image))
        url = client.chat.completions.create.call_args.kwargs["messages"][0]["content"][1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
    finally:
        image.unlink()


def test_handoff_reference_and_score_decisions(monkeypatch):
    now = handoff_service.datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=handoff_service.datetime.UTC)
    monkeypatch.setattr(handoff_service.secrets, "token_hex", lambda size: "abcdef")
    assert handoff_service.generate_handoff_reference_id(now) == "HO-20260102-030405-ABCDEF"
    assert handoff_service.is_explicit_handoff_request("Please connect me to a human")
    assert not handoff_service.is_explicit_handoff_request("A human is mentioned in this lesson")
    assert handoff_service.evaluate_score(1.0, 1.0, "q", False)["trigger"] is False
    assert handoff_service.evaluate_score(None, 1.0, "q", False)["trigger"] is True
    assert handoff_service.evaluate_score(1.0, 1.0, "q", True)["reason"] == "retrieval returned no context"
    assert handoff_service.evaluate_score(0.1, 1.0, "q", False)["reason"] == "faithfulness below threshold"
    assert handoff_service.evaluate_score(1.0, 0.1, "q", False)["reason"] == "answer relevance below threshold"


@pytest.mark.asyncio
async def test_classify_intent_llm_and_keyword_fallback(monkeypatch):
    llm = SimpleNamespace(acomplete=AsyncMock(return_value=SimpleNamespace(text="MEDICAL")))
    monkeypatch.setattr(handoff_service.Settings, "_llm", llm)
    assert (await handoff_service.classify_medical_intent("pain"))["medical"]
    llm.acomplete.return_value = SimpleNamespace(text="NON_MEDICAL")
    assert not (await handoff_service.classify_medical_intent("weather"))["medical"]
    llm.acomplete.side_effect = RuntimeError("unavailable")
    assert (await handoff_service.classify_medical_intent("medical pain"))["medical"]
    assert not (await handoff_service.classify_medical_intent("travel plans"))["medical"]


@pytest.mark.asyncio
async def test_confidence_score_parsing_and_failure(monkeypatch):
    llm = SimpleNamespace(acomplete=AsyncMock(return_value=SimpleNamespace(text="Confidence: 35")))
    monkeypatch.setattr(handoff_service.Settings, "_llm", llm)
    result = await handoff_service.evaluate_confidence_score("answer", "question", [{"text": "context"}])
    assert result["trigger"] is True and result["confidence"] == 35
    llm.acomplete.return_value = SimpleNamespace(text="100")
    result = await handoff_service.evaluate_confidence_score("answer", "question", [])
    assert result["confidence"] == 100 and result["trigger"] is False
    llm.acomplete.side_effect = RuntimeError("unavailable")
    assert (await handoff_service.evaluate_confidence_score("a", "q", []))["confidence"] == 50


def test_send_handoff_email_missing_and_success(monkeypatch):
    context = {
        "reference_id": "HO-1", "trace_id": "t", "timestamp_utc": "now", "priority": "high",
        "trigger_reason": "low score", "user_metadata": {"email": "a@b.test"},
        "session_id": "s", "query_history": [], "generated_answer": "a",
        "evaluation_scores": {"faithfulness": 0.1, "relevance": 0.2, "confidence": 10},
        "retrieved_chunks": [], "conversation_flow": [],
    }
    for name in ("SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "EMAIL_FROM", "EMAIL_TO"):
        monkeypatch.setattr(handoff_service, name, None)
    handoff_service.send_handoff_email(context)

    server = MagicMock()
    smtp = MagicMock(return_value=server)
    monkeypatch.setattr(handoff_service, "SMTP_HOST", "smtp.test")
    monkeypatch.setattr(handoff_service, "SMTP_PORT", 587)
    monkeypatch.setattr(handoff_service, "SMTP_USERNAME", "user")
    monkeypatch.setattr(handoff_service, "SMTP_PASSWORD", "pass")
    monkeypatch.setattr(handoff_service, "EMAIL_FROM", "from@test")
    monkeypatch.setattr(handoff_service, "EMAIL_TO", "to@test")
    monkeypatch.setattr(handoff_service.smtplib, "SMTP", smtp)
    handoff_service.send_handoff_email(context)
    server.__enter__.return_value.starttls.assert_called_once_with()
    server.__enter__.return_value.login.assert_called_once_with("user", "pass")