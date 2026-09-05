from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import BackgroundTasks, HTTPException, UploadFile

from main.routes import routes
from main.schema.models import QueryRequest


def user():
    return {"user_id": "u1", "email": "learner@example.com", "role": "user"}


def decision():
    return SimpleNamespace(
        allowed=True,
        sanitized_text="question",
        phi_detected=False,
        phi_summaries=[],
        reason=None,
    )


def request(question="How does pain work?"):
    return QueryRequest(question=question, user_email="learner@example.com")


def test_sanitize_filename_handles_traversal_special_characters_and_length():
    assert routes.sanitize_filename(r"..\secret/report?.pdf") == "report.pdf"
    assert len(routes.sanitize_filename("a" * 300 + ".pdf")) == 254

    with pytest.raises(ValueError, match="cannot be empty"):
        routes.sanitize_filename("")


def test_guarded_response_sanitizes_sources_and_deduplicates_summaries(monkeypatch):
    output = SimpleNamespace(sanitized_text="safe answer", output_sanitized=True, blocked=False, reason=None, pii_summaries=["email"])
    source = SimpleNamespace(sanitized_text="safe source", output_sanitized=True, blocked=False, reason=None, pii_summaries=["email", "phone"])
    validator = MagicMock()
    validator.validate_and_sanitize_output.side_effect = [output, source]
    monkeypatch.setattr(routes, "_guardrails_validator", validator)

    response = routes._guarded_response("q", "raw", [{"text": "private", "content": "private", "source": "x"}], decision())

    assert response.answer == "safe answer"
    assert response.source_nodes[0]["text"] == "safe source"
    assert response.source_nodes[0]["content"] == "safe source"
    assert response.validation_results.pii_summaries == ["email", "phone"]


@pytest.mark.asyncio
async def test_get_current_user_auth_disabled(monkeypatch):
    monkeypatch.setattr(routes, "AUTH_ENABLED", False)
    result = await routes.get_current_user(None)
    assert result["user_id"] == "dev-user"


@pytest.mark.asyncio
async def test_get_current_user_auth_enabled_validates_roles(monkeypatch):
    monkeypatch.setattr(routes, "AUTH_ENABLED", True)
    monkeypatch.setattr(routes, "validate_token", lambda token: {"sub": "a", "email": "a@x.test", "https://medlearn.local/roles": "admin"})
    result = await routes.get_current_user("token")
    assert result == {"user_id": "a", "email": "a@x.test", "role": "admin", "iss": None, "aud": None}


@pytest.mark.asyncio
async def test_get_current_user_rejects_missing_and_invalid_tokens(monkeypatch):
    monkeypatch.setattr(routes, "AUTH_ENABLED", True)
    with pytest.raises(HTTPException) as missing:
        await routes.get_current_user(None)
    assert missing.value.status_code == 401

    monkeypatch.setattr(routes, "validate_token", MagicMock(side_effect=RuntimeError("bad")))
    with pytest.raises(HTTPException) as invalid:
        await routes.get_current_user("token")
    assert invalid.value.status_code == 401


@pytest.mark.asyncio
async def test_upload_success_and_processing_error(monkeypatch):
    monkeypatch.setattr(routes, "process_document", MagicMock(return_value={"message": "indexed", "document_indexed": 2}))
    upload = UploadFile(filename="lesson.pdf", file=__import__("io").BytesIO(b"pdf"))
    result = await routes.upload_document(upload, user())
    assert result.message == "indexed"
    assert result.documents_indexed == 2

    monkeypatch.setattr(routes, "process_document", MagicMock(side_effect=RuntimeError("broken")))
    upload = UploadFile(filename="lesson.pdf", file=__import__("io").BytesIO(b"pdf"))
    with pytest.raises(HTTPException) as error:
        await routes.upload_document(upload, user())
    assert error.value.status_code == 500


@pytest.mark.asyncio
async def test_upload_rejects_large_file(monkeypatch):
    upload = UploadFile(filename="lesson.pdf", file=__import__("io").BytesIO(b"x" * (routes.MAX_FILE_SIZE + 1)))
    with pytest.raises(HTTPException) as error:
        await routes.upload_document(upload, user())
    assert error.value.detail == f"File size exceeds {routes.MAX_FILE_SIZE_KB} KB limit"


@pytest.mark.asyncio
async def test_stream_emits_chunks_and_done(monkeypatch):
    response = routes.QueryResponse(query="q", answer="x" * 161, source_nodes=[])
    monkeypatch.setattr(routes, "query_rag", AsyncMock(return_value=response))
    streamed = await routes.query_rag_stream(request("q"), BackgroundTasks(), user())
    body = "".join([chunk async for chunk in streamed.body_iterator])
    assert "event: chunk" in body
    assert "event: done" in body


@pytest.mark.asyncio
async def test_root_contract():
    result = await routes.root()
    assert result["message"] == "MedLearn AI Assistant"
    assert "POST /query" in result["endpoints"]
