from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from main.routes import routes


@pytest.fixture
def api_app():
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_current_user] = lambda: {
        "user_id": "test-user",
        "email": "learner@example.com",
        "role": "user",
    }
    return app


@pytest.fixture
async def client(api_app):
    async with AsyncClient(
        transport=ASGITransport(app=api_app), base_url="http://test"
    ) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def restore_agent_service():
    original = routes._agent_service
    yield
    routes._agent_service = original


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_query_rejects_invalid_payload_before_calling_services(client):
    response = await client.post("/query", json={"question": "", "user_email": "bad"})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_blocks_destructive_database_request(client, monkeypatch):
    classify = AsyncMock()
    monkeypatch.setattr(routes, "classify_medical_intent", classify)

    response = await client.post(
        "/query",
        json={
            "question": "delete from medical_records where id = 1",
            "user_email": "learner@example.com",
        },
    )

    assert response.status_code == 400
    assert "read-only" in response.json()["detail"]
    classify.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_returns_agent_answer_and_source_metadata(client, monkeypatch):
    monkeypatch.setattr(
        routes,
        "classify_medical_intent",
        AsyncMock(return_value={"medical": True, "reason": "medical education"}),
    )
    monkeypatch.setattr(routes, "get_langfuse_client", MagicMock(side_effect=RuntimeError("disabled")))
    service = SimpleNamespace(
        execute_query=AsyncMock(
            return_value=SimpleNamespace(
                answer="Pacinian corpuscles detect rapid vibration.",
                source_nodes=[{"text": "Pacinian evidence", "source": "lesson.pdf", "score": 0.91}],
                tools_used=["semantic_search"],
                tool_errors=[],
                routing="agent:semantic_search",
                evaluation={"mode": "rag", "faithfulness": 0.9, "relevance": 0.9, "overall_confidence": 0.9},
            )
        )
    )
    routes._agent_service = service

    response = await client.post(
        "/query",
        json={
            "question": "How do Pacinian corpuscles detect vibration?",
            "user_email": "learner@example.com",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Pacinian corpuscles detect rapid vibration."
    assert body["tools_used"] == ["semantic_search"]
    assert body["routing"] == "agent:semantic_search"
    assert body["source_nodes"][0]["source"] == "lesson.pdf"
    assert body["validation_results"]["input_allowed"] is True
    service.execute_query.assert_awaited_once_with(
        "How do Pacinian corpuscles detect vibration?"
    )


@pytest.mark.asyncio
async def test_upload_rejects_non_pdf_without_processing(client, monkeypatch):
    process_document = MagicMock()
    monkeypatch.setattr(routes, "process_document", process_document)

    response = await client.post(
        "/upload", files={"file": ("notes.txt", b"medical notes", "text/plain")}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported"
    process_document.assert_not_called()
