import importlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from main.evaluation import dataset
from main.security import security
from main.service.agent_service import AgentService


@pytest.mark.asyncio
async def test_application_module_lifespan_success_and_failure():
    fake_service = MagicMock()
    fake_service.initialize = AsyncMock()
    with patch("main.logs.logging.configure_logging"), patch(
        "main.service.agent_service.AgentService", return_value=fake_service
    ):
        sys.modules.pop("main.app", None)
        application = importlib.import_module("main.app")

    with patch.object(application, "set_agent_service") as setter:
        async with application.lifespan(application.app):
            pass
        setter.assert_called_once_with(fake_service)

    fake_service.initialize.side_effect = RuntimeError("startup unavailable")
    async with application.lifespan(application.app):
        pass

    response = await application.root()
    assert response["status"] == "running"
    assert application.app.version == "1.0.0"


def test_dataset_adds_all_cases_and_main_orchestrates():
    langfuse = MagicMock()
    dataset.add_items(langfuse, "medlearn-test")

    assert langfuse.create_dataset_item.call_count == 15
    first = langfuse.create_dataset_item.call_args_list[0].kwargs
    assert first["dataset_name"] == "medlearn-test"
    assert first["metadata"]["source"] == "medlearn"

    with patch.object(dataset, "get_langfuse_client", return_value=langfuse), patch.object(
        dataset, "create_dataset"
    ) as create, patch.object(dataset, "add_items") as add:
        dataset.main()
    create.assert_called_once_with(langfuse, "medlearn")
    add.assert_called_once_with(langfuse, "medlearn")


def test_dataset_creation_contract():
    langfuse = MagicMock()
    dataset.create_dataset(langfuse, "course")
    assert langfuse.create_dataset.call_args.kwargs["name"] == "course"


def test_validate_token_maps_decode_failures(monkeypatch):
    monkeypatch.setattr(security.jwt, "get_unverified_header", MagicMock(return_value={"kid": "one"}))
    monkeypatch.setattr(security, "_get_rsa_key", MagicMock(return_value={"kid": "one"}))

    for error, detail in (
        (security.jwt.ExpiredSignatureError("expired"), "Token expired"),
        (security.jwt.JWTClaimsError("claims"), "Incorrect claims"),
        (RuntimeError("invalid"), "Unable to parse"),
    ):
        monkeypatch.setattr(security.jwt, "decode", MagicMock(side_effect=error))
        with pytest.raises(HTTPException, match=detail):
            security.validate_token("token")


def test_agent_parser_handles_invalid_numeric_score():
    service = object.__new__(AgentService)
    assert service._parse_response("Sentiment: POSITIVE Confidence: not-a-number") == {
        "sentiment": "POSITIVE", "confidence": 0.0, "model_used": "unknown"
    }
    # A numeric-looking value that cannot be converted exercises defensive parsing.
    with patch("main.service.agent_service.re.search") as search:
        sentiment = MagicMock()
        sentiment.group.return_value = "positive"
        confidence = MagicMock()
        confidence.group.return_value = "invalid"
        search.side_effect = [sentiment, confidence, None]
        assert service._parse_response("response") is None
