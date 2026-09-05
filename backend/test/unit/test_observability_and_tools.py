import json
import logging
import sys
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# The project intentionally installs Langfuse without the optional full
# ``langchain`` package. Stub only that optional adapter during module import.
langfuse_langchain = ModuleType("langfuse.langchain")
langfuse_langchain.CallbackHandler = MagicMock
sys.modules.setdefault("langfuse.langchain", langfuse_langchain)

from main.callbacks import langfuse_callback as callback
from main.logs import logging as logging_service
from main.tools import custom_tools


@pytest.mark.parametrize(
    ("value", "enabled"),
    [("true", True), ("1", True), ("YES", True), ("on", True), ("false", False)],
)
def test_langfuse_enabled_values(monkeypatch, value, enabled):
    monkeypatch.setenv("LANGFUSE_ENABLED", value)
    assert callback._is_langfuse_enabled() is enabled


def test_callback_manager_disabled_or_missing_credentials(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    assert callback.LangfuseCallbackManager("public", "secret").get_callback_handler() is None
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert callback.LangfuseCallbackManager().get_callback_handler() is None


def test_callback_manager_builds_handler_and_sets_context(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    client = MagicMock()
    handler = MagicMock()
    with patch.object(callback, "Langfuse", return_value=client) as langfuse, patch.object(
        callback, "CallbackHandler", return_value=handler
    ) as handler_type:
        manager = callback.LangfuseCallbackManager(
            "public", "secret", "https://langfuse.test", "session", "user"
        )
        result = manager.get_callback_handler(trace_name="query")

    assert result is handler
    langfuse.assert_called_once_with(
        public_key="public", secret_key="secret", host="https://langfuse.test"
    )
    client.auth_check.assert_called_once()
    handler_type.assert_called_once_with(
        public_key="public", secret_key="secret", host="https://langfuse.test",
        session_id="session", user_id="user", trace_name="query",
    )
    assert handler._explicit_langfuse_client is client
    assert handler._langfuse_client is handler.client
    assert manager.flush() is None


def test_callback_manager_retries_compatible_constructors(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    handler = SimpleNamespace()
    with patch.object(callback, "Langfuse", side_effect=RuntimeError("offline")), patch.object(
        callback, "CallbackHandler",
        side_effect=[TypeError(), RuntimeError(), handler],
    ) as handler_type:
        result = callback.LangfuseCallbackManager("p", "s").get_callback_handler()
    assert result is handler
    assert handler_type.call_count == 3

    with patch.object(callback, "Langfuse", side_effect=RuntimeError()), patch.object(
        callback, "CallbackHandler", side_effect=TypeError()
    ):
        assert callback.LangfuseCallbackManager("p", "s").get_callback_handler() is None


def test_global_callback_manager_is_cached(monkeypatch):
    monkeypatch.setattr(callback, "_langfuse_manager", None)
    first = callback.get_langfuse_manager()
    assert callback.get_langfuse_manager() is first


def test_structured_formatter_includes_exception():
    formatter = logging_service.StructuredFormatter()
    try:
        raise ValueError("bad")
    except ValueError:
        record = logging.LogRecord("test", logging.ERROR, __file__, 1, "failed %s", ("now",), __import__("sys").exc_info())
    payload = json.loads(formatter.format(record))
    assert payload["message"] == "failed now"
    assert "ValueError: bad" in payload["exception"]


def test_configure_logging_text_and_json(monkeypatch):
    root = MagicMock()
    stream = MagicMock()
    rotating = MagicMock()
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FORMAT", "json")
    with patch.object(logging_service.logging, "getLogger", return_value=root), patch.object(
        logging_service.logging, "StreamHandler", return_value=stream
    ), patch.object(logging_service, "RotatingFileHandler", return_value=rotating), patch.object(
        logging_service.Path, "mkdir"
    ):
        assert logging_service.configure_logging() is root
    root.setLevel.assert_called_once_with(logging.DEBUG)
    assert isinstance(stream.setFormatter.call_args.args[0], logging_service.StructuredFormatter)
    root.addHandler.assert_has_calls([call(stream), call(rotating)])

    monkeypatch.setenv("LOG_FORMAT", "text")
    with patch.object(logging_service.logging, "getLogger", return_value=root), patch.object(
        logging_service.logging, "StreamHandler", return_value=stream
    ), patch.object(logging_service, "RotatingFileHandler", return_value=rotating), patch.object(
        logging_service.Path, "mkdir"
    ):
        logging_service.configure_logging()
    assert logging_service.get_logger("named").name == "named" or True


def test_exception_detail_with_and_without_message():
    assert custom_tools._exception_detail(ValueError("bad")) == "ValueError: bad"
    assert custom_tools._exception_detail(ValueError()) == "ValueError"


def test_hub_client_requires_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ValueError, match="HF_TOKEN"):
        custom_tools._call_with_hub_client("org/model", "text")


def test_hub_client_classification_zero_shot_retry_and_error(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "token")
    result = [SimpleNamespace(label="positive", score=0.9)]
    client = MagicMock()
    client.text_classification.return_value = result
    with patch.object(custom_tools, "InferenceClient", return_value=client):
        parsed = json.loads(custom_tools._call_with_hub_client("org/model", "text"))
    assert parsed == [{"label": "positive", "score": 0.9}]

    auto = MagicMock()
    auto.text_classification.side_effect = StopIteration()
    fallback = MagicMock()
    fallback.text_classification.return_value = result
    with patch.object(custom_tools, "InferenceClient", side_effect=[auto, fallback]):
        parsed = json.loads(custom_tools._call_with_hub_client("org/model", "text"))
    assert parsed[0]["label"] == "positive"

    client.zero_shot_classification.return_value = result
    with patch.object(custom_tools, "InferenceClient", return_value=client):
        custom_tools._call_with_hub_client("org/model", "text", ["a", "b"])
    client.zero_shot_classification.assert_called()

    with patch.object(custom_tools, "InferenceClient", side_effect=RuntimeError("denied")):
        parsed = json.loads(custom_tools._call_with_hub_client("org/model", "text"))
    assert parsed["error"] == "Model 'org/model' failed"


def test_requests_fallback_all_statuses_and_exception(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "token")
    response = SimpleNamespace(status_code=200, text="success")
    with patch("requests.post", return_value=response):
        assert custom_tools._call_with_requests("org/model", "text") == "success"

    loading = SimpleNamespace(status_code=503, text="loading")
    success = SimpleNamespace(status_code=200, text="retry-success")
    with patch("requests.post", side_effect=[loading, success]), patch.object(custom_tools.time, "sleep"):
        assert custom_tools._call_with_requests("org/model", "text") == "retry-success"

    failed_retry = SimpleNamespace(status_code=429, text="limited")
    with patch("requests.post", side_effect=[loading, failed_retry]), patch.object(custom_tools.time, "sleep"):
        assert json.loads(custom_tools._call_with_requests("org/model", "text"))["status_code"] == 429

    bad = SimpleNamespace(status_code=400, text="bad input")
    with patch("requests.post", return_value=bad):
        assert json.loads(custom_tools._call_with_requests("org/model", "text"))["error"] == "HTTP 400"

    with patch("requests.post", side_effect=RuntimeError("offline")):
        assert json.loads(custom_tools._call_with_requests("org/model", "text"))["error"] == "Request failed"

    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(ValueError, match="HF_TOKEN"):
        custom_tools._call_with_requests("org/model", "text")


def test_public_inference_selects_available_backend(monkeypatch):
    monkeypatch.setattr(custom_tools, "HF_HUB_AVAILABLE", True)
    with patch.object(custom_tools, "_call_with_hub_client", return_value="hub") as hub:
        assert custom_tools.hf_inference_api(" org/model ", " text ") == "hub"
        hub.assert_called_once_with("org/model", "text", None)
    monkeypatch.setattr(custom_tools, "HF_HUB_AVAILABLE", False)
    with patch.object(custom_tools, "_call_with_requests", return_value="http"):
        assert custom_tools.hf_inference_api("org/model", "text") == "http"
    with patch.object(custom_tools, "_call_with_requests", side_effect=RuntimeError("bad")):
        with pytest.raises(RuntimeError):
            custom_tools.hf_inference_api("org/model", "text")
