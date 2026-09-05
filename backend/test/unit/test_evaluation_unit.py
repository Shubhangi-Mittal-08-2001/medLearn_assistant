from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from main.evaluation import dataset, dataset_evaluation


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "function_name,response_text,expected",
    [
        ("evaluate_faithfulness_score", "Reasoning: supported\nScore: 1.4", 1.0),
        ("evaluate_faithfulness_score", "no score", 0.0),
        ("evaluate_answer_relevance", "Score: -0.2", 0.0),
        ("evaluate_answer_relevance", "no score", 0.0),
    ],
)
async def test_evaluators_parse_and_clamp_scores(function_name, response_text, expected):
    llm = MagicMock()
    llm.complete.return_value = SimpleNamespace(text=response_text)
    original = dataset_evaluation.Settings._llm
    dataset_evaluation.Settings._llm = llm
    try:
        langfuse = MagicMock()
        if function_name == "evaluate_faithfulness_score":
            result = await dataset_evaluation.evaluate_faithfulness_score(langfuse, "trace", "q", "ctx", "a")
        else:
            result = await dataset_evaluation.evaluate_answer_relevance(langfuse, "trace", "q", "a")
        assert result == expected
    finally:
        dataset_evaluation.Settings._llm = original


@pytest.mark.asyncio
async def test_query_with_evaluation_returns_scores(monkeypatch):
    span = MagicMock(trace_id="trace")
    langfuse = MagicMock()
    langfuse.start_span.return_value = span
    query_engine = MagicMock()
    query_engine.query.return_value = SimpleNamespace(
        source_nodes=[SimpleNamespace(text="context"), SimpleNamespace(no_text=True)],
        __str__=lambda self: "answer",
    )
    monkeypatch.setattr(dataset_evaluation, "get_query_engine", lambda similarity_top_k: query_engine)
    monkeypatch.setattr(dataset_evaluation, "evaluate_faithfulness_score", AsyncMock(return_value=0.9))
    monkeypatch.setattr(dataset_evaluation, "evaluate_answer_relevance", AsyncMock(return_value=0.8))
    result = await dataset_evaluation.query_with_evaluation(langfuse, "question", "expected", 3)
    assert result["trace_id"] == "trace"
    assert result["faithfulness"] == 0.9
    span.end.assert_called_once_with()


def test_dataset_client_requires_keys_and_uses_base_url(monkeypatch):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    with pytest.raises(RuntimeError):
        dataset.get_langfuse_client()

    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.test")
    client_type = MagicMock(return_value="client")
    monkeypatch.setattr(dataset, "Langfuse", client_type)
    assert dataset.get_langfuse_client() == "client"
    assert client_type.call_args.kwargs["host"] == "https://langfuse.test"


def test_dataset_creation_and_items_call_langfuse(monkeypatch):
    langfuse = MagicMock()
    dataset.create_dataset(langfuse, "medlearn")
    dataset.add_items(langfuse, "medlearn")
    langfuse.create_dataset.assert_called_once()
    assert langfuse.create_dataset_item.call_count == 15