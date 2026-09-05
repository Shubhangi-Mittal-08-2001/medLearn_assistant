from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from main.service.mcp_client import (
    HuggingFaceMCPService,
    MedicalMCPService,
    RemoteMCPService,
    root_exception_message,
)


def test_remote_mcp_requires_a_url():
    with pytest.raises(ValueError, match="URL is required"):
        RemoteMCPService("  ", name="pubmed")


def test_remote_mcp_builds_authorization_header_only_when_token_exists():
    assert RemoteMCPService("https://example.test/mcp").headers == {}
    assert RemoteMCPService("https://example.test/mcp", token="secret").headers == {
        "Authorization": "Bearer secret"
    }


def test_root_exception_message_flattens_exception_groups():
    error = ExceptionGroup("wrapper", [ValueError("bad query"), RuntimeError("offline")])

    assert root_exception_message(error) == "ValueError: bad query; RuntimeError: offline"


@pytest.mark.asyncio
async def test_discovery_failure_is_wrapped_and_client_is_closed():
    client = MagicMock()
    client.http_client.aclose = AsyncMock()
    spec = MagicMock()
    spec.to_tool_list_async = AsyncMock(side_effect=RuntimeError("server offline"))

    with patch("main.service.mcp_client.BasicMCPClient", return_value=client), patch(
        "main.service.mcp_client.McpToolSpec", return_value=spec
    ):
        service = RemoteMCPService("https://example.test/mcp", name="pubmed")
        with pytest.raises(ConnectionError, match="RuntimeError: server offline"):
            await service.load_tools()

    client.http_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_discovered_tools_are_filtered_invoked_and_closed():
    selected_tool = MagicMock()
    selected_tool.metadata.name = "search_articles"
    selected_tool.metadata.description = "Search PubMed"
    selected_tool.metadata.get_name.return_value = "search_articles"
    ignored_tool = MagicMock()
    ignored_tool.metadata.name = "delete_article"
    ignored_tool.metadata.description = "Not allowed"
    discovery_client = MagicMock()
    discovery_client.http_client.aclose = AsyncMock()
    invocation_client = MagicMock()
    invocation_client.http_client.aclose = AsyncMock()
    invocation_client.call_tool = AsyncMock(return_value={"articles": []})
    spec = MagicMock()
    spec.to_tool_list_async = AsyncMock(return_value=[selected_tool, ignored_tool])

    with patch(
        "main.service.mcp_client.BasicMCPClient",
        side_effect=[discovery_client, invocation_client],
    ), patch("main.service.mcp_client.McpToolSpec", return_value=spec):
        service = RemoteMCPService(
            "https://example.test/mcp", allowed_tools=["search_articles"]
        )
        tools = await service.load_tools()
        result = await tools[0].acall(query="diabetes")

    assert [tool.metadata.name for tool in tools] == ["search_articles"]
    assert result.raw_output == {"articles": []}
    invocation_client.call_tool.assert_awaited_once_with(
        "search_articles", {"query": "diabetes"}
    )
    discovery_client.http_client.aclose.assert_awaited_once()
    invocation_client.http_client.aclose.assert_awaited_once()


def test_huggingface_service_requires_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ValueError, match="HF_TOKEN is required"):
        HuggingFaceMCPService()


def test_environment_specific_services_use_configured_values(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.setenv("HF_MCP_URL", "https://hf.example.test/mcp")
    monkeypatch.setenv("MEDICAL_MCP_URL", "https://pubmed.example.test/mcp")
    monkeypatch.setenv("MEDICAL_MCP_TOKEN", "pubmed-test")

    huggingface = HuggingFaceMCPService()
    medical = MedicalMCPService()

    assert huggingface.url == "https://hf.example.test/mcp"
    assert huggingface.headers == {"Authorization": "Bearer hf-test"}
    assert medical.url == "https://pubmed.example.test/mcp"
    assert medical.headers == {"Authorization": "Bearer pubmed-test"}


@pytest.mark.parametrize("name", ["hub_repo_search", "hub_repo_details", "other"])
def test_huggingface_metadata_is_improved_only_for_known_tools(name):
    metadata = SimpleNamespace(name=name, description="Original")
    RemoteMCPService._improve_huggingface_metadata(metadata)

    if name == "hub_repo_search":
        assert "repo_types" in metadata.description
    elif name == "hub_repo_details":
        assert "repo_ids" in metadata.description
    else:
        assert metadata.description == "Original"
