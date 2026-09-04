from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from main.service.mcp_client import MedicalMCPService, RemoteMCPService


def test_medical_mcp_is_optional(monkeypatch):
    monkeypatch.delenv("MEDICAL_MCP_URL", raising=False)
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    assert MedicalMCPService.from_environment() is None


def test_medical_mcp_reads_url_token_and_allow_list(monkeypatch):
    monkeypatch.setenv("MEDICAL_MCP_URL", "https://medical.example/mcp")
    monkeypatch.setenv("MEDICAL_MCP_TOKEN", "secret-token")
    monkeypatch.setenv("MEDICAL_MCP_ALLOWED_TOOLS", "pubmed_search, article_details")
    service = MedicalMCPService.from_environment()
    assert service is not None
    assert service.url == "https://medical.example/mcp"
    assert service.allowed_tools == {"pubmed_search", "article_details"}


@pytest.mark.asyncio
async def test_remote_mcp_discovers_filters_and_executes_tools():
    search = MagicMock()
    search.metadata.name = "pubmed_search"
    search.metadata.description = "Search PubMed"
    search.metadata.get_name.return_value = "pubmed_search"
    other = MagicMock()
    other.metadata.name = "unrelated_tool"
    other.metadata.description = "Other"
    discovery_client = MagicMock()
    discovery_client.http_client.aclose = AsyncMock()
    invocation_client = MagicMock()
    invocation_client.http_client.aclose = AsyncMock()
    invocation_client.call_tool = AsyncMock(return_value={"articles": []})
    spec = MagicMock()
    spec.to_tool_list_async = AsyncMock(return_value=[search, other])

    with patch("main.service.mcp_client.BasicMCPClient", side_effect=[discovery_client, invocation_client]), patch(
        "main.service.mcp_client.McpToolSpec", return_value=spec
    ):
        service = RemoteMCPService("https://medical.example/mcp", token="secret-token",
                                   name="medical", allowed_tools=["pubmed_search"])
        tools = await service.load_tools()
        assert [tool.metadata.name for tool in tools] == ["pubmed_search"]
        result = await tools[0].acall(query="warfarin interactions")

    invocation_client.call_tool.assert_awaited_once_with(
        "pubmed_search", {"query": "warfarin interactions"}
    )
    assert result.raw_output == {"articles": []}
    discovery_client.http_client.aclose.assert_awaited_once()
    invocation_client.http_client.aclose.assert_awaited_once()
