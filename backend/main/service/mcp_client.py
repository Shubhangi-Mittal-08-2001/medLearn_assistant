"""Reusable clients for remote streamable-HTTP MCP servers."""
from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any

from dotenv import load_dotenv
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

load_dotenv()
logger = logging.getLogger(__name__)


def iter_leaf_exceptions(exc: BaseException):
    children = getattr(exc, "exceptions", None)
    if children:
        for child in children:
            yield from iter_leaf_exceptions(child)
    else:
        yield exc


def root_exception_message(exc: BaseException) -> str:
    leaves = list(iter_leaf_exceptions(exc))
    return "; ".join(f"{type(item).__name__}: {item}" for item in leaves) or str(exc)


async def _close_client(client: Any) -> None:
    close = getattr(getattr(client, "http_client", None), "aclose", None)
    if close is not None:
        await close()


class RemoteMCPService:
    """Discover and execute tools from one authenticated remote MCP server."""

    def __init__(self, url: str, *, token: str | None = None, name: str = "remote_mcp",
                 allowed_tools: Iterable[str] | None = None) -> None:
        if not url or not url.strip():
            raise ValueError(f"{name} URL is required")
        self.url = url.strip()
        self.name = name
        self.allowed_tools = set(allowed_tools or []) or None
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    async def load_tools(self, allowed_tools: list[str] | None = None) -> list[BaseTool]:
        selected = set(allowed_tools or []) or self.allowed_tools
        logger.info("Connecting to %s MCP server at %s", self.name, self.url)
        discovery_client = BasicMCPClient(command_or_url=self.url, headers=self.headers)
        try:
            discovered = await McpToolSpec(client=discovery_client).to_tool_list_async()
            tools: list[BaseTool] = []
            for tool in discovered:
                metadata = tool.metadata
                if selected is not None and metadata.name not in selected:
                    continue
                self._improve_huggingface_metadata(metadata)

                async def invoke(_tool_name=metadata.name, **arguments):
                    client = BasicMCPClient(command_or_url=self.url, headers=self.headers)
                    try:
                        result = await client.call_tool(_tool_name, arguments)
                        logger.info("mcp.tool.result server=%s name=%s status=%s", self.name,
                                    _tool_name, "error" if getattr(result, "isError", False) else "success")
                        return result
                    finally:
                        await _close_client(client)

                tools.append(FunctionTool.from_defaults(async_fn=invoke, tool_metadata=metadata))
                logger.info("mcp.tool.discovered server=%s name=%s", self.name, metadata.name)
            logger.info("Loaded %d tools from %s MCP", len(tools), self.name)
            return tools
        except Exception as exc:
            for leaf in iter_leaf_exceptions(exc):
                logger.error("mcp.connection.error server=%s type=%s message=%s", self.name,
                             type(leaf).__name__, leaf)
            raise ConnectionError(
                f"Could not connect to {self.name} MCP server: {root_exception_message(exc)}"
            ) from exc
        finally:
            await _close_client(discovery_client)

    @staticmethod
    def _improve_huggingface_metadata(metadata: Any) -> None:
        if metadata.name == "hub_repo_search":
            metadata.description += (' For inference searches use repo_types=["model"], put task and '
                                     'language tags in filters, and prefer sort="downloads". Use only schema fields.')
        elif metadata.name == "hub_repo_details":
            metadata.description += (" Pass repo_ids as a non-empty JSON array. Confirm the pipeline task "
                                     "and a live inference provider before selecting a model.")


class HuggingFaceMCPService(RemoteMCPService):
    """Hugging Face Hub MCP integration for repository/model operations."""

    def __init__(self) -> None:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN is required to enable Hugging Face MCP")
        super().__init__(os.getenv("HF_MCP_URL", "https://huggingface.co/mcp"), token=token,
                         name="huggingface")


class MedicalMCPService(RemoteMCPService):
    """Optional external medical API exposed through an MCP server."""

    @classmethod
    def from_environment(cls) -> "MedicalMCPService | None":
        url = os.getenv("MEDICAL_MCP_URL") or os.getenv("MCP_SERVER_URL")
        if not url:
            return None
        allowed = [item.strip() for item in os.getenv("MEDICAL_MCP_ALLOWED_TOOLS", "").split(",")
                   if item.strip()]
        return cls(url, token=os.getenv("MEDICAL_MCP_TOKEN") or os.getenv("MCP_SERVER_TOKEN"),
                   name="medical", allowed_tools=allowed)
