import logging
import os
from typing import List, Any

from dotenv import load_dotenv
from llama_index.core.tools import BaseTool

# Note: In a real environment, ensure 'mcp' and 'llama-index-tools-mcp' are installed.
# This implementation mocks the library structure based on the Sprint Doc's API description
# if the specific library version isn't perfectly aligned, but assumes the interface exists.
try:
    from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
    from llama_index.core.tools import FunctionTool
except ImportError:
    # Fallback or mock for demonstration if specific package combination is tricky in generated code
    logging.warning("llama-index-tools-mcp not found. Ensure dependencies are installed.")
    BasicMCPClient = Any
    McpToolSpec = Any
    FunctionTool = Any

load_dotenv()
logger = logging.getLogger(__name__)


def iter_leaf_exceptions(exc: BaseException):
    """Yield actionable leaf errors from Python 3.11 exception groups."""
    children = getattr(exc, "exceptions", None)
    if children:
        for child in children:
            yield from iter_leaf_exceptions(child)
    else:
        yield exc


def root_exception_message(exc: BaseException) -> str:
    """Return all leaf errors, avoiding an unhelpful TaskGroup-only message."""
    leaves = list(iter_leaf_exceptions(exc))
    return "; ".join(f"{type(item).__name__}: {item}" for item in leaves) or str(exc)

class HuggingFaceMCPService:
    """
    Service to handle connections to the Hugging Face MCP Server.
    """

    def __init__(self) -> None:
        """
        Initialize the service with configuration credentials.
        """
        self.url = os.getenv("HF_MCP_URL", "https://huggingface.co/mcp")
        self.token = os.getenv("HF_TOKEN")
        if not self.token:
            raise ValueError("HF_TOKEN environment variable is required")
        self.headers = {"Authorization": f"Bearer {self.token}"}

    async def load_tools(self, allowed_tools: List[str] | None = None) -> List[BaseTool]:
        """
        Connects to the remote MCP server and wraps capabilities as LlamaIndex tools.

        Args:
            allowed_tools (List[str] | None): A list of tool names to filter (e.g., ['search_models']).
                                              If None, returns all discovered tools.

        Returns:
            List[BaseTool]: A list of executable LlamaIndex tools.

        Raises:
            ConnectionError: If connection to MCP server fails.
        """
        logger.info("Connecting to MCP Server at %s", self.url)
        
        try:
            # 1. Initialize Client
            # 
            # NOTE: You may see a 405 "Method Not Allowed" error in logs when the client first connects.
            # This is expected behavior and does not indicate a failure:
            # 
            # - BasicMCPClient initially attempts to use GET requests for SSE (Server-Sent Events) streaming
            # - The HuggingFace MCP server only supports POST requests (not GET for SSE)
            # - The client automatically detects the 405 error and retries using POST-based streamable-http transport
            # - The connection succeeds after the automatic retry
            # 
            # This retry mechanism is built into the MCP client library and ensures compatibility
            # with servers that only support POST-based transports (like HuggingFace's MCP server).
            client = BasicMCPClient(
                command_or_url=self.url,
                headers=self.headers
            )
            
            # 2. Wrap as LlamaIndex Tools
            tool_spec = McpToolSpec(client=client)
            
            # 3. Get all tools using async method
            discovered_tools = await tool_spec.to_tool_list_async()

            # McpToolSpec's generated sync function bridges its async function by
            # creating another event loop. BasicMCPClient, however, owns an
            # httpx.AsyncClient created on this discovery loop. Reusing it from
            # the bridge is unsafe and caused intermittent AnyIO TaskGroup errors.
            # Each executable wrapper therefore constructs its client on the same
            # loop that performs the call. Discovery and execution remain MCP-based.
            tools = []
            for discovered in discovered_tools:
                metadata = discovered.metadata
                schema = metadata.fn_schema.model_json_schema()
                if metadata.name == "hub_repo_search":
                    metadata.description += (
                        " Use only fields present in this tool's schema. For model inference "
                        "searches, use repo_types=[\"model\"], put pipeline/task and language "
                        "tags in filters (for example [\"text-classification\", \"language:en\"]), "
                        "and prefer sort=\"downloads\". Do not send pipeline_tag or language fields."
                    )
                if metadata.name == "hub_repo_details" and "repo_ids" in schema.get("properties", {}):
                    metadata.description += (
                        " Call only when search-result metadata is insufficient for model selection. "
                        "Pass repo_ids as a non-empty JSON array, for example "
                        '{"repo_ids": ["some-org/some-model"]}. Never pass repo_id. For inference, '
                        "use overview details to prefer a model with a live inference provider."
                    )
                logger.info(
                    "mcp.tool.discovered name=%s description=%s input_schema=%s",
                    metadata.name,
                    metadata.description,
                    schema,
                )

                async def invoke(_tool_name=metadata.name, **arguments):
                    logger.debug("mcp.tool.invoke name=%s arguments=%s", _tool_name, arguments)
                    invocation_client = BasicMCPClient(
                        command_or_url=self.url,
                        headers=self.headers,
                    )
                    try:
                        result = await invocation_client.call_tool(_tool_name, arguments)
                        logger.info(
                            "mcp.tool.result name=%s status=%s",
                            _tool_name,
                            "error" if getattr(result, "isError", False) else "success",
                        )
                        return result
                    except Exception as exc:
                        for leaf in iter_leaf_exceptions(exc):
                            logger.error(
                                "mcp.tool.error name=%s root_exception_type=%s root_exception_message=%s",
                                _tool_name,
                                type(leaf).__name__,
                                leaf,
                            )
                        raise
                    finally:
                        await invocation_client.http_client.aclose()

                tools.append(FunctionTool.from_defaults(async_fn=invoke, tool_metadata=metadata))

            # The discovery client is no longer captured by returned tools.
            await client.http_client.aclose()
            
            # 4. Filter specific tools if requested
            if allowed_tools:
                filtered_tools = [t for t in tools if t.metadata.name in allowed_tools]
                logger.info(f"Filtered to {len(filtered_tools)} tools from {len(tools)} available")
                logger.info(f"Successfully loaded {len(filtered_tools)} tools: {[t.metadata.name for t in filtered_tools]}")
                return filtered_tools
                
            logger.info(f"Successfully loaded {len(tools)} tools: {[t.metadata.name for t in tools]}")
            return tools
            
        except Exception as exc:
            for leaf in iter_leaf_exceptions(exc):
                logger.error(
                    "Failed to load MCP tools: root_exception_type=%s root_exception_message=%s",
                    type(leaf).__name__,
                    leaf,
                )
            raise ConnectionError(
                f"Could not connect to HF MCP Server: {root_exception_message(exc)}"
            ) from exc
