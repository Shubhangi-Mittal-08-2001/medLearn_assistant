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
except ImportError:
    # Fallback or mock for demonstration if specific package combination is tricky in generated code
    logging.warning("llama-index-tools-mcp not found. Ensure dependencies are installed.")
    BasicMCPClient = Any
    McpToolSpec = Any

load_dotenv()
logger = logging.getLogger(__name__)

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
        logger.info(f"Connecting to MCP Server at {self.url}...")
        
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
            tools = await tool_spec.to_tool_list_async()
            
            # 4. Filter specific tools if requested
            if allowed_tools:
                filtered_tools = [t for t in tools if t.metadata.name in allowed_tools]
                logger.info(f"Filtered to {len(filtered_tools)} tools from {len(tools)} available")
                logger.info(f"Successfully loaded {len(filtered_tools)} tools: {[t.metadata.name for t in filtered_tools]}")
                return filtered_tools
                
            logger.info(f"Successfully loaded {len(tools)} tools: {[t.metadata.name for t in tools]}")
            return tools
            
        except Exception as e:
            logger.error(f"Failed to load MCP tools: {str(e)}")
            raise ConnectionError(f"Could not connect to HF MCP Server: {e}")
