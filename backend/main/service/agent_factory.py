import logging
import os
from typing import List

from dotenv import load_dotenv
from llama_index.core.tools import BaseTool
from llama_index.llms.azure_openai import AzureOpenAI

# Import ReActAgent from workflow module (0.14.10 API)
from llama_index.core.agent.workflow import ReActAgent
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

load_dotenv()
logger = logging.getLogger(__name__)

def create_llm():
    """
    Factory to create the Azure OpenAI LLM instance.
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    llm_deployment = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
    if not azure_endpoint or not llm_deployment:
        raise ValueError("Azure OpenAI configuration required. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_LLM_DEPLOYMENT.")
    
    if not api_key:
        raise ValueError("Azure OpenAI API key required. Set AZURE_OPENAI_API_KEY.")
    
    logger.info("Initializing Azure OpenAI LLM...")
    
    return AzureOpenAI(
        engine=llm_deployment,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        temperature=1.0
    )

def create_embel_model():
    logger.info("Initializing Azure OpenAI Embed Model...")
  
    _embed_model = AzureOpenAIEmbedding(
        model=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-3-small'),
        deployment_name=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
    )
    return _embed_model

def create_agent(tools: List[BaseTool], verbose: bool = True) -> ReActAgent:
    """
    Creates a ReActAgent configured with the provided tools and LLM.
    Uses the 0.14.10 workflow API.

    Args:
        tools (List[BaseTool]): The list of MCP-derived tools.
        verbose (bool): Whether to print chain-of-thought reasoning (not used in workflow API).

    Returns:
        ReActAgent: An initialized agent ready for use.
    """
    llm = create_llm()
    
    # In 0.14.10, ReActAgent is instantiated directly (no from_tools)
    agent = ReActAgent(
        tools=tools,
        llm=llm
    )
    
    return agent