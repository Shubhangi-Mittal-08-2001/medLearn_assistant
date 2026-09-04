import json
import os
import time
import logging
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, ConfigDict, Field

# Import HuggingFace library with fallback
try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    import requests

logger = logging.getLogger(__name__)


def _exception_detail(exc: Exception) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


class HFInferenceInput(BaseModel):
    """Canonical public argument contract for the hf_inference tool."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Exact Hugging Face model repository ID returned by hub_repo_search, "
            "in organization/model-name form."
        ),
    )
    text: str = Field(
        ...,
        min_length=1,
        description="The input text that the selected model should classify or analyze.",
    )

def hf_inference_api(model_id: str, text: str) -> str:
    """
    Runs inference on a specific Hugging Face model using the Inference API.
    This tool is a simple wrapper around the HuggingFace Inference API.
    
    Use this ONLY when you have a specific model ID from hub_repo_search.
    If a model fails, ask the agent to search for a different model instead of 
    using hardcoded fallbacks.
    
    Args:
        model_id: Exact model repository ID returned by hub_repo_search.
        text (str): The text to analyze.
        
    Returns:
        str: JSON-formatted response containing results and scores, or an error message.
    """
    if not model_id or not model_id.strip():
        raise ValueError("model_id is required")
    if not text or not text.strip():
        raise ValueError("text is required")
    model_id = model_id.strip()
    text = text.strip()

    logger.info("[HF_INFERENCE] Starting inference with model_id=%s, text_length=%d", model_id, len(text))
    
    try:
        # Try using HuggingFace Hub library if available
        if HF_HUB_AVAILABLE:
            logger.debug("[HF_INFERENCE] Using HuggingFace Hub client for inference")
            return _call_with_hub_client(model_id, text)
        else:
            # Fallback to requests library
            logger.debug("[HF_INFERENCE] Using requests library fallback for inference")
            return _call_with_requests(model_id, text)
    except Exception as e:
        logger.error(f"[HF_INFERENCE] Unexpected error during inference: {str(e)}", exc_info=True)
        raise

def _call_with_hub_client(model_id: str, text: str) -> str:
    """
    Call inference using the HuggingFace Hub client library.
    
    Args:
        model_id (str): The ID of the model.
        text (str): The text to analyze.
        
    Returns:
        str: JSON-formatted response or error message.
    """
    logger.debug(f"[HF_HUB_CLIENT] Initializing InferenceClient")
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("[HF_HUB_CLIENT] HF_TOKEN environment variable is not set")
        raise ValueError("HF_TOKEN environment variable is required")
    
    # `provider=None` means automatic provider selection. Hub repositories with
    # no Inference Providers mapping make that path raise a bare StopIteration.
    # hf-inference is Hugging Face's own hosted inference provider and can
    # evaluate the selected repository directly (or return an actionable HTTP
    # error when that repository is not hosted).
    provider = os.getenv("HF_INFERENCE_PROVIDER", "auto")
    timeout = float(os.getenv("HF_INFERENCE_TIMEOUT", "30"))
    client = InferenceClient(token=hf_token, provider=provider, timeout=timeout)
    logger.debug(f"[HF_HUB_CLIENT] InferenceClient initialized successfully")
    
    try:
        logger.info(f"[HF_HUB_CLIENT] Calling text_classification for model: {model_id}")
        # Try text classification first
        result = client.text_classification(text, model=model_id)
        
        logger.info(f"[HF_HUB_CLIENT] Classification succeeded for model {model_id}, got {len(result)} results")
        logger.debug(f"[HF_HUB_CLIENT] Classification results: {result}")
        
        return json.dumps([{"label": r.label, "score": r.score} for r in result], indent=2)
    except Exception as e:
        error_msg = _exception_detail(e)
        logger.error(f"[HF_HUB_CLIENT] Classification failed for model '{model_id}': {error_msg}", exc_info=True)
        
        # Return detailed error to help agent decide next steps
        return json.dumps({
            "error": f"Model '{model_id}' failed",
            "details": error_msg,
            "suggestion": "Try searching for a different model using hub_repo_search. Look for models with high downloads and a text-classification task."
        }, indent=2)

def _call_with_requests(model_id: str, text: str) -> str:
    """
    Fallback method using requests library directly.
    
    Args:
        model_id (str): The ID of the model.
        text (str): The text to analyze.
        
    Returns:
        str: JSON-formatted response or error message.
    """
    import requests
    
    logger.debug(f"[HF_REQUESTS] Using requests library fallback for model: {model_id}")
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("[HF_REQUESTS] HF_TOKEN environment variable is not set")
        raise ValueError("HF_TOKEN environment variable is required")
    
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": text}
    
    logger.debug(f"[HF_REQUESTS] Sending POST request to {api_url}")
    
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        logger.debug(f"[HF_REQUESTS] Received response with status code: {response.status_code}")
        
        if response.status_code == 200:
            logger.info(f"[HF_REQUESTS] Inference succeeded for model {model_id}")
            return response.text
        elif response.status_code == 503:
            # Model is loading - wait and retry once
            logger.warning(f"[HF_REQUESTS] Model {model_id} is loading (HTTP 503), retrying after 5 seconds...")
            time.sleep(5)
            
            logger.debug(f"[HF_REQUESTS] Retrying POST request to {api_url}")
            response = requests.post(api_url, json=payload, headers=headers, timeout=30)
            logger.debug(f"[HF_REQUESTS] Retry response status code: {response.status_code}")
            
            if response.status_code == 200:
                logger.info(f"[HF_REQUESTS] Inference succeeded on retry for model {model_id}")
                return response.text
            else:
                logger.error(f"[HF_REQUESTS] Retry failed with status {response.status_code} for model {model_id}")
                return json.dumps({
                    "error": "Model is still loading",
                    "status_code": response.status_code,
                    "suggestion": "Try a different model with more downloads (likely already loaded)"
                }, indent=2)
        else:
            logger.error(f"[HF_REQUESTS] HTTP {response.status_code} error for model {model_id}: {response.text[:200]}")
            return json.dumps({
                "error": f"HTTP {response.status_code}",
                "details": response.text[:200],
                "suggestion": "Search for a different model using hub_repo_search with a text-classification filter and sort by downloads."
            }, indent=2)
    except Exception as e:
        logger.error(f"[HF_REQUESTS] Request failed for model {model_id}: {str(e)}", exc_info=True)
        return json.dumps({
            "error": "Request failed",
            "details": str(e),
            "suggestion": "Try searching for a different model with hub_repo_search"
        }, indent=2)

# Create the LlamaIndex tool
custom_inference_tool = FunctionTool.from_defaults(
    fn=hf_inference_api,
    name="hf_inference",
    fn_schema=HFInferenceInput,
    description="""Run inference using a Hugging Face model.

Arguments (use these exact names):
- model_id: exact model repository ID returned by hub_repo_search, in "org/model-name" form
- text: input text to classify or analyze

Call this tool only after hub_repo_search returns a valid model ID. Call it as:
hf_inference(model_id="some-org/some-model", text="The patient has severe chest pain.")
Never use model, repo_id, engine, or inputs as argument names. No model is selected by default.""",
)
