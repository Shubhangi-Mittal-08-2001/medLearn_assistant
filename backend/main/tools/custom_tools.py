import json
import os
import time
import logging
from llama_index.core.tools import FunctionTool

# Import HuggingFace library with fallback
try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    import requests

logger = logging.getLogger(__name__)

def hf_inference_api(model_id: str, text: str) -> str:
    """
    Runs inference on a specific Hugging Face model using the Inference API.
    This tool is a simple wrapper around the HuggingFace Inference API.
    
    Use this ONLY when you have a specific model ID from the model_search MCP tool.
    If a model fails, ask the agent to search for a different model instead of 
    using hardcoded fallbacks.
    
    Args:
        model_id: Exact model ID from HuggingFace medical models:
    - pritamdeka/PubMedBERT-MNLI-MedNLI (NLI classification)
    - bvanaken/clinical-assertion-negation-bert (assertion/negation detection)
        text (str): The text to analyze.
        
    Returns:
        str: JSON-formatted response containing results and scores, or an error message.
    """
    logger.info(f"[HF_INFERENCE] Starting inference with model_id={model_id}, text_length={len(text)}")
    
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
    
    client = InferenceClient(token=hf_token)
    logger.debug(f"[HF_HUB_CLIENT] InferenceClient initialized successfully")
    
    try:
        logger.info(f"[HF_HUB_CLIENT] Calling text_classification for model: {model_id}")
        # Try text classification first
        result = client.text_classification(text, model=model_id)
        
        logger.info(f"[HF_HUB_CLIENT] Classification succeeded for model {model_id}, got {len(result)} results")
        logger.debug(f"[HF_HUB_CLIENT] Classification results: {result}")
        
        return json.dumps([{"label": r.label, "score": r.score} for r in result], indent=2)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[HF_HUB_CLIENT] Classification failed for model '{model_id}': {error_msg}", exc_info=True)
        
        # Return detailed error to help agent decide next steps
        return json.dumps({
            "error": f"Model '{model_id}' failed",
            "details": error_msg,
            "suggestion": "Try searching for a different model using model_search MCP tool. Look for models with high downloads and 'text-classification' task."
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
                "suggestion": "Search for a different model using model_search. Filter by task='text-classification' and high downloads."
            }, indent=2)
    except Exception as e:
        logger.error(f"[HF_REQUESTS] Request failed for model {model_id}: {str(e)}", exc_info=True)
        return json.dumps({
            "error": "Request failed",
            "details": str(e),
            "suggestion": "Try searching for a different model with model_search"
        }, indent=2)

# Create the LlamaIndex tool
custom_inference_tool = FunctionTool.from_defaults(
    fn=hf_inference_api,
    name="hf_inference",
    description="""Runs inference on a Hugging Face model using the Inference API.
    
    IMPORTANT: Use this tool ONLY after getting a specific model ID from model_search.
    If inference fails, DO NOT retry with the same model. Instead:
    1. Use model_search to find a different model
    2. Look for models with high downloads (more likely to have active endpoints)
    3. Try the new model with this tool
    
    Args:
        model_id: Exact model ID from HuggingFace medical models:
    - pritamdeka/PubMedBERT-MNLI-MedNLI (NLI classification)
    - bvanaken/clinical-assertion-negation-bert (assertion/negation detection)
        text: Text to analyze
    
    Returns: JSON with results or error with suggestions"""
)
