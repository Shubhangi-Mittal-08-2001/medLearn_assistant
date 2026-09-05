"""Image Captioning Service Module
Handles image captioning operations.

This module provides:
- Caption generation interface for extracted images
"""
import logging
from typing import Optional
import os
from pathlib import Path
import base64
from dotenv import load_dotenv
from openai import AzureOpenAI as AzureOpenAIClient

logger = logging.getLogger(__name__)


def generate_caption(image_path: str, prompt: Optional[str] = None) -> str:
    """
    Generate a caption for an image.
    
    Args:
        image_path: Path to the image file
        prompt: Optional custom prompt for caption generation
        
    Returns:
        str: Generated caption text
    """
    logger.info(f"Generating caption for image: {image_path}")
    load_dotenv()

    if prompt is None:
        prompt = (
            "Describe the content and purpose of this image in a concise caption, "
            "focusing on any nutritional or dietary information."
        )

    # TODO: Implement image captioning and return a caption string.
    #
    # 1) Read image bytes
    # with open(image_path, "rb") as f:
    #     image_bytes = f.read()
    #
    # 2) Call your vision model (e.g., Azure OpenAI vision / OpenAI vision)
    # HINT: Provide both the prompt text and the image payload.
    # HINT: If your API expects a data URL, base64-encode the image bytes.
    #
    # 3) Parse response and return caption string
    # caption = ...
    # return caption

    logger.debug(f"Reading and encoding image: {image_path}")
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        image_data = base64.b64encode(image_bytes).decode("utf-8")
    
    image_size_mb = len(image_bytes) / (1024 * 1024)
    logger.debug(f"Image size: {image_size_mb:.2f} MB, base64 encoded length: {len(image_data)} characters")
    
    # Determine image MIME type from file extension
    ext = Path(image_path).suffix.lower()
    mime_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_type_map.get(ext, "image/png")
    logger.debug(f"Detected MIME type: {mime_type} for extension: {ext}")
    
    # Get Azure OpenAI configuration
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    llm_deployment = os.environ.get("AZURE_OPENAI_LLM_DEPLOYMENT")
    
    if not all([api_key, endpoint, llm_deployment]):
        logger.error("Missing required environment variables for Azure OpenAI")
        raise EnvironmentError(
            "Missing required environment variables for Azure OpenAI"
        )
    
    logger.debug(f"Initializing Azure OpenAI client: endpoint={endpoint}, deployment={llm_deployment}")
    client = AzureOpenAIClient(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )
    
    # Make the vision API call using chat completions format
    logger.info(f"Calling Azure OpenAI vision API for caption generation (deployment: {llm_deployment})")
    response = client.chat.completions.create(
        model=llm_deployment,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}"
                        }
                    }
                ]
            }
        ],
        # max_tokens=300,
    )
    
    caption = response.choices[0].message.content.strip()
    logger.info(f"Caption generated successfully (length: {len(caption)} characters): {caption[:100]}...")
    
    return caption
