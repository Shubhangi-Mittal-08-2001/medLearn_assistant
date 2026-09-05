"""Centralised instructions for MedLearn's tool-selecting agent."""

from __future__ import annotations
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPT_VERSION = "medlearn-agent-v2.6"

from langfuse import get_client

langfuse = get_client()



def get_agent_system_prompt() -> str:
    prompt = langfuse.get_prompt("MEDLEARN_SYSTEM_PROMPT")
    logger.info("get_agent_system_prompt done successfully")
    return prompt.prompt


__all__ = ["PROMPT_VERSION", "get_agent_system_prompt"]
