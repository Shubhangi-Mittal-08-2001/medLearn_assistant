import os
import logging
import smtplib
import datetime
import secrets
from email.mime.text import MIMEText
from typing import Dict, Any, Optional
from llama_index.core.settings import Settings


logger = logging.getLogger(__name__)

FAITHFULNESS_THRESHOLD = 0.6
RELEVANCE_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 40

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_FROM = os.getenv("APPLICATION_EMAIL")
EMAIL_TO = os.getenv("SUPPORT_EMAIL")


def generate_handoff_reference_id(now: Optional[datetime.datetime] = None) -> str:
    """Generate a unique handoff reference ID in format: HO-YYYYMMDD-HHMMSS-HEXCODE"""
    # TODO: Generate ID with current UTC time (if now is None) + 3-byte hex token
    # Format: f"HO-{timestamp}-{secrets.token_hex(3).upper()}"
    now = now or datetime.datetime.now(datetime.UTC)
    return f"HO-{now.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3).upper()}"


def evaluate_score(
    faithfulness: Optional[float],
    relevance: Optional[float],
    user_question: str,
    no_chunks: bool
) -> Dict[str, Any]:
    """Determines if the system should escalate based on evaluation scores."""
    # Check triggers in order and return {"trigger": bool, "reason": str}:
    # 1. no_chunks → "retrieval returned no context"
    # 2. faithfulness or relevance is None → "evaluation scores unavailable"
    # 3. faithfulness < 0.6 → "faithfulness below threshold"
    # 4. relevance < 0.5 → "answer relevance below threshold"
    # 5. Keywords in question (human/agent/support/escalate) → "explicit user request"
    # 6. Otherwise → {"trigger": False}
    if no_chunks:
        return {"trigger": True, "reason": "retrieval returned no context"}
    if faithfulness is None or relevance is None:
        return {"trigger": True, "reason": "evaluation scores unavailable"}
    if faithfulness < FAITHFULNESS_THRESHOLD:
        return {"trigger": True, "reason": "faithfulness below threshold"}
    if relevance < RELEVANCE_THRESHOLD:
        return {"trigger": True, "reason": "answer relevance below threshold"}
    if any(keyword in user_question.lower() for keyword in ["human", "agent", "support", "escalate"]):
        return {"trigger": True, "reason": "explicit user request"}
    return {"trigger": False, "reason": "no escalation needed"}


async def evaluate_confidence_score(answer: str) -> Dict[str, Any]:
    """Evaluate LLM confidence score (0-100) and trigger handoff if below threshold (40)."""
    # Use LLM to rate confidence
    # 1. Get Settings.llm (raise ValueError if None)
    # 2. Prompt: "Rate confidence 0–100.\nAnswer: {answer}"
    # 3. Parse score from response (extract digits, default to 50)
    # 4. Return {"trigger": bool, "reason": str, "confidence": int}
    try:
        llm = Settings.llm
        if llm is None:
            raise ValueError("LLM is not configured in Settings.")
        confidence_prompt = f"Rate confidence 0–100.\nAnswer: {answer}"
        confidence_response = await llm.acomplete(confidence_prompt)
        # Extract digits from response
        confidence_score = int("".join(filter(str.isdigit, confidence_response.text)) or 50)
        confidence_score = max(0, min(100, confidence_score))
    except Exception as exc:
        logger.warning("Confidence evaluation unavailable: %s", exc)
        return {
            "trigger": False,
            "reason": "confidence evaluation unavailable",
            "confidence": 50,
        }

    if confidence_score < CONFIDENCE_THRESHOLD:
        return {
            "trigger": True,
            "reason": f"LLM confidence below {CONFIDENCE_THRESHOLD} (score: {confidence_score})",
            "confidence": confidence_score
        }

    return {
        "trigger": False,
        "confidence": confidence_score
    }



async def evaluate_explicit_user_request(message: str) -> Dict[str, Any]:
    """LLM-based classifier for explicit user request or frustration."""
    # Use LLM to classify if user is asking for human help
    # 1. Get Settings.llm (raise ValueError if None)
    # 2. Prompt: Ask LLM to return "YES" or "NO" if user wants human help
    # 3. Check if "YES" in response.text.upper()
    # 4. Return {"trigger": bool, "reason": str}
    try:
        llm = Settings.llm
        if llm is None:
            raise ValueError("LLM is not configured in Settings.")

        user_request_prompt = f"Does the user want to speak with a human agent? Answer with 'YES' or 'NO'.\nMessage: {message}"
        user_request_response = await llm.acomplete(user_request_prompt)

        if "YES" in user_request_response.text.upper():
            return {"trigger": True, "reason": "explicit user request for human help"}
        return {"trigger": False, "reason": "no explicit user request for human help"}
    except Exception as exc:
        logger.warning("Explicit user request evaluation unavailable: %s", exc)
        return {
            "trigger": False,
            "reason": "explicit user request evaluation unavailable",
        }


def send_handoff_email(context: Dict[str, Any]):
    """Send a handoff email using SMTP configuration from environment variables.

    Required variables: SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD,
    APPLICATION_EMAIL, and SUPPORT_EMAIL.
    """
    # Implement email sending
    # 1. Check SMTP config (SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, EMAIL_FROM, EMAIL_TO)
    # 2. Build subject: "[HUMAN HANDOFF] Ref {reference_id} – {trigger_reason}"
    # 3. Build body with all context fields (reference_id, trace_id, timestamp, priority,
    #    trigger_reason, user_metadata, query_history, generated_answer, evaluation_scores,
    #    retrieved_chunks, conversation_flow)
    # 4. Create MIMEText message and set headers
    # 5. Send via SMTP with TLS (use try/except, log errors)
    smtp_config = {
        "SMTP_HOST": SMTP_HOST,
        "SMTP_USERNAME": SMTP_USERNAME,
        "SMTP_PASSWORD": SMTP_PASSWORD,
        "APPLICATION_EMAIL": EMAIL_FROM,
        "SUPPORT_EMAIL": EMAIL_TO,
    }
    missing_config = [name for name, value in smtp_config.items() if not value]
    if missing_config:
        logger.warning(
            "SMTP configuration is incomplete; missing %s. Cannot send handoff email.",
            ", ".join(missing_config),
        )
        return
    subject = f"[HUMAN HANDOFF] Ref {context.get('reference_id')} – {context.get('trigger_reason')}"
    body = f"""
A human handoff has been triggered.

Reference ID: {context['reference_id']}
Trace ID: {context['trace_id']}
Timestamp: {context['timestamp_utc']}
Priority: {context['priority']}

Reason for Handoff:
{context['trigger_reason']}

User Metadata:
Email: {context['user_metadata'].get('email')}
Session ID: {context['session_id']}

------------------------------------------------------------
User Query History:
------------------------------------------------------------
{context['query_history']}

------------------------------------------------------------
Generated Answer:
------------------------------------------------------------
{context['generated_answer']}

------------------------------------------------------------
Evaluation Scores:
------------------------------------------------------------
Faithfulness: {context['evaluation_scores']['faithfulness']}
Relevance: {context['evaluation_scores']['relevance']}
LLM Confidence: {context['evaluation_scores']['confidence']}

------------------------------------------------------------
Retrieved Chunks (Full Content):
------------------------------------------------------------
{context['retrieved_chunks']}

------------------------------------------------------------
Conversation Flow:
------------------------------------------------------------
{context['conversation_flow']}
"""

    msg = MIMEText(body)
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        logger.info("Human handoff email sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send handoff email: {e}")

