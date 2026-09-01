from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

# Guardrails Hub validators are installed separately from ``guardrails-ai`` in
# some deployments.  A missing validator must not stop the API from starting.
try:
    from guardrails import Guard
    from guardrails.hub import DetectPII
    DETECT_PII_AVAILABLE = True
except (ImportError, AttributeError):
    DETECT_PII_AVAILABLE = False
    Guard = None
    DetectPII = None

logger = logging.getLogger(__name__)

_FALLBACK_PHI_PATTERNS = (
    ("email address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    ("phone number", re.compile(r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{2,4}\)?[ .-]?)\d{3,4}[ .-]\d{4}(?!\w)"), "[REDACTED_PHONE]"),
    ("Social Security number", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    ("payment card number", re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[REDACTED_CARD]"),
    ("medical record number", re.compile(r"\b(?:MRN|medical\s+record(?:\s+number)?|chart(?:\s+number)?)\s*[:#-]?\s*[A-Za-z0-9-]{4,}\b", re.I), "[REDACTED_MRN]"),
    ("health plan or insurance identifier", re.compile(r"\b(?:member\s*ID|subscriber\s*ID|insurance(?:\s+(?:ID|policy(?:\s+number)?))?|health\s*plan\s*ID)\s*[:#-]?\s*[A-Za-z0-9-]{4,}\b", re.I), "[REDACTED_HEALTH_PLAN_ID]"),
    ("patient account number", re.compile(r"\b(?:patient\s+)?account(?:\s+number|\s*#)?\s*[:#-]?\s*[A-Za-z0-9-]{4,}\b", re.I), "[REDACTED_ACCOUNT]"),
    ("date of birth", re.compile(r"\b(?:DOB|date\s+of\s+birth|birth\s*date)\s*[:#-]?\s*(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})", re.I), "[REDACTED_DOB]"),
    ("patient name", re.compile(r"\b(?:patient(?:\s+name)?|name)\s*[:#-]\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", re.I), "[REDACTED_PATIENT_NAME]"),
    ("street address", re.compile(r"\b(?:address|home\s+address)\s*[:#-]\s*\d{1,6}\s+[^,\n]{2,80}", re.I), "[REDACTED_ADDRESS]"),
    ("IP address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED_IP]"),
)

# This endpoint is read-only.  Detect both SQL syntax and plain-language
# requests so a destructive action never reaches a future tool integration.
_DESTRUCTIVE_DATABASE_REQUEST = re.compile(
    r"\b(?:"
    r"drop\s+(?:table|tables|database|schema)|"
    r"truncate\s+(?:table|tables)|"
    r"delete\s+(?:from\s+)?(?:the\s+|your\s+|our\s+|ur\s+)?(?:table|tables|database|records|rows|data)|"
    r"remove\s+(?:the\s+|your\s+|our\s+|ur\s+)?(?:table|tables|database|records|data)|"
    r"wipe\s+(?:the\s+|your\s+|our\s+|ur\s+)?(?:table|tables|database|data)|"
    r"destroy\s+(?:the\s+|your\s+|our\s+|ur\s+)?(?:table|tables|database|data)|"
    r"alter\s+table\s+.+?\s+drop\b"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class InputDecision:
    allowed: bool
    reason: Optional[str] = None
    sanitized_text: str = ""
    phi_detected: bool = False
    phi_summaries: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class OutputDecision:
    blocked: bool
    reason: Optional[str]
    sanitized_text: str
    pii_detected: bool
    pii_summaries: List[str]
    output_sanitized: bool


class GuardrailsValidator:
    """Protect direct PII and healthcare PHI in MedLearn requests and responses."""

    def __init__(self) -> None:
        """Configure PII redaction once for the lifetime of the application."""
        self._pii_guard = None
        if DETECT_PII_AVAILABLE and Guard is not None and DetectPII is not None:
            try:
                self._pii_guard = Guard().use(
                    DetectPII(
                        pii_entities=[
                            "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN",
                            "CREDIT_CARD", "US_PASSPORT", "US_DRIVER_LICENSE",
                            "PERSON", "LOCATION", "DATE_TIME",
                        ],
                        on_fail="fix",
                    )
                )
            except Exception:
                logger.warning("Failed to initialize DetectPII; fallback PHI redaction remains active", exc_info=True)
        else:
            logger.warning("DetectPII is unavailable; using fallback PHI redaction")

    def decide_input(self, text: str) -> InputDecision:
        """Reject malformed or unsafe input and redact PHI before processing it."""
        if not text or not text.strip():
            return InputDecision(allowed=False, reason="Empty query", sanitized_text="")
        if _DESTRUCTIVE_DATABASE_REQUEST.search(text):
            return InputDecision(
                allowed=False,
                reason="Destructive database operations are not allowed on this read-only query endpoint.",
                sanitized_text="",
            )
        sanitized, summaries = self._fallback_redact(text.strip())
        return InputDecision(
            allowed=True,
            sanitized_text=sanitized,
            phi_detected=bool(summaries),
            phi_summaries=summaries,
        )

    @staticmethod
    def _fallback_redact(text: str) -> tuple[str, List[str]]:
        """Redact direct HIPAA identifiers when the optional Hub validator is unavailable."""
        redacted = text
        summaries: List[str] = []
        for label, pattern, replacement in _FALLBACK_PHI_PATTERNS:
            redacted, count = pattern.subn(replacement, redacted)
            if count:
                summaries.append(f"{label} redacted (fallback pattern)")
        return redacted, summaries

    def validate_and_sanitize_output(self, text: str) -> OutputDecision:
        """Redact PII/PHI from an answer and return auditable validation metadata."""
        pii_summaries: List[str] = []
        pii_detected: bool = False
        output_sanitized: bool = False
        sanitized: str = text

        if self._pii_guard is not None:
            try:
                result = self._pii_guard.validate(sanitized)
                candidate = getattr(result, "validated_output", result)
                if isinstance(candidate, str) and candidate != sanitized:
                    pii_detected = True
                    output_sanitized = True
                    pii_summaries.append("PII detected and redacted (DetectPII/Presidio)")
                    sanitized = candidate
            except Exception:
                # Failing open here keeps the RAG service available; the failure is
                # logged so it can be monitored and fixed.
                logger.warning("PII output validation failed", exc_info=True)

        if not output_sanitized:
            fallback_text, fallback_summaries = self._fallback_redact(sanitized)
            if fallback_summaries:
                sanitized = fallback_text
                pii_detected = True
                output_sanitized = True
                pii_summaries.extend(fallback_summaries)

        return OutputDecision(blocked=False, reason=None, sanitized_text=sanitized, pii_detected=pii_detected, pii_summaries=pii_summaries, output_sanitized=output_sanitized)
