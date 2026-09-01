from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional
from guardrails import Guard

# Try to import DetectPII, but make it optional
try:
    from guardrails.hub import DetectPII
    DETECT_PII_AVAILABLE = True
except (ImportError, AttributeError):
    logger = logging.getLogger(__name__)
    logger.warning("DetectPII not available in this version of guardrails-ai")
    DETECT_PII_AVAILABLE = False
    DetectPII = None

import time

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputDecision:
    allowed: bool
    reason: Optional[str] = None


@dataclass(frozen=True)
class OutputDecision:
    blocked: bool
    reason: Optional[str]
    sanitized_text: str
    pii_detected: bool
    pii_summaries: List[str]
    output_sanitized: bool


class GuardrailsValidator:
    """Guardrails using Guardrails Hub DetectPII validator for PII detection and redaction."""

    def __init__(self) -> None:
        """
        TODO: Initialize Guardrails Hub DetectPII validator
        
        Steps to implement:
        1. Initialize self._pii_guard to None
        2. Try to import Guard and DetectPII from guardrails
        3. Create a Guard instance and configure it with DetectPII
        4. Set up PII entities to detect (EMAIL_ADDRESS, PHONE_NUMBER, US_SSN, etc.)
        5. Handle exceptions gracefully - if initialization fails, keep _pii_guard as None
        """
        # TODO: Initialize self._pii_guard to None
        # Hint: Set self._pii_guard = None initially
        self._pii_guard = None
        
        # TODO: Try to import and initialize Guardrails Hub DetectPII validator
        # Hint: Import Guard from guardrails and DetectPII from guardrails.hub. Create a Guard instance using Guard().use(DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "US_PASSPORT", "US_DRIVER_LICENSE", "PERSON", "LOCATION", "DATE_TIME"], on_fail="fix")) to detect various PII entity types with automatic redaction enabled. Store the configured guard in self._pii_guard.
        if DETECT_PII_AVAILABLE and DetectPII is not None:
            try:
                self._pii_guard = Guard().use(DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "US_PASSPORT", "US_DRIVER_LICENSE", "PERSON", "LOCATION", "DATE_TIME"], on_fail="fix"))
            except Exception as e:
                logger.warning("Failed to initialize PII guard: %s", e)
                self._pii_guard = None
        else:
            logger.info("DetectPII not available - PII detection disabled")

    def decide_input(self, text: str) -> InputDecision:
        """
        TODO: Validate input - check if the input is valid
        
        Steps to implement:
        1. Check if the text is empty or only whitespace
        2. Return InputDecision with allowed=False and reason="Empty query" if invalid
        3. Return InputDecision with allowed=True and reason=None if valid
        """
        # TODO: Check if the input text is empty or only whitespace
        # Hint: Verify whether the text is empty or contains only whitespace characters using a condition like: if not text or not text.strip()
        if not text or not text.strip():
            # TODO: Return InputDecision(False, "Empty query") if input is invalid
            # Hint: If the input is invalid, return InputDecision(allowed=False, reason="Empty query") to indicate the query is not allowed with an appropriate reason message
            return InputDecision(allowed=False, reason="Empty query")
        
        # TODO: Return InputDecision(True, None) if input is valid
        # Hint: If the input is valid, return InputDecision(allowed=True, reason=None) to indicate the query is allowed with no blocking reason
        return InputDecision(allowed=True, reason=None)

    def validate_and_sanitize_output(self, text: str) -> OutputDecision:
        """
        TODO: Validate and sanitize output using Guardrails Hub DetectPII validator
        
        Steps to implement:
        1. Initialize tracking variables (pii_summaries, pii_detected, output_sanitized)
        2. Start with sanitized text equal to input text
        3. If _pii_guard is available, validate the text
        4. Check if PII was detected and redacted
        5. Update tracking variables accordingly
        6. Return OutputDecision with all the results
        """
        # TODO: Initialize tracking variables
        # Hint: Create variables: pii_summaries: List[str] = [], pii_detected: bool = False, output_sanitized: bool = False, and sanitized: str = text (start with the original input text)
        pii_summaries: List[str] = []
        pii_detected: bool = False
        output_sanitized: bool = False
        sanitized: str = text

        # TODO: Use Guardrails Hub DetectPII validator if available
        # Hint: Check if self._pii_guard is not None. If available, wrap in try/except block. Call self._pii_guard.validate(sanitized) to validate the text. Extract the validated output using result.validated_output if getattr(result, "validated_output", None) else sanitized. Compare the candidate with the original sanitized text. If different, set pii_detected=True, output_sanitized=True, 
        # and add "PII detected and redacted (DetectPII/Presidio)" to pii_summaries.
        #  Update sanitized to the candidate value. On exception, keep the original text (use pass).
        if self._pii_guard is not None:
            try:
                result = self._pii_guard.validate(sanitized)
                candidate = getattr(result, "validated_output", None)
                if candidate is not None and candidate != sanitized:
                    pii_detected = True
                    output_sanitized = True
                    pii_summaries.append("PII detected and redacted (DetectPII/Presidio)")
                    sanitized = candidate
            except Exception as e:
                logger.warning("Error occurred while validating PII: %s", e)

        # TODO: Return OutputDecision with all results
        # Hint: Return OutputDecision(blocked=False, reason=None, sanitized_text=sanitized, pii_detected=pii_detected, pii_summaries=pii_summaries, output_sanitized=output_sanitized) containing all the validation results
        
        return OutputDecision(blocked=False, reason=None, sanitized_text=sanitized, pii_detected=pii_detected, pii_summaries=pii_summaries, output_sanitized=output_sanitized)
