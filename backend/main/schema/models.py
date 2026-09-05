from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional
MAX_QUERY_LENGTH = 2000


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    user_email: str = Field(..., min_length=3, max_length=320, description="User email for human handoff follow-up")
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        """Validate and sanitize question input."""
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

    @field_validator("user_email")
    @classmethod
    def validate_user_email(cls, v: str):
        v = (v or "").strip()
        if "@" not in v or v.startswith("@") or v.endswith("@"):
            raise ValueError("Invalid user_email")
        return v


class NodeResponse(BaseModel):
    """Normalized retrieval node returned alongside the legacy source node shape."""

    text: str
    score: Optional[float] = None
    source: Optional[str] = None


class ValidationResults(BaseModel):
    input_allowed: bool
    input_block_reason: Optional[str] = None
    pii_detected: bool = False
    pii_summaries: List[str] = Field(default_factory=list)
    output_sanitized: bool = False
    output_blocked: bool = False
    output_block_reason: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    query: Optional[str] = None
    source_nodes: List[dict] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    tool_errors: List[Dict[str, str]] = Field(default_factory=list)
    routing: str = "agent"
    validation_results: Optional[ValidationResults] = None


class UploadResponse(BaseModel):
    """Response model for upload endpoint."""
    message: str
    documents_indexed: int
