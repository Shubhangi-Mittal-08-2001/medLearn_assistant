from pydantic import BaseModel, Field, field_validator
from typing import List, Dict
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


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    source_nodes: List[dict] = []
    tools_used: List[str] = []
    tool_errors: List[Dict[str, str]] = []
    routing: str = "agent"


class UploadResponse(BaseModel):
    """Response model for upload endpoint."""
    message: str
    documents_indexed: int
