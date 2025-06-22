"""
Simplified Pydantic models for API request/response validation
Only includes the models that are actually used
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's question in natural language"
    )
    
    @field_validator('text')
    @classmethod
    def clean_text(cls, v):
        """Remove extra whitespace"""
        return v.strip()


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str = Field(
        ...,
        description="AI-generated answer with Swiss number formatting"
    )
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Source documents used for the answer"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the answer"
    )
    timing: Optional[Dict[str, float]] = Field(
        default=None,
        description="Detailed timing information for performance analysis"
    )