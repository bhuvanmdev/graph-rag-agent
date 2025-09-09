#!/usr/bin/env python3
"""
models.py - Pydantic models and enums for the RAG application

This module contains all the data models and enumerations used throughout
the application for structured data handling.
"""
from typing import List, Dict, Optional
from enum import Enum
from pydantic import BaseModel, Field


# Enumeration classes for ticket classification
class TopicEnum(str, Enum):
    """Enumeration of ticket topic categories"""
    HOW_TO = "How-to"
    PRODUCT = "Product"
    CONNECTOR = "Connector"
    LINEAGE = "Lineage"
    API_SDK = "API/SDK"
    SSO = "SSO"
    GLOSSARY = "Glossary"
    BEST_PRACTICES = "Best practices"
    SENSITIVE_DATA = "Sensitive data"


class SentimentEnum(str, Enum):
    """Enumeration of sentiment categories"""
    FRUSTRATED = "Frustrated"
    CURIOUS = "Curious"
    ANGRY = "Angry"
    NEUTRAL = "Neutral"


class PriorityEnum(str, Enum):
    """Enumeration of priority levels"""
    HIGH = "P0 (High)"
    MEDIUM = "P1 (Medium)"
    LOW = "P2 (Low)"


# Pydantic models for structured responses
class TicketClassification(BaseModel):
    """Structured response model for ticket classification"""
    topic: TopicEnum = Field(description="The topic category that best describes the ticket")
    sentiment: SentimentEnum = Field(description="The emotional tone of the customer")
    priority: PriorityEnum = Field(description="The priority level based on urgency and impact")
    reasoning: str = Field(description="Brief explanation of the classification decisions")


class RAGResponse(BaseModel):
    """Structured response model for RAG answers"""
    answer: str = Field(description="The generated answer to the user's question")
    confidence: float = Field(description="Confidence score between 0 and 1")
    needs_human_review: bool = Field(description="Whether this answer should be reviewed by a human")
    reasoning: str = Field(description="Brief explanation of how the answer was derived")


class TicketData(BaseModel):
    """Model for ticket data structure"""
    id: str = Field(description="Unique ticket identifier")
    subject: str = Field(description="Ticket subject line")
    body: str = Field(description="Ticket body content")


class ChunkData(BaseModel):
    """Model for chunk data from vector search"""
    text: str
    source_url: str
    page_title: str
    chunk_index: int
    score: float


class RAGQueryResult(BaseModel):
    """Model for RAG query results"""
    answer: str
    sources: List[str]
    chunks_used: int
    relevance_scores: Optional[List[float]] = None
