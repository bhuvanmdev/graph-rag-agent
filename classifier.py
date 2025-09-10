#!/usr/bin/env python3
"""
classifier.py - Ticket classification using Google Gemini

This module handles automatic classification of customer support tickets
into topics, sentiment, and priority levels using Google Gemini.
"""
import os
import json
import logging
from typing import Dict

import google.generativeai as genai

from models import TicketData, TicketClassification, TopicEnum, SentimentEnum, PriorityEnum

logger = logging.getLogger(__name__)


class TicketClassifier:
    """Handles ticket classification using Gemini"""

    def __init__(self):
        """Initialize Gemini for classification"""
        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)

        # Initialize Gemini model
        self.model = genai.GenerativeModel('gemini-1.5-flash')

        # Generation config for classification
        self.generation_config = genai.GenerationConfig(
            temperature=0.3,  # Lower temperature for more consistent classification
            top_p=0.95,
            top_k=40,
            max_output_tokens=512,
        )

        logger.info("Ticket classifier initialized")

    def classify_ticket(self, ticket: TicketData) -> TicketClassification:
        """
        Classify a single ticket into topic, sentiment, and priority using structured output

        Args:
            ticket: Dictionary with 'id', 'subject', 'body'

        Returns:
            Dictionary with classification results
        """

        # Construct classification prompt
        prompt = f"""You are an expert customer support ticket classifier for Atlan, a data catalog platform.

Analyze the following ticket and classify it into exactly one category for each classification type.

TICKET INFORMATION:
ID: {ticket.id}
Subject: {ticket.subject}
Body: {ticket.body}

CLASSIFICATION INSTRUCTIONS:

1. TOPIC TAGS:
   - How-to: Questions about using Atlan features, tutorials, guidance
   - Product: General product questions, feature requests, capabilities
   - Connector: Issues with data source connections, connectors, integrations
   - Lineage: Questions about data lineage, tracing data flow
   - API/SDK: Questions about APIs, SDKs, programmatic access
   - SSO: Single Sign-On, authentication, user management
   - Glossary: Business glossary, terms, metadata management
   - Best practices: Recommendations, optimization, governance
   - Sensitive data: PII, data security, privacy, compliance

2. SENTIMENT:
   - Frustrated: Shows signs of frustration, blocked work, urgency
   - Curious: Learning-oriented, exploratory questions
   - Angry: Strong negative emotions, complaints, dissatisfaction
   - Neutral: Professional, matter-of-fact tone

3. PRIORITY:
   - P0 (High): Critical issues, production blockers, urgent deadlines
   - P1 (Medium): Important but not blocking, business impact
   - P2 (Low): General questions, nice-to-have features

Provide a structured classification with reasoning for your decisions."""

        try:
            # Generate classification with structured output using manual schema
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=512,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "enum": ["How-to", "Product", "Connector", "Lineage", "API/SDK", "SSO", "Glossary", "Best practices", "Sensitive data"],
                                "description": "The topic category that best describes the ticket"
                            },
                            "sentiment": {
                                "type": "string",
                                "enum": ["Frustrated", "Curious", "Angry", "Neutral"],
                                "description": "The emotional tone of the customer"
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["P0 (High)", "P1 (Medium)", "P2 (Low)"],
                                "description": "The priority level based on urgency and impact"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of the classification decisions"
                            }
                        },
                        "required": ["topic", "sentiment", "priority", "reasoning"]
                    }
                )
            )

            # Check if response was blocked by safety filters
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 2:  # SAFETY
                    logger.warning(f"Response blocked by safety filters for ticket {ticket.id}")
                    # Return safe default classification
                    return TicketClassification(
                        topic=TopicEnum.PRODUCT,
                        sentiment=SentimentEnum.NEUTRAL,
                        priority=PriorityEnum.MEDIUM,
                        reasoning='Classification blocked by safety filters - using default values'
                    )

            # Parse the structured response
            classification_data = json.loads(response.text)

            # Convert string values back to enums
            topic_enum = TopicEnum(classification_data['topic'])
            sentiment_enum = SentimentEnum(classification_data['sentiment'])
            priority_enum = PriorityEnum(classification_data['priority'])

            # Return structured classification
            result = TicketClassification(
                topic=topic_enum,
                sentiment=sentiment_enum,
                priority=priority_enum,
                reasoning=classification_data['reasoning']
            )

            logger.info(f"Successfully classified ticket {ticket.id}")
            return result

        except Exception as e:
            logger.error(f"Error classifying ticket {ticket.id}: {str(e)}")
            # Return default classification on error
            return TicketClassification(
                topic=TopicEnum.PRODUCT,
                sentiment=SentimentEnum.NEUTRAL,
                priority=PriorityEnum.MEDIUM,
                reasoning=f'Classification failed: {str(e)}'
            )

    def should_use_rag(self, topic: TopicEnum) -> bool:
        """
        Determine if a topic should use RAG or be routed to human agents

        Args:
            topic: The classified topic

        Returns:
            Boolean indicating if RAG should be used
        """
        # rag_topics = [TopicEnum.HOW_TO, TopicEnum.PRODUCT, TopicEnum.BEST_PRACTICES,
        #              TopicEnum.API_SDK, TopicEnum.SSO]
        return 1  or topic in rag_topics
