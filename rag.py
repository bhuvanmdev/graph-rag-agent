#!/usr/bin/env python3
"""
rag.py - Retrieval-Augmented Generation with Google Gemini

This module implements the RAG pipeline using Google Gemini for answer generation
based on retrieved context from the Neo4j knowledge base.
"""
import os
import json
import logging
from typing import List, Dict, Tuple, Optional

import google.generativeai as genai

from database import Neo4jRetriever
from models import ChunkData, RAGQueryResult

logger = logging.getLogger(__name__)

SYS_INSTRUCTION = """You are a helpful AI assistant that provides detailed, actionable answers based on the provided context.

Key guidelines:
1. Extract and synthesize information from the context to provide comprehensive solutions
2. Include step-by-step procedures, code examples, or specific instructions when available
3. Provide specific details, examples, and actionable steps from the context
4. If the context contains multiple approaches, explain the different options
5. Only state information is not available if it truly cannot be found in the context
6. Cite sources naturally within your explanation using [Source 1](link), [Source 2](link) format

Provide a structured response with detailed answer."""

safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
response_schema = {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "The generated answer to the user's question"
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence score between 0 and 1"
                            },
                            "needs_human_review": {
                                "type": "boolean",
                                "description": "Whether this answer should be reviewed by a human"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of how the answer was derived"
                            }
                        },
                        "required": ["answer", "confidence", "needs_human_review", "reasoning"]
                    }

class GeminiRAG:
    """Handles RAG operations with Google Gemini"""

    def __init__(self, retriever: Neo4jRetriever):
        """Initialize Gemini and retriever"""
        self.retriever = retriever

        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)

        
        # Generation config for consistent outputs
        self.generation_config = genai.GenerationConfig(
            temperature=0.3,
            max_output_tokens=2048,
            response_mime_type="application/json",
            response_schema=response_schema
        )
        
        # Using gemini-flash for faster responses, can switch to gemini-2.5-pro for better quality
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp', system_instruction=SYS_INSTRUCTION, safety_settings=safety_settings, generation_config=self.generation_config,)


        logger.info("Gemini model initialized")

    def retrieve_context(self, query: str, top_k: int = 8) -> Tuple[str, List[ChunkData]]:
        """
        Retrieve relevant context for the query

        Returns:
            Tuple of (context_string, chunk_metadata)
        """
        try:
            # Generate query embedding
            logger.info(f"Generating embedding for query: {query[:100]}...")
            query_embedding = self.retriever.embed_query(query)

            # Perform vector search
            logger.info(f"Searching for top {top_k} relevant chunks...")
            chunks = self.retriever.vector_search(query_embedding, top_k)

            if not chunks:
                logger.warning("No relevant chunks found")
                return "", []

            # Assemble context from chunks with better structure
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                # Include relevance score to help LLM prioritize information
                score_indicator = "⭐" * min(5, max(1, int(chunk.score * 5))) if chunk.score else ""
                context_parts.append(
                    f"[Source {i}: {chunk.page_title} {score_indicator}]\n"
                    f"URL: {chunk.source_url}\n"
                    f"Content:\n{chunk.text}\n"
                )

            context = "\n" + "="*50 + "\n".join(context_parts)

            logger.info(f"Retrieved {len(chunks)} chunks for context")
            return context, chunks

        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            raise

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate answer using Gemini based on query and context with structured output

        Args:
            query: User's question
            context: Retrieved context from database

        Returns:
            Generated answer
        """

        # Combine query with context
        full_prompt = f"""CONTEXT:
{context}

{query}

Please provide a detailed answer based on the context above."""

        try:
            response = self.model.generate_content(full_prompt)

            # Check if response was blocked by safety filters
            # if response.candidates and len(response.candidates) > 0:
            #     candidate = response.candidates[0]
            #     if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 2:  # SAFETY
            #         logger.warning(f"RAG response blocked by safety filters for query: {query[:100]}...")
            #         # Return safe fallback answer
            #         return "I apologize, but I cannot provide an answer to this query due to content safety restrictions. Please rephrase your question or contact human support for assistance."

            # Parse the structured response
            rag_data = json.loads(response.text)

            # Return just the answer for backward compatibility
            # You could also return the full structured data if needed
            answer = rag_data['answer']
            print(answer,'******')
            # Log additional structured data for monitoring
            # note:- the ai based confidence is not used for the confidence score in the ui.
            logger.info(f"RAG Response - AI given Confidence: {rag_data.get('confidence', 'N/A')}, "
                       f"Needs Review: {rag_data.get('needs_human_review', 'N/A')}")

            logger.info("Successfully generated answer")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer with Gemini: {str(e)}")
            return f"I encountered an error while generating the answer: {str(e)}"

    def answer_query(self, query: str, top_k: int = 8) -> RAGQueryResult:
        """
        Complete RAG pipeline: retrieve context and generate answer

        Args:
            query: User's question
            top_k: Number of chunks to retrieve

        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Input validation
            if not query or not query.strip():
                return RAGQueryResult(
                    answer="Please provide a valid question.",
                    sources=[],
                    chunks_used=0
                )

            # Retrieve relevant context
            context, chunks = self.retrieve_context(query, top_k)

            if not context:
                return RAGQueryResult(
                    answer="I couldn't find any relevant information in the knowledge base to answer your question.",
                    sources=[],
                    chunks_used=0
                )

            # Generate answer
            answer = self.generate_answer(query, context)

            # Extract unique sources
            sources = list(set([chunk.source_url for chunk in chunks]))

            return RAGQueryResult(
                answer=answer,
                sources=sources,
                chunks_used=len(chunks),
                relevance_scores=[chunk.score for chunk in chunks]
            )

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            return RAGQueryResult(
                answer=f"An error occurred while processing your question: {str(e)}",
                sources=[],
                chunks_used=0
            )
