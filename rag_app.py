#!/usr/bin/env python3
"""
rag_app.py - Main entry point for the RAG-based Customer Support Copilot

This application provides a user-friendly interface for querying the knowledge base
created by ingest.py. It uses retrieval-augmented generation (RAG) with Google's
Gemini LLM to answer questions based on the scraped website content.

Architecture:
1. User submits a query through Gradio interface
2. Query is embedded using the same model as ingestion
3. Vector similarity search finds relevant chunks in Neo4j
4. Retrieved context is sent to Gemini LLM for answer generation
5. Generated answer is displayed to the user
"""
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from database import Neo4jRetriever
from rag import GeminiRAG
from interface import GradioInterface
# from utils import setup_environment
from dotenv import load_dotenv

load_dotenv()

def main():
    """Main application entry point"""
    logger.info("Starting Customer Support Copilot - RAG System")

    try:
        # Initialize components
        logger.info("Initializing Neo4j retriever...")
        retriever = Neo4jRetriever()

        logger.info("Initializing Gemini RAG system...")
        rag_system = GeminiRAG(retriever)

        logger.info("Setting up Customer Support Copilot interface...")
        demo = GradioInterface(rag_system)

        # Launch the application
        demo.launch(
            share=False,  # Set to True to create a public link
            server_name='0.0.0.0',
            server_port=7860,
            show_error=True
        )

    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup
        try:
            if 'retriever' in locals():
                retriever.close()
        except:
            pass


if __name__ == "__main__":
    main()