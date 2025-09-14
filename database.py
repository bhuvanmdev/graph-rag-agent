#!/usr/bin/env python3
"""
database.py - Database operations for Neo4j graph database

This module handles all database interactions including vector search,
ticket storage, and database statistics.
"""
import os
import json
import logging
from typing import List, Dict, Optional

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from models import ChunkData

logger = logging.getLogger(__name__)


class Neo4jRetriever:
    """Handles retrieval operations from Neo4j graph database"""

    def __init__(self):
        """Initialize Neo4j connection and embedding model"""
        # Database connection
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")

        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

        # Initialize embedding model (MUST be same as used in ingest.py)
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model loaded")

    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for user query"""
        try:
            embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise

    def vector_search(self, query_embedding: List[float], top_k: int = 8) -> List[ChunkData]:
        """
        Perform vector similarity search to find relevant chunks

        Args:
            query_embedding: Vector embedding of the user's query
            top_k: Number of top results to retrieve

        Returns:
            List of relevant chunks with metadata
        """
        with self.driver.session() as session:
            # Use Neo4j's vector similarity search
            result = session.run("""
                CALL db.index.vector.queryNodes(
                    'consumers',
                    $top_k,
                    $query_embedding
                ) YIELD node, score
                MATCH (p:Page)-[:HAS_CHUNK]->(node)
                RETURN node.text AS chunk_text,
                       node.source_url AS source_url,
                       node.chunk_index AS chunk_index,
                       p.title AS page_title,
                       score
                ORDER BY score DESC
            """, query_embedding=query_embedding, top_k=top_k)

            chunks = []
            for record in result:
                chunks.append(ChunkData(
                    text=record['chunk_text'],
                    source_url=record['source_url'],
                    page_title=record['page_title'],
                    chunk_index=record['chunk_index'],
                    score=record['score']
                ))

            return chunks

    def get_database_stats(self) -> Dict:
        """Get statistics about the database"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Page)
                WITH count(p) as page_count
                MATCH (c:Chunk)
                WITH page_count, count(c) as chunk_count
                MATCH (:Page)-[r:LINKS_TO]->(:Page)
                RETURN page_count, chunk_count, count(r) as link_count
            """)

            record = result.single()
            if record:
                return {
                    'pages': record['page_count'],
                    'chunks': record['chunk_count'],
                    'links': record['link_count']
                }
            return {'pages': 0, 'chunks': 0, 'links': 0}

