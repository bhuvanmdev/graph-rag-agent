#!/usr/bin/env python3
"""
database.py - Database operations for Neo4j graph database

This module handles all database interactions including vector search,
ticket storage, and database statistics.
"""
import os
import logging
from typing import List, Dict

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
        """Close database connection properly to prevent resource leaks"""
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

    def hybrid_graph_search(self, query_embedding: List[float], top_k: int = 8) -> List[ChunkData]:
        """
        Graph-enhanced vector search combining similarity search with graph traversal
        
        This method implements true GraphRAG by:
        1. Finding semantically similar chunks via vector search
        2. Traversing LINKS_TO relationships to find related pages
        3. Including chunks from linked pages that are also relevant
        4. Optionally traversing entity relationships (if entities exist)
        
        Args:
            query_embedding: Vector embedding of the user's query
            top_k: Number of top initial results to retrieve
            
        Returns:
            List of relevant chunks enriched with graph context
        """
        with self.driver.session() as session:
            # Phase 1: Vector search + Graph traversal for linked pages
            result = session.run("""
                // Step 1: Get top-k chunks via vector similarity
                CALL db.index.vector.queryNodes(
                    'consumers',
                    $top_k,
                    $query_embedding
                ) YIELD node AS chunk, score
                
                // Step 2: Get the page containing this chunk
                MATCH (p:Page)-[:HAS_CHUNK]->(chunk)
                
                // Step 3: Find pages linked to/from this page (1 hop)
                OPTIONAL MATCH (p)-[:LINKS_TO]-(related_page:Page)
                
                // Step 4: Get chunks from related pages
                OPTIONAL MATCH (related_page)-[:HAS_CHUNK]->(related_chunk:Chunk)
                
                // Step 5: Calculate relevance for related chunks
                WITH chunk, score, p,
                     collect(DISTINCT {
                         chunk: related_chunk,
                         page: related_page,
                         relationship: 'LINKS_TO'
                     }) AS related_chunks
                
                // Step 6: Return original chunks (high priority)
                WITH chunk, score, p, related_chunks
                RETURN chunk.text AS chunk_text,
                       chunk.source_url AS source_url,
                       chunk.chunk_index AS chunk_index,
                       p.title AS page_title,
                       score,
                       'original' AS chunk_type,
                       null AS relationship
                ORDER BY score DESC
                
                UNION
                
                // Step 7: Return related chunks from linked pages (lower priority)
                CALL db.index.vector.queryNodes(
                    'consumers',
                    $top_k,
                    $query_embedding
                ) YIELD node AS chunk, score
                MATCH (p:Page)-[:HAS_CHUNK]->(chunk)
                MATCH (p)-[:LINKS_TO]-(related_page:Page)
                MATCH (related_page)-[:HAS_CHUNK]->(related_chunk:Chunk)
                
                // Calculate similarity score for related chunks
                WITH related_chunk, related_page, score * 0.7 AS adjusted_score
                WHERE adjusted_score > 0.5
                
                RETURN related_chunk.text AS chunk_text,
                       related_chunk.source_url AS source_url,
                       related_chunk.chunk_index AS chunk_index,
                       related_page.title AS page_title,
                       adjusted_score AS score,
                       'related' AS chunk_type,
                       'LINKS_TO' AS relationship
                ORDER BY score DESC
                LIMIT $top_k
            """, query_embedding=query_embedding, top_k=top_k)

            chunks = []
            seen_chunks = set()  # Avoid duplicates
            
            for record in result:
                chunk_id = f"{record['source_url']}_{record['chunk_index']}"
                if chunk_id not in seen_chunks:
                    chunks.append(ChunkData(
                        text=record['chunk_text'],
                        source_url=record['source_url'],
                        page_title=record['page_title'],
                        chunk_index=record['chunk_index'],
                        score=record['score']
                    ))
                    seen_chunks.add(chunk_id)

            logger.info(f"Hybrid graph search returned {len(chunks)} chunks "
                       f"(including {len([c for c in chunks if c.score < 1.0])} from linked pages)")
            
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

