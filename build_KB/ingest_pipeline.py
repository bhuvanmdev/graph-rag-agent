#!/usr/bin/env python3
"""
ingest.py - Web Scraping and Graph Database Ingestion Pipeline

This script recursively scrapes a website starting from a root URL,
processes the content into chunks with embeddings, and stores everything
in a Neo4j graph database for use as a RAG knowledge base.

Database Choice: Neo4j
- Native graph database with excellent Python support
- Built-in vector indexing capabilities (Neo4j 5.11+)
- ACID compliance for data integrity
- Cypher query language for complex graph traversals
- Scalable for large knowledge graphs
"""

import os
import sys
import asyncio
import hashlib
from collections import deque
from typing import List, Set, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse
import argparse
import logging
from datetime import datetime

# Web scraping
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy, NoExtractionStrategy

# Text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from bs4 import BeautifulSoup
import html2text

# Embeddings
from sentence_transformers import SentenceTransformer

# Database
from neo4j import GraphDatabase
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WebScraper:
    """Handles web scraping operations using crawl4ai"""
    
    def __init__(self, root_url: str, max_pages: int):
        self.root_url = root_url
        self.max_pages = max_pages
        self.visited_urls: Set[str] = set()
        self.url_queue: deque = deque([root_url])
        self.scraped_data: List[Dict] = []
        
        # Initialize HTML to Markdown converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = False
        self.html_converter.body_width = 0  # Don't wrap text
        
    async def _scrape_with_retry(self, crawler, url: str, max_retries: int = 3):
        """
        Attempt to scrape a URL with retry logic
        Returns the result if successful, None if all retries fail
        """
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                result = await crawler.arun(
                    url=url,
                    extraction_strategy=NoExtractionStrategy(),
                    bypass_cache=True
                )
                
                if result.success:
                    return result
                else:
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {url}, retrying...")
                        await asyncio.sleep(1)  # Brief delay before retry
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {url}")
                        return None
                        
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed for {url} with error: {str(e)}, retrying...")
                    await asyncio.sleep(1)  # Brief delay before retry
                else:
                    logger.error(f"All {max_retries + 1} attempts failed for {url} with final error: {str(e)}")
                    return None
        
        return None
        
    async def scrape(self) -> List[Dict]:
        """
        Perform BFS web scraping starting from root_url
        Returns list of scraped page data
        """
        async with AsyncWebCrawler(verbose=False) as crawler:
            pages_scraped = 0
            
            while self.url_queue and pages_scraped < self.max_pages:
                current_url = self.url_queue.popleft()
                
                # Skip if already visited
                if current_url in self.visited_urls:
                    continue
                    
                # Only crawl URLs within the root domain
                if not current_url.startswith(self.root_url):
                    continue
                
                try:
                    logger.info(f"Scraping: {current_url}")
                    self.visited_urls.add(current_url)
                    
                    # Scrape the page with retry logic
                    result = await self._scrape_with_retry(crawler, current_url, max_retries=3)
                    
                    if result and result.success:
                        # Convert HTML to clean Markdown
                        markdown_content = self._html_to_markdown(result.html)
                        
                        # Extract title
                        soup = BeautifulSoup(result.html, 'html.parser')
                        title = soup.find('title').text if soup.find('title') else 'Untitled'
                        
                        # Extract links for BFS traversal
                        links = self._extract_links(result.html, current_url)
                        
                        # Add new links to queue
                        for link in links:
                            if link not in self.visited_urls and link.startswith(self.root_url):
                                self.url_queue.append(link)
                        
                        # Store scraped data
                        page_data = {
                            'url': current_url,
                            'title': title.strip(),
                            'markdown_content': markdown_content,
                            'links': links,
                            'scraped_at': datetime.now().isoformat()
                        }
                        self.scraped_data.append(page_data)
                        pages_scraped += 1
                        
                        logger.info(f"Successfully scraped: {current_url} ({pages_scraped}/{self.max_pages})")
                    else:
                        logger.warning(f"Failed to scrape after all retry attempts: {current_url}")
                        
                except Exception as e:
                    logger.error(f"Error scraping {current_url}: {str(e)}")
                    continue
            
        logger.info(f"Scraping complete. Total pages scraped: {len(self.scraped_data)}")
        return self.scraped_data
    
    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML content to clean Markdown format"""
        try:
            # Remove script and style elements
            soup = BeautifulSoup(html, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Convert to markdown
            markdown = self.html_converter.handle(str(soup))
            
            # Clean up excessive whitespace
            lines = markdown.split('\n')
            cleaned_lines = [line.strip() for line in lines if line.strip()]
            markdown = '\n\n'.join(cleaned_lines)
            
            return markdown
        except Exception as e:
            logger.error(f"Error converting HTML to Markdown: {str(e)}")
            return ""
    
    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract all links from HTML content"""
        links = []
        try:
            soup = BeautifulSoup(html, 'html.parser')
            for link in soup.find_all('a', href=True):
                absolute_url = urljoin(base_url, link['href'])
                # Remove fragments and query parameters for cleaner URLs
                parsed = urlparse(absolute_url)
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                links.append(clean_url)
        except Exception as e:
            logger.error(f"Error extracting links: {str(e)}")
        
        return list(set(links))  # Remove duplicates


class ContentProcessor:
    """Handles content chunking and embedding generation"""
    
    def __init__(self):
        """
        Initialize content processor with embedding model
        
        Embedding Model Choice: all-MiniLM-L6-v2
        - Excellent balance between quality and speed
        - 384-dimensional embeddings (compact but effective)
        - Trained on over 1 billion sentence pairs
        - Optimized for semantic similarity tasks
        - Fast inference suitable for real-time RAG applications
        """
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize text splitters
        # Use Markdown-aware splitting for better semantic chunks
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                # ("###", "Header 3"),
            ]
        )
        self.min_chunk_size = 1500
        # Fallback recursive splitter for content without clear headers
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.min_chunk_size,  # Optimal size for context window
            chunk_overlap=50,  # Maintain context between chunks
        )
    
    def chunk_content(self, markdown_content: str, url: str) -> List[Dict]:
        """
        Split markdown content into semantic chunks
        Returns list of chunk dictionaries with text and metadata
        """
        chunks = []
        
        try:
            # First try markdown-aware splitting
            md_chunks = self.markdown_splitter.split_text(markdown_content)
            
            # Further split large chunks if necessary
            for md_chunk in md_chunks:
                if len(md_chunk.page_content) > self.min_chunk_size:  # Use configurable size
                    # Split large chunks recursively
                    sub_chunks = self.text_splitter.split_text(md_chunk.page_content)
                    for i, sub_chunk in enumerate(sub_chunks):
                        chunk_data = {
                            'text': sub_chunk,
                            'metadata': {
                                'source_url': url,
                                'chunk_index': len(chunks),
                                **md_chunk.metadata  # Include header metadata
                            }
                        }
                        chunks.append(chunk_data)
                else:
                    chunk_data = {
                        'text': md_chunk.page_content,
                        'metadata': {
                            'source_url': url,
                            'chunk_index': len(chunks),
                            **md_chunk.metadata
                        }
                    }
                    chunks.append(chunk_data)
            
            # If no markdown headers found, fall back to recursive splitting
            if not chunks:
                text_chunks = self.text_splitter.split_text(markdown_content)
                for i, text_chunk in enumerate(text_chunks):
                    chunk_data = {
                        'text': text_chunk,
                        'metadata': {
                            'source_url': url,
                            'chunk_index': i
                        }
                    }
                    chunks.append(chunk_data)
            
            # Combine small chunks to reach minimum size
            if len(chunks) > 1:
                combined_chunks = []
                i = 0
                while i < len(chunks):
                    current_chunk = chunks[i]
                    if len(current_chunk['text']) >= self.min_chunk_size:
                        combined_chunks.append(current_chunk)
                        i += 1
                    else:
                        # Start combining consecutive small chunks
                        combined_text = current_chunk['text']
                        combined_metadata = current_chunk['metadata'].copy()
                        i += 1
                        while i < len(chunks) and len(combined_text) < self.min_chunk_size:
                            combined_text += '\n\n' + chunks[i]['text']
                            i += 1
                        # Create combined chunk
                        combined_chunk = {
                            'text': combined_text,
                            'metadata': combined_metadata
                        }
                        combined_chunks.append(combined_chunk)
                
                # Update chunk indices
                for idx, chunk in enumerate(combined_chunks):
                    chunk['metadata']['chunk_index'] = idx
                
                chunks = combined_chunks
            
        except Exception as e:
            logger.error(f"Error chunking content for {url}: {str(e)}")
            # Fallback to single chunk
            chunks = [{
                'text': markdown_content[:self.min_chunk_size * 2],  # Limit size
                'metadata': {'source_url': url, 'chunk_index': 0}
            }]
        
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []


class Neo4jGraphDB:
    """Handles Neo4j database operations"""
    
    def __init__(self):
        """Initialize Neo4j connection"""
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
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
    
    def clear_database(self):
        """Clear existing data (use with caution)"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")
    
    def create_constraints_and_indexes(self):
        """Create necessary constraints and indexes"""
        with self.driver.session() as session:
            # Create unique constraint on Page URL
            try:
                session.run("""
                    CREATE CONSTRAINT page_url_unique IF NOT EXISTS
                    FOR (p:Page) REQUIRE p.url IS UNIQUE
                """)
            except:
                pass  # Constraint might already exist
            
            # Create vector index for chunk embeddings
            try:
                session.run("""
                    CREATE VECTOR INDEX consumers IF NOT EXISTS
                    FOR (c:Chunk) ON (c.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)
                logger.info("Created vector index for chunk embeddings")
            except Exception as e:
                logger.warning(f"Vector index might already exist: {str(e)}")
    
    def store_page(self, page_data: Dict) -> str:
        """Store a page node in the database"""
        with self.driver.session() as session:
            result = session.run("""
                MERGE (p:Page {url: $url})
                SET p.title = $title,
                    p.markdown_content = $markdown_content,
                    p.scraped_at = $scraped_at
                RETURN id(p) as page_id
            """, **page_data)
            
            page_id = result.single()['page_id']
            return str(page_id)
    
    def store_chunk(self, chunk_data: Dict, page_url: str, embedding: List[float]):
        """Store a chunk node with its embedding"""
        with self.driver.session() as session:
            # Create unique chunk ID
            chunk_id = hashlib.md5(
                f"{page_url}_{chunk_data['metadata']['chunk_index']}".encode()
            ).hexdigest()
            
            session.run("""
                MATCH (p:Page {url: $page_url})
                CREATE (c:Chunk {
                    id: $chunk_id,
                    text: $text,
                    chunk_index: $chunk_index,
                    source_url: $source_url,
                    embedding: $embedding
                })
                CREATE (p)-[:HAS_CHUNK]->(c)
            """, 
                page_url=page_url,
                chunk_id=chunk_id,
                text=chunk_data['text'],
                chunk_index=chunk_data['metadata']['chunk_index'],
                source_url=chunk_data['metadata']['source_url'],
                embedding=embedding
            )
    
    def create_page_links(self, source_url: str, target_urls: List[str]):
        """Create LINKS_TO relationships between pages"""
        with self.driver.session() as session:
            for target_url in target_urls:
                try:
                    session.run("""
                        MATCH (source:Page {url: $source_url})
                        MATCH (target:Page {url: $target_url})
                        MERGE (source)-[:LINKS_TO]->(target)
                    """, source_url=source_url, target_url=target_url)
                except:
                    pass  # Target page might not exist yet


async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Web scraping and database ingestion pipeline')
    parser.add_argument('--root-url', type=str, default='https://gemini.google.com/faq',
                        help='Starting URL for web scraping')
    parser.add_argument('--max-pages', type=int, default=10,
                        help='Maximum number of pages to scrape')
    parser.add_argument('--clear-db', action='store_true',
                        help='Clear existing database before ingestion')
    
    args = parser.parse_args()
    
    logger.info(f"Starting ingestion pipeline...")
    logger.info(f"Root URL: {args.root_url}")
    logger.info(f"Max pages: {args.max_pages}")
    
    # Initialize components
    scraper = WebScraper(args.root_url, args.max_pages)
    processor = ContentProcessor()
    db = Neo4jGraphDB()
    
    try:
        # Optionally clear database
        if args.clear_db:
            db.clear_database()
        
        # Create database schema
        db.create_constraints_and_indexes()
        
        # Step 1: Scrape websites
        logger.info("Starting web scraping...")
        scraped_data = await scraper.scrape()
        
        if not scraped_data:
            logger.error("No data scraped. Exiting.")
            return
        
        # Step 2: Process and store data
        logger.info("Processing and storing data...")
        total_chunks = 0
        
        for i, page_data in enumerate(scraped_data, 1):
            logger.info(f"Processing page {i}/{len(scraped_data)}: {page_data['url']}")
            
            # Store page node
            page_id = db.store_page({
                'url': page_data['url'],
                'title': page_data['title'],
                'markdown_content': page_data['markdown_content'],
                'scraped_at': page_data['scraped_at']
            })
            
            # Chunk content
            chunks = processor.chunk_content(
                page_data['markdown_content'],
                page_data['url']
            )
            
            # Generate embeddings for chunks
            chunk_texts = [chunk['text'] for chunk in chunks]
            if chunk_texts:
                embeddings = processor.generate_embeddings(chunk_texts)
                
                # Store chunks with embeddings
                for chunk, embedding in zip(chunks, embeddings):
                    db.store_chunk(chunk, page_data['url'], embedding)
                    total_chunks += 1
            
            logger.info(f"Stored {len(chunks)} chunks for {page_data['url']}")
        
        # Step 3: Create page relationships
        logger.info("Creating page relationships...")
        for page_data in scraped_data:
            # Only create links to pages we've actually scraped
            valid_links = [link for link in page_data['links'] 
                          if link in scraper.visited_urls]
            if valid_links:
                db.create_page_links(page_data['url'], valid_links)
        
        logger.info("=" * 50)
        logger.info("Ingestion pipeline completed successfully!")
        logger.info(f"Total pages stored: {len(scraped_data)}")
        logger.info(f"Total chunks created: {total_chunks}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())