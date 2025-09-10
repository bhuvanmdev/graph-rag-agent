#!/usr/bin/env python3
"""
ticket_manager.py - Simple SQLite-based ticket management

This module provides a lightweight alternative to Neo4j for ticket storage
using SQLite for better simplicity and performance.
"""
import sqlite3
import json
import logging
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path

from models import TicketData, TicketClassification, RAGQueryResult

logger = logging.getLogger(__name__)


class SQLiteTicketManager:
    """Handles storage and retrieval of classified tickets using SQLite"""

    def __init__(self, db_path: str = "tickets.db"):
        """Initialize SQLite database connection"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._create_tables()

    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tickets (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    body TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS classifications (
                    ticket_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    reasoning TEXT,
                    use_rag BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ticket_id) REFERENCES tickets (id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS answers (
                    ticket_id TEXT PRIMARY KEY,
                    answer TEXT NOT NULL,
                    sources TEXT,  -- JSON string
                    chunks_used INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ticket_id) REFERENCES tickets (id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS human_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT NOT NULL,
                    original_question TEXT NOT NULL,
                    rag_answer TEXT,
                    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'pending',  -- pending, in_progress, completed
                    human_response TEXT,
                    responded_at TIMESTAMP,
                    FOREIGN KEY (ticket_id) REFERENCES tickets (id)
                )
            ''')

            # Create indexes for better performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_topic ON classifications(topic)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_use_rag ON classifications(use_rag)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON tickets(created_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_human_review_status ON human_reviews(status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_human_review_ticket ON human_reviews(ticket_id)')

    def store_ticket(self, ticket: TicketData, classification: TicketClassification, rag_answer: Optional[RAGQueryResult] = None):
        """
        Store a classified ticket in SQLite

        Args:
            ticket: Original ticket data
            classification: Classification results
            rag_answer: RAG answer if applicable
        """
        with sqlite3.connect(self.db_path) as conn:
            # Store ticket
            conn.execute('''
                INSERT OR REPLACE INTO tickets (id, subject, body, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (ticket.id, ticket.subject, ticket.body))

            # Store classification
            conn.execute('''
                INSERT OR REPLACE INTO classifications
                (ticket_id, topic, sentiment, priority, reasoning, use_rag)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                ticket.id,
                classification.topic.value,
                classification.sentiment.value,
                classification.priority.value,
                classification.reasoning,
                rag_answer is not None
            ))

            # Store RAG answer if provided
            if rag_answer:
                conn.execute('''
                    INSERT OR REPLACE INTO answers
                    (ticket_id, answer, sources, chunks_used)
                    VALUES (?, ?, ?, ?)
                ''', (
                    ticket.id,
                    rag_answer.answer,
                    json.dumps(rag_answer.sources),
                    rag_answer.chunks_used
                ))
            else:
                # Remove any existing answer if no RAG answer provided
                conn.execute('DELETE FROM answers WHERE ticket_id = ?', (ticket.id,))

            conn.commit()
            logger.info(f"Stored ticket {ticket.id}")

    def get_ticket_stats(self) -> Dict:
        """Get statistics about stored tickets"""
        with sqlite3.connect(self.db_path) as conn:
            # Get basic counts
            cursor = conn.execute('''
                SELECT
                    COUNT(*) as total_tickets,
                    COUNT(CASE WHEN use_rag = 1 THEN 1 END) as rag_tickets,
                    COUNT(CASE WHEN use_rag = 0 THEN 1 END) as human_tickets
                FROM classifications
            ''')

            row = cursor.fetchone()
            stats = {
                'total_tickets': row[0] if row[0] else 0,
                'rag_tickets': row[1] if row[1] else 0,
                'human_tickets': row[2] if row[2] else 0,
                'topics': {}
            }

            # Get topic distribution
            cursor = conn.execute('''
                SELECT topic, COUNT(*) as count
                FROM classifications
                GROUP BY topic
                ORDER BY count DESC
            ''')

            for row in cursor:
                stats['topics'][row[0]] = row[1]

            return stats

    def get_ticket(self, ticket_id: str) -> Optional[Dict]:
        """Get a specific ticket with its classification and answer"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT
                    t.id, t.subject, t.body, t.created_at, t.updated_at,
                    c.topic, c.sentiment, c.priority, c.reasoning, c.use_rag,
                    a.answer, a.sources, a.chunks_used
                FROM tickets t
                LEFT JOIN classifications c ON t.id = c.ticket_id
                LEFT JOIN answers a ON t.id = a.ticket_id
                WHERE t.id = ?
            ''', (ticket_id,))

            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'subject': row[1],
                    'body': row[2],
                    'created_at': row[3],
                    'updated_at': row[4],
                    'classification': {
                        'topic': row[5],
                        'sentiment': row[6],
                        'priority': row[7],
                        'reasoning': row[8],
                        'use_rag': bool(row[9])
                    } if row[5] else None,
                    'answer': {
                        'answer': row[10],
                        'sources': json.loads(row[11]) if row[11] else [],
                        'chunks_used': row[12]
                    } if row[10] else None
                }
            return None

    def get_recent_tickets(self, limit: int = 10) -> List[Dict]:
        """Get recent tickets"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT
                    t.id, t.subject, t.body, t.created_at,
                    c.topic, c.sentiment, c.priority, c.use_rag
                FROM tickets t
                LEFT JOIN classifications c ON t.id = c.ticket_id
                ORDER BY t.created_at DESC
                LIMIT ?
            ''', (limit,))

            tickets = []
            for row in cursor:
                tickets.append({
                    'id': row[0],
                    'subject': row[1],
                    'body': row[2],
                    'created_at': row[3],
                    'classification': {
                        'topic': row[4],
                        'sentiment': row[5],
                        'priority': row[6],
                        'use_rag': bool(row[7])
                    } if row[4] else None
                })
            return tickets

    def clear_all_tickets(self):
        """Clear all tickets (useful for testing)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM answers')
            conn.execute('DELETE FROM classifications')
            conn.execute('DELETE FROM tickets')
            conn.commit()
            logger.info("Cleared all tickets")

    def get_database_info(self) -> Dict:
        """Get database file information"""
        return {
            'db_path': str(self.db_path),
            'db_size_mb': round(self.db_path.stat().st_size / (1024 * 1024), 2) if self.db_path.exists() else 0,
            'exists': self.db_path.exists()
        }

    def request_human_review(self, ticket_id: str, original_question: str, rag_answer: str) -> bool:
        """
        Store a request for human review of a RAG answer
        
        Args:
            ticket_id: The ticket ID
            original_question: The original customer question
            rag_answer: The RAG-generated answer that needs review
            
        Returns:
            bool: True if request was stored successfully, False if already exists
        """
        with sqlite3.connect(self.db_path) as conn:
            # Check if human review already requested for this ticket
            cursor = conn.execute('''
                SELECT id FROM human_reviews 
                WHERE ticket_id = ? AND status = 'pending'
            ''', (ticket_id,))
            
            if cursor.fetchone():
                logger.warning(f"Human review already requested for ticket {ticket_id}")
                return False
            
            # Store the human review request
            conn.execute('''
                INSERT INTO human_reviews (ticket_id, original_question, rag_answer)
                VALUES (?, ?, ?)
            ''', (ticket_id, original_question, rag_answer))
            
            conn.commit()
            logger.info(f"Human review requested for ticket {ticket_id}")
            return True

    def check_human_review_status(self, ticket_id: str) -> Optional[str]:
        """
        Check if a human review has been requested for a ticket
        
        Args:
            ticket_id: The ticket ID to check
            
        Returns:
            str: Status of human review ('pending', 'in_progress', 'completed') or None if no review requested
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT status FROM human_reviews 
                WHERE ticket_id = ? 
                ORDER BY requested_at DESC LIMIT 1
            ''', (ticket_id,))
            
            row = cursor.fetchone()
            return row[0] if row else None

    def get_pending_human_reviews(self) -> List[Dict]:
        """Get all pending human review requests"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT 
                    hr.id, hr.ticket_id, hr.original_question, hr.rag_answer, 
                    hr.requested_at, t.subject, t.body
                FROM human_reviews hr
                JOIN tickets t ON hr.ticket_id = t.id
                WHERE hr.status = 'pending'
                ORDER BY hr.requested_at ASC
            ''')
            
            reviews = []
            for row in cursor:
                reviews.append({
                    'review_id': row[0],
                    'ticket_id': row[1],
                    'original_question': row[2],
                    'rag_answer': row[3],
                    'requested_at': row[4],
                    'ticket_subject': row[5],
                    'ticket_body': row[6]
                })
            return reviews

